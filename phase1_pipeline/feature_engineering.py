"""
Phase 1 — Feature Engineering
==============================
Computes all 5 core operational features:
  1. WIP Count              — active cases in queue
  2. Rolling Throughput     — completions/hr at 15/30/60-min windows
  3. Agent Utilization Rate — % of agents actively working
  4. Time Pressure Ratio    — fraction of WIP within 20% of SLA deadline
  5. Complexity-Weighted Backlog — WIP weighted by task complexity

Also computes derived features for ML:
  - Arrival rate trend (acceleration)
  - SLA breach velocity (rate of change in time_pressure)
  - Queue saturation index
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from phase1_pipeline.schemas import RawCaseEvent, FeatureSnapshot

logger = logging.getLogger(__name__)

COMPLEXITY_WEIGHTS = {"LOW": 0.7, "MEDIUM": 1.0, "HIGH": 1.8}

QUEUE_META = {
    "Document Review":    {"sla_hours": 8,  "max_agents": 12},
    "Compliance Check":   {"sla_hours": 4,  "max_agents": 8},
    "Payment Processing": {"sla_hours": 2,  "max_agents": 15},
    "Customer Onboarding":{"sla_hours": 24, "max_agents": 10},
    "Risk Assessment":    {"sla_hours": 12, "max_agents": 8},
    "Audit Preparation":  {"sla_hours": 48, "max_agents": 6},
}


class FeatureEngineer:
    """
    Computes feature snapshots from rolling event windows.
    Designed to be called by the Kafka consumer every 5 minutes.
    """

    def compute(
        self,
        queue_name: str,
        events_60: list[RawCaseEvent],
        events_30: list[RawCaseEvent],
        events_15: list[RawCaseEvent],
        snapshot_time: datetime,
    ) -> FeatureSnapshot:
        """
        Main entry point.
        events_Xm = all valid events in the last X minutes for this queue.
        """
        meta = QUEUE_META.get(queue_name, {"sla_hours": 24, "max_agents": 10})
        sla_hours  = meta["sla_hours"]
        max_agents = meta["max_agents"]

        # ── 1. WIP Count ──────────────────────────────────────────────────────
        arrived   = {e.case_id for e in events_60 if e.event_type.value == "ARRIVE"}
        completed = {e.case_id for e in events_60 if e.event_type.value == "COMPLETE"}
        wip_count = float(max(0, len(arrived - completed)))

        # ── 2. Rolling Throughput (completions per hour, extrapolated) ────────
        comp_60 = len([e for e in events_60 if e.event_type.value == "COMPLETE"])
        comp_30 = len([e for e in events_30 if e.event_type.value == "COMPLETE"])
        comp_15 = len([e for e in events_15 if e.event_type.value == "COMPLETE"])

        throughput_60m = float(comp_60)              # already per-hour
        throughput_30m = float(comp_30 * 2)          # extrapolate to /hr
        throughput_15m = float(comp_15 * 4)          # extrapolate to /hr

        # ── 3. Agent Utilization ──────────────────────────────────────────────
        active_starts = len([e for e in events_60 if e.event_type.value == "START"])
        utilization_rate = float(min(1.0, active_starts / max_agents))

        # ── 4. Time Pressure ─────────────────────────────────────────────────
        now_naive = snapshot_time.replace(tzinfo=None)
        sla_warning_window = timedelta(hours=sla_hours * 0.2)
        deadline_cutoff    = now_naive + sla_warning_window

        near_deadline = [
            e for e in events_60
            if e.event_type.value == "ARRIVE"
            and e.sla_deadline.replace(tzinfo=None) <= deadline_cutoff
            and e.sla_deadline.replace(tzinfo=None) >= now_naive
        ]
        time_pressure = float(min(1.0, len(near_deadline) / max(1.0, wip_count)))

        # ── 5. Complexity-Weighted Backlog ────────────────────────────────────
        backlog_events = [e for e in events_60 if e.event_type.value == "ARRIVE"]
        complexity_weights = [COMPLEXITY_WEIGHTS.get(e.complexity.value, 1.0) for e in backlog_events]
        complexity_backlog = float(np.sum(complexity_weights)) if complexity_weights else 0.0

        return FeatureSnapshot(
            snapshot_time    = snapshot_time,
            queue_name       = queue_name,
            wip_count        = wip_count,
            throughput_15m   = throughput_15m,
            throughput_30m   = throughput_30m,
            throughput_60m   = throughput_60m,
            utilization_rate = round(utilization_rate, 4),
            time_pressure    = round(time_pressure, 4),
            complexity_backlog = round(complexity_backlog, 2),
            sla_hours        = float(sla_hours),
            hour_of_day      = float(snapshot_time.hour),
            day_of_week      = float(snapshot_time.weekday()),
            is_weekend       = float(int(snapshot_time.weekday() >= 5)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame-level feature engineering (for CSV → ML training)
# ─────────────────────────────────────────────────────────────────────────────
def add_temporal_features(df) -> object:
    """Add lag and rolling features for walk-forward CV training."""
    import pandas as pd

    df = df.copy()
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"])
    df = df.sort_values(["queue_name", "snapshot_time"])

    for col in ["wip_count", "throughput_60m", "utilization_rate", "time_pressure"]:
        if col not in df.columns:
            continue
        grp = df.groupby("queue_name")[col]
        df[f"{col}_lag1"]    = grp.shift(1)
        df[f"{col}_lag3"]    = grp.shift(3)
        df[f"{col}_lag6"]    = grp.shift(6)
        df[f"{col}_roll_mean_6"]  = grp.transform(lambda x: x.rolling(6,  min_periods=1).mean())
        df[f"{col}_roll_std_6"]   = grp.transform(lambda x: x.rolling(6,  min_periods=1).std().fillna(0))
        df[f"{col}_roll_mean_12"] = grp.transform(lambda x: x.rolling(12, min_periods=1).mean())

    # Breach velocity — rate of change in time_pressure
    if "time_pressure" in df.columns:
        df["breach_velocity"] = df.groupby("queue_name")["time_pressure"].diff().fillna(0)

    # Queue saturation index
    if "wip_count" in df.columns and "throughput_60m" in df.columns:
        df["saturation_index"] = df["wip_count"] / (df["throughput_60m"].replace(0, 1))

    return df


def prepare_sequences(df, queue_name: str, seq_len: int = 12) -> tuple:
    """
    Prepare LSTM input sequences for a specific queue.
    Returns: (X_sequences, y_labels, timestamps)
    Shape: X → (N, seq_len, num_features)
    """
    import pandas as pd
    import numpy as np

    feature_cols = FeatureSnapshot.FEATURE_NAMES + [
        f"{c}_lag1" for c in ["wip_count", "throughput_60m", "utilization_rate"]
        if f"{c}_lag1" in df.columns
    ]

    q_df = df[df["queue_name"] == queue_name].dropna(subset=feature_cols).copy()
    X_list, y_list, t_list = [], [], []

    for i in range(seq_len, len(q_df)):
        seq = q_df.iloc[i - seq_len:i][feature_cols].values
        label = q_df.iloc[i]["breach_label"] if "breach_label" in q_df.columns else 0
        ts = q_df.iloc[i]["snapshot_time"]
        X_list.append(seq)
        y_list.append(label)
        t_list.append(ts)

    if not X_list:
        return np.array([]), np.array([]), []

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), t_list
