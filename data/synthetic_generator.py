"""
Appian Operations Center — Synthetic Data Generator
====================================================
Generates realistic synthetic operational data with:
  - Business-hour seasonality (morning surge, post-lunch dip)
  - Day-of-week patterns (Mon/Fri peaks)
  - Queue-specific SLA pressures
  - Correlated agent utilization
  - Controlled noise for drift simulation

Outputs:
  - data/sample_data.csv            : Historical feature dataset for ML training
  - Kafka stream (appian.events.raw): Live event JSON messages
"""

import os
import json
import time
import random
import logging
import argparse
import uuid
from datetime import datetime, timedelta
from typing import Generator

import numpy as np
import pandas as pd
from confluent_kafka import Producer
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
QUEUE_CONFIGS = {
    "Document Review":    {"sla_hours": 8,  "base_arrival": 12, "min_agents": 3,  "max_agents": 12},
    "Compliance Check":   {"sla_hours": 4,  "base_arrival": 8,  "min_agents": 2,  "max_agents": 8},
    "Payment Processing": {"sla_hours": 2,  "base_arrival": 20, "min_agents": 4,  "max_agents": 15},
    "Customer Onboarding":{"sla_hours": 24, "base_arrival": 6,  "min_agents": 2,  "max_agents": 10},
    "Risk Assessment":    {"sla_hours": 12, "base_arrival": 5,  "min_agents": 2,  "max_agents": 8},
    "Audit Preparation":  {"sla_hours": 48, "base_arrival": 3,  "min_agents": 1,  "max_agents": 6},
}

COMPLEXITY_WEIGHTS = {"LOW": 0.4, "MEDIUM": 0.35, "HIGH": 0.25}
COMPLEXITY_MULTIPLIERS = {"LOW": 0.7, "MEDIUM": 1.0, "HIGH": 1.8}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Seasonality Functions
# ─────────────────────────────────────────────────────────────────────────────
def business_hour_factor(hour: int) -> float:
    """Returns arrival rate multiplier based on hour of day."""
    if 0 <= hour < 7:
        return 0.05   # night — near zero
    elif 7 <= hour < 9:
        return 0.4    # pre-work ramp-up
    elif 9 <= hour < 11:
        return 1.4    # morning surge
    elif 11 <= hour < 13:
        return 1.1    # mid-morning stable
    elif 13 <= hour < 14:
        return 0.7    # lunch dip
    elif 14 <= hour < 16:
        return 1.2    # afternoon peak
    elif 16 <= hour < 18:
        return 0.9    # wind-down
    elif 18 <= hour < 20:
        return 0.3    # after-hours low
    else:
        return 0.05   # night


def day_of_week_factor(weekday: int) -> float:
    """Returns day-of-week multiplier (0=Monday, 6=Sunday)."""
    factors = {0: 1.3, 1: 1.0, 2: 1.0, 3: 1.1, 4: 1.4, 5: 0.3, 6: 0.05}
    return factors.get(weekday, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_features(
    events_df: pd.DataFrame,
    queue_name: str,
    snapshot_time: datetime,
    queue_config: dict,
) -> dict:
    """Compute all 5 core features for a queue at a given snapshot time."""
    sla_hours = queue_config["sla_hours"]
    max_agents = queue_config["max_agents"]

    # Filter events in last 60 min
    window_60 = snapshot_time - timedelta(hours=1)
    window_30 = snapshot_time - timedelta(minutes=30)
    window_15 = snapshot_time - timedelta(minutes=15)

    q_events = events_df[events_df["queue_name"] == queue_name]
    recent_60 = q_events[(q_events["timestamp"] >= window_60) & (q_events["timestamp"] <= snapshot_time)]
    recent_30 = q_events[(q_events["timestamp"] >= window_30) & (q_events["timestamp"] <= snapshot_time)]
    recent_15 = q_events[(q_events["timestamp"] >= window_15) & (q_events["timestamp"] <= snapshot_time)]

    # 1. WIP Count — cases in-flight (arrived but not completed)
    arrived = q_events[
        (q_events["event_type"] == "ARRIVE") &
        (q_events["timestamp"] <= snapshot_time)
    ]["case_id"].nunique()
    completed = q_events[
        (q_events["event_type"] == "COMPLETE") &
        (q_events["timestamp"] <= snapshot_time)
    ]["case_id"].nunique()
    wip_count = max(0, arrived - completed)

    # 2. Rolling throughput (cases completed per hour)
    throughput_15m = recent_15[recent_15["event_type"] == "COMPLETE"].shape[0] * 4  # extrapolated to /hr
    throughput_30m = recent_30[recent_30["event_type"] == "COMPLETE"].shape[0] * 2
    throughput_60m = recent_60[recent_60["event_type"] == "COMPLETE"].shape[0]

    # 3. Agent utilization
    active_cases = q_events[
        (q_events["event_type"] == "START") &
        (q_events["timestamp"] >= window_60) &
        (q_events["timestamp"] <= snapshot_time)
    ].shape[0]
    utilization_rate = min(1.0, active_cases / max_agents) if max_agents > 0 else 0.0

    # 4. Time pressure ratio — fraction of WIP within 20% of SLA deadline
    sla_window = timedelta(hours=sla_hours * 0.2)
    about_to_breach = q_events[
        (q_events["event_type"] == "ARRIVE") &
        (q_events["timestamp"] <= snapshot_time) &
        (q_events["sla_deadline"] <= snapshot_time + sla_window) &
        (q_events["sla_deadline"] >= snapshot_time)
    ].shape[0]
    time_pressure = min(1.0, about_to_breach / max(1, wip_count))

    # 5. Complexity-weighted backlog
    complexity_backlog = 0.0
    if not recent_60.empty and "complexity" in recent_60.columns:
        complexity_backlog = (
            recent_60[recent_60["event_type"] == "ARRIVE"]["complexity"]
            .apply(lambda x: COMPLEXITY_MULTIPLIERS.get(x, 1.0))
            .sum()
        )

    # SLA breach label — did any case breach SLA in this window?
    breached = q_events[
        (q_events["event_type"] == "ARRIVE") &
        (q_events["timestamp"] <= snapshot_time) &
        (q_events["sla_deadline"] < snapshot_time)
    ]
    completed_ids = q_events[
        (q_events["event_type"] == "COMPLETE") &
        (q_events["timestamp"] <= snapshot_time)
    ]["case_id"].unique()
    breach_count = breached[~breached["case_id"].isin(completed_ids)].shape[0]
    breach_label = int(breach_count > 0)

    return {
        "snapshot_time":   snapshot_time.isoformat(),
        "queue_name":      queue_name,
        "wip_count":       float(wip_count),
        "throughput_15m":  float(throughput_15m),
        "throughput_30m":  float(throughput_30m),
        "throughput_60m":  float(throughput_60m),
        "utilization_rate": float(round(utilization_rate, 4)),
        "time_pressure":   float(round(time_pressure, 4)),
        "complexity_backlog": float(round(complexity_backlog, 2)),
        "sla_hours":       float(sla_hours),
        "hour_of_day":     float(snapshot_time.hour),
        "day_of_week":     float(snapshot_time.weekday()),
        "is_weekend":      float(int(snapshot_time.weekday() >= 5)),
        "breach_label":    breach_label,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Event Generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_events(
    start_date: datetime,
    num_days: int,
    interval_minutes: int = 5,
) -> pd.DataFrame:
    """Generate a full historical events DataFrame."""
    logger.info(f"Generating {num_days} days of synthetic events …")
    events = []
    np.random.seed(42)

    for queue_name, config in QUEUE_CONFIGS.items():
        current_time = start_date
        end_time = start_date + timedelta(days=num_days)
        open_cases = {}   # case_id → (arrive_time, complexity, sla_deadline)

        while current_time < end_time:
            hour_factor = business_hour_factor(current_time.hour)
            day_factor = day_of_week_factor(current_time.weekday())

            # Arrivals — Poisson process
            lam = config["base_arrival"] * hour_factor * day_factor * (interval_minutes / 60)
            num_arrivals = np.random.poisson(max(0.01, lam))

            for _ in range(num_arrivals):
                case_id = str(uuid.uuid4())[:8].upper()
                complexity = np.random.choice(
                    list(COMPLEXITY_WEIGHTS.keys()),
                    p=list(COMPLEXITY_WEIGHTS.values())
                )
                sla_deadline = current_time + timedelta(hours=config["sla_hours"])
                open_cases[case_id] = (current_time, complexity, sla_deadline)
                events.append({
                    "event_id":    str(uuid.uuid4()),
                    "case_id":     case_id,
                    "queue_name":  queue_name,
                    "event_type":  "ARRIVE",
                    "timestamp":   current_time + timedelta(seconds=random.randint(0, 60)),
                    "complexity":  complexity,
                    "sla_deadline": sla_deadline,
                })

            # Completions — process random subset of open cases
            num_agents = np.random.randint(config["min_agents"], config["max_agents"] + 1)
            completable = list(open_cases.keys())[:num_agents]

            for case_id in completable:
                if random.random() < 0.3:   # 30% completion per interval
                    _, complexity, sla_deadline = open_cases.pop(case_id)
                    mult = COMPLEXITY_MULTIPLIERS.get(complexity, 1.0)
                    processing_mins = np.random.lognormal(
                        mean=np.log(30 * mult), sigma=0.5
                    )
                    complete_time = current_time + timedelta(minutes=min(processing_mins, interval_minutes))
                    events.append({
                        "event_id":    str(uuid.uuid4()),
                        "case_id":     case_id,
                        "queue_name":  queue_name,
                        "event_type":  "START",
                        "timestamp":   current_time,
                        "complexity":  complexity,
                        "sla_deadline": sla_deadline,
                    })
                    events.append({
                        "event_id":    str(uuid.uuid4()),
                        "case_id":     case_id,
                        "queue_name":  queue_name,
                        "event_type":  "COMPLETE",
                        "timestamp":   complete_time,
                        "complexity":  complexity,
                        "sla_deadline": sla_deadline,
                    })

            current_time += timedelta(minutes=interval_minutes)

    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["sla_deadline"] = pd.to_datetime(df["sla_deadline"])
    logger.info(f"Generated {len(df):,} raw events across {len(QUEUE_CONFIGS)} queues")
    return df.sort_values("timestamp").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# CSV Dataset Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_dataset(
    events_df: pd.DataFrame,
    interval_minutes: int = 5,
) -> pd.DataFrame:
    """Build a feature snapshot dataset for ML training."""
    logger.info("Computing feature snapshots …")
    records = []
    start = events_df["timestamp"].min().floor("H")
    end = events_df["timestamp"].max()
    ts = start

    while ts <= end:
        for queue_name, config in QUEUE_CONFIGS.items():
            feat = compute_features(events_df, queue_name, ts, config)
            records.append(feat)
        ts += timedelta(minutes=interval_minutes)

    df = pd.DataFrame(records)
    logger.info(f"Built {len(df):,} feature snapshots")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Kafka Live Stream Producer
# ─────────────────────────────────────────────────────────────────────────────
def kafka_delivery_report(err, msg):
    if err:
        logger.error(f"Kafka delivery failed: {err}")


def stream_to_kafka(interval_seconds: float = 1.0):
    """
    Continuously stream synthetic events to Kafka in real-time.
    Simulates live production data feed at configured rate.
    """
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_TOPIC_EVENTS", "appian.events.raw")

    producer = Producer({
        "bootstrap.servers": bootstrap_servers,
        "acks":              "all",
        "retries":           3,
        "linger.ms":         5,
        "compression.type":  "snappy",
    })

    logger.info(f"Starting Kafka live stream → {topic} @ {bootstrap_servers}")
    seq = 0

    while True:
        now = datetime.utcnow()
        hour_factor = business_hour_factor(now.hour)
        day_factor = day_of_week_factor(now.weekday())

        for queue_name, config in QUEUE_CONFIGS.items():
            lam = config["base_arrival"] * hour_factor * day_factor * (interval_seconds / 3600)
            num_arrivals = max(0, int(np.random.poisson(max(0.01, lam))))

            for _ in range(num_arrivals):
                complexity = np.random.choice(
                    list(COMPLEXITY_WEIGHTS.keys()),
                    p=list(COMPLEXITY_WEIGHTS.values())
                )
                event = {
                    "event_id":    str(uuid.uuid4()),
                    "seq":         seq,
                    "case_id":     str(uuid.uuid4())[:8].upper(),
                    "queue_name":  queue_name,
                    "event_type":  "ARRIVE",
                    "timestamp":   now.isoformat() + "Z",
                    "complexity":  complexity,
                    "sla_deadline": (now + timedelta(hours=config["sla_hours"])).isoformat() + "Z",
                    "source":      "synthetic_live",
                }
                producer.produce(
                    topic,
                    key=queue_name.encode(),
                    value=json.dumps(event).encode(),
                    callback=kafka_delivery_report,
                )
                seq += 1

        producer.poll(0)

        # Also publish features snapshot
        feat_topic = os.getenv("KAFKA_TOPIC_FEATURES", "appian.features.computed")
        for queue_name, config in QUEUE_CONFIGS.items():
            feat = {
                "snapshot_time":   now.isoformat() + "Z",
                "queue_name":      queue_name,
                "wip_count":       float(np.random.poisson(config["base_arrival"] * 2)),
                "throughput_15m":  float(np.random.lognormal(2.0, 0.3)),
                "throughput_30m":  float(np.random.lognormal(2.1, 0.3)),
                "throughput_60m":  float(np.random.lognormal(2.2, 0.3)),
                "utilization_rate": float(min(1.0, np.random.beta(5, 2) * hour_factor)),
                "time_pressure":   float(min(1.0, np.random.beta(2, 5))),
                "complexity_backlog": float(np.random.lognormal(3, 0.5)),
                "hour_of_day":     float(now.hour),
                "day_of_week":     float(now.weekday()),
            }
            producer.produce(
                feat_topic,
                key=queue_name.encode(),
                value=json.dumps(feat).encode(),
            )

        producer.flush()
        logger.info(f"[{seq}] Pushed events + features to Kafka (hour={now.hour})")
        time.sleep(interval_seconds)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Appian Synthetic Data Generator")
    parser.add_argument("--mode", choices=["csv", "stream", "both"], default="csv",
                        help="csv: generate training dataset | stream: Kafka live stream | both: do both")
    parser.add_argument("--days",      type=int,   default=90,  help="Historical days for CSV generation")
    parser.add_argument("--interval",  type=int,   default=5,   help="Snapshot interval in minutes")
    parser.add_argument("--stream-interval", type=float, default=2.0, help="Kafka push interval in seconds")
    parser.add_argument("--output",    type=str,   default="data/sample_data.csv")
    args = parser.parse_args()

    if args.mode in ("csv", "both"):
        start_date = datetime.utcnow() - timedelta(days=args.days)
        events_df = generate_events(start_date, args.days, args.interval)
        feature_df = build_feature_dataset(events_df, args.interval)
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        feature_df.to_csv(args.output, index=False)
        logger.info(f"✓ Saved training dataset → {args.output} ({len(feature_df):,} rows)")

        # Also save raw events
        raw_path = args.output.replace("sample_data.csv", "raw_events.csv")
        events_df.to_csv(raw_path, index=False)
        logger.info(f"✓ Saved raw events → {raw_path} ({len(events_df):,} rows)")

    if args.mode in ("stream", "both"):
        stream_to_kafka(interval_seconds=args.stream_interval)


if __name__ == "__main__":
    main()
