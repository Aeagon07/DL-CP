"""
Phase 2 — SHAP Explainability Engine
======================================
Generates human-readable explanations for every breach prediction.

Features:
  - SHAP TreeExplainer for XGBoost (exact, fast)
  - Top-3 contributing features per prediction
  - Plain-language narrative generation
  - Waterfall chart data (for dashboard rendering)
  - Batch explanation with caching

Example output:
  "Breach probability 87% — driven by:
   1. Review utilization 94% (+0.31 impact)
   2. WIP growth rate +22 cases/hr (+0.28 impact)
   3. 2.1h until SLA deadline (+0.19 impact)"
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

# Human-readable feature name mappings
FEATURE_DISPLAY_NAMES = {
    "wip_count":              "Active work-in-progress cases",
    "throughput_15m":         "15-min throughput (cases/hr)",
    "throughput_30m":         "30-min throughput (cases/hr)",
    "throughput_60m":         "60-min throughput (cases/hr)",
    "utilization_rate":       "Agent utilization rate",
    "time_pressure":          "SLA time pressure",
    "complexity_backlog":     "Complexity-weighted backlog",
    "sla_hours":              "SLA deadline window (hrs)",
    "hour_of_day":            "Hour of day",
    "day_of_week":            "Day of week",
    "is_weekend":             "Weekend indicator",
    "wip_count_lag1":         "WIP count 5 min ago",
    "wip_count_lag3":         "WIP count 15 min ago",
    "wip_count_lag6":         "WIP count 30 min ago",
    "throughput_60m_lag1":    "Throughput 5 min ago",
    "utilization_rate_lag1":  "Utilization 5 min ago",
    "time_pressure_lag1":     "Time pressure 5 min ago",
    "breach_velocity":        "Time pressure rate of change",
    "saturation_index":       "Queue saturation index",
    "lstm_forecast_1h":       "LSTM volume forecast (1h)",
    "lstm_forecast_2h":       "LSTM volume forecast (2h)",
    "lstm_forecast_4h":       "LSTM volume forecast (4h)",
}


def _display_name(feature: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(feature, feature.replace("_", " ").title())


def _format_value(feature: str, value: float) -> str:
    """Format feature value with appropriate units."""
    if "utilization" in feature or "pressure" in feature or "is_weekend" in feature:
        return f"{value * 100:.0f}%"
    elif "wip" in feature or "backlog" in feature:
        return f"{value:.0f} cases"
    elif "throughput" in feature or "forecast" in feature:
        return f"{value:.1f} cases/hr"
    elif "hour" in feature:
        return f"{int(value):02d}:00"
    elif "day" in feature:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        return days[int(value)] if 0 <= int(value) <= 6 else str(value)
    elif "sla_hours" in feature:
        return f"{value:.1f}h SLA"
    else:
        return f"{value:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Explainer
# ─────────────────────────────────────────────────────────────────────────────
class SHAPExplainer:
    """
    Wraps SHAP TreeExplainer for XGBoost breach classifiers.
    Returns structured explanation data for dashboard rendering.
    """

    def __init__(self, model, feature_names: list[str]):
        """
        model: trained XGBClassifier
        feature_names: list of feature column names in order
        """
        self.feature_names = feature_names
        try:
            self.explainer = shap.TreeExplainer(model)
            logger.info(f"SHAPExplainer initialised with {len(feature_names)} features")
        except Exception as e:
            logger.error(f"SHAP initialisation failed: {e}")
            self.explainer = None

    def explain(self, X: pd.DataFrame, top_k: int = 5) -> list[dict]:
        """
        Explain predictions for a DataFrame of feature rows.
        Returns list of explanation dicts (one per row).
        """
        if self.explainer is None:
            return [{}] * len(X)

        try:
            shap_values = self.explainer.shap_values(X)

            # For binary classification, shap_values may be a list [neg_class, pos_class]
            if isinstance(shap_values, list) and len(shap_values) == 2:
                sv = shap_values[1]  # positive class SHAP values
            else:
                sv = shap_values

            explanations = []
            for i, row_sv in enumerate(sv):
                exp = self._build_explanation(
                    shap_vals=row_sv,
                    feature_vals=X.iloc[i].values,
                    top_k=top_k,
                )
                explanations.append(exp)

            return explanations
        except Exception as e:
            logger.error(f"SHAP explain error: {e}")
            return [{}] * len(X)

    def explain_single(
        self,
        features: dict,
        breach_prob: float,
        top_k: int = 3,
    ) -> dict:
        """
        Explain a single prediction dict.
        Returns rich explanation with narrative text.
        """
        df = pd.DataFrame([features])[self.feature_names]
        explanations = self.explain(df, top_k=top_k)
        if not explanations:
            return {}

        exp = explanations[0]
        exp["breach_probability"] = breach_prob
        exp["narrative"] = self._build_narrative(exp, breach_prob)
        return exp

    def _build_explanation(
        self,
        shap_vals: np.ndarray,
        feature_vals: np.ndarray,
        top_k: int,
    ) -> dict:
        """Build structured explanation dict from SHAP values."""
        # Sort by absolute SHAP value
        abs_sv    = np.abs(shap_vals)
        top_idx   = np.argsort(abs_sv)[::-1][:top_k]

        factors = []
        for idx in top_idx:
            fname  = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            sv     = float(shap_vals[idx])
            fval   = float(feature_vals[idx])
            direction = "increases" if sv > 0 else "decreases"
            factors.append({
                "feature":      fname,
                "display_name": _display_name(fname),
                "value":        fval,
                "value_str":    _format_value(fname, fval),
                "shap_value":   round(sv, 4),
                "direction":    direction,
                "abs_impact":   round(abs(sv), 4),
            })

        return {
            "top_factors": factors,
            "all_shap_values": {
                self.feature_names[i]: round(float(shap_vals[i]), 4)
                for i in range(len(shap_vals))
                if i < len(self.feature_names)
            },
            "waterfall_data": self._build_waterfall(shap_vals, feature_vals),
        }

    def _build_waterfall(
        self,
        shap_vals: np.ndarray,
        feature_vals: np.ndarray,
        max_bars: int = 8,
    ) -> list[dict]:
        """Build data for waterfall chart rendering in the dashboard."""
        abs_sv  = np.abs(shap_vals)
        top_idx = np.argsort(abs_sv)[::-1][:max_bars]
        bars = []
        for r, idx in enumerate(top_idx):
            fname = self.feature_names[idx] if idx < len(self.feature_names) else f"f{idx}"
            bars.append({
                "rank":    r + 1,
                "feature": _display_name(fname),
                "value":   round(float(shap_vals[idx]), 4),
                "color":   "#ef4444" if shap_vals[idx] > 0 else "#22c55e",
            })
        return bars

    def _build_narrative(self, explanation: dict, breach_prob: float) -> str:
        """Generate plain-language explanation narrative."""
        factors = explanation.get("top_factors", [])
        if not factors:
            return f"Breach probability: {breach_prob * 100:.0f}%"

        level = (
            "CRITICAL" if breach_prob >= 0.85 else
            "HIGH"     if breach_prob >= 0.70 else
            "MEDIUM"   if breach_prob >= 0.50 else
            "LOW"
        )

        lines = [f"⚠ {level} breach probability {breach_prob * 100:.0f}% — top drivers:"]
        for i, f in enumerate(factors[:3], 1):
            arrow = "↑" if f["direction"] == "increases" else "↓"
            lines.append(
                f"  {i}. {f['display_name']} = {f['value_str']} "
                f"({arrow}{f['abs_impact']:+.2f} probability impact)"
            )

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Batch Explanation Cache
# ─────────────────────────────────────────────────────────────────────────────
class CachedSHAPExplainer:
    """
    Wraps SHAPExplainer with an LRU cache to avoid recomputing
    explanations for identical feature vectors.
    """

    def __init__(self, explainer: SHAPExplainer, cache_size: int = 256):
        self._exp = explainer
        from functools import lru_cache
        self._cache: dict = {}
        self._max = cache_size

    def explain_single(
        self,
        features: dict,
        breach_prob: float,
        top_k: int = 3,
    ) -> dict:
        # Create a hashable cache key
        key = hash(frozenset(
            (k, round(v, 4) if isinstance(v, float) else v)
            for k, v in features.items()
        ))
        if key in self._cache:
            return self._cache[key]

        result = self._exp.explain_single(features, breach_prob, top_k)

        if len(self._cache) >= self._max:
            # Evict oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        self._cache[key] = result
        return result
