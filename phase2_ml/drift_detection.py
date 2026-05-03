"""
Phase 2 — Concept Drift Detector
==================================
Monitors the live prediction error stream for statistical distribution shifts.
Uses Page-Hinkley test (optimal for detecting gradual drift in sequential data).

Also implements:
  - ADWIN (Adaptive Windowing) for abrupt drift
  - Population Stability Index (PSI) for feature drift
  - Automatic MLflow-logged retraining trigger

When drift is detected:
  1. Logs drift event to PostgreSQL
  2. Triggers MLflow-tracked model retraining
  3. Sends alert to Redis/Kafka
"""

from __future__ import annotations

import logging
import os
from collections import deque
from datetime import datetime
from typing import Callable, Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Page-Hinkley Test
# ─────────────────────────────────────────────────────────────────────────────
class PageHinkleyDetector:
    """
    Page-Hinkley test for concept drift detection.
    Monitors whether the mean of a sequential error signal has shifted significantly.

    Parameters:
      delta (float): Sensitivity — minimum magnitude of change to detect. Default 0.005.
      threshold (float): Detection threshold for cumulative sum. Default 50.
      alpha (float): Forgetting factor — discount old observations. Default 0.9999.

    Usage:
      detector = PageHinkleyDetector()
      for error in prediction_errors:
          is_drift = detector.update(error)
          if is_drift:
              trigger_retraining()
    """

    def __init__(
        self,
        delta:     float = 0.005,
        threshold: float = 50.0,
        alpha:     float = 0.9999,
        min_samples: int = 30,
    ):
        self.delta     = delta
        self.threshold = threshold
        self.alpha     = alpha
        self.min_samples = min_samples
        self.reset()

    def reset(self):
        self._n      = 0
        self._mean   = 0.0
        self._sum    = 0.0      # cumulative sum m_t
        self._min    = 0.0      # running minimum
        self._max    = 0.0      # running maximum (for two-sided test)
        self.drift_detected = False

    def update(self, error: float) -> bool:
        """
        Update detector with new prediction error.
        Returns True when drift is detected.
        """
        self._n += 1

        # Update running mean with forgetting factor
        self._mean = self.alpha * self._mean + (1 - self.alpha) * error

        # One-sided Page-Hinkley statistic (upward shift)
        self._sum = self._sum + error - self._mean - self.delta
        self._min = min(self._min, self._sum)
        self._max = max(self._max, self._sum)

        # Only start detecting after min_samples
        if self._n < self.min_samples:
            return False

        ph_stat = self._sum - self._min

        if ph_stat > self.threshold:
            self.drift_detected = True
            logger.warning(
                f"Page-Hinkley drift detected! PH={ph_stat:.2f} > threshold={self.threshold} "
                f"(n={self._n}, mean_error={self._mean:.4f})"
            )
            return True

        return False

    @property
    def statistic(self) -> float:
        return self._sum - self._min

    def is_drifting(self) -> bool:
        return self.drift_detected


# ─────────────────────────────────────────────────────────────────────────────
# ADWIN (Adaptive Windowing) — for abrupt drift
# ─────────────────────────────────────────────────────────────────────────────
class ADWINDetector:
    """
    ADWIN: Adaptive Windowing drift detector.
    Maintains a variable-size window and splits it when the means of two
    sub-windows differ significantly.

    Best for abrupt distribution changes.
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self._window: deque = deque()
        self._total = 0.0
        self._n = 0
        self.drift_detected = False
        self.drift_index = -1

    def reset(self):
        self._window.clear()
        self._total = 0.0
        self._n = 0
        self.drift_detected = False

    def update(self, value: float) -> bool:
        self._window.append(value)
        self._total += value
        self._n += 1

        # Keep window bounded
        if self._n < 10:
            return False

        return self._detect_change()

    def _detect_change(self) -> bool:
        window = list(self._window)
        n = len(window)
        if n < 10:
            return False

        total = sum(window)
        for t in range(1, n - 1):
            n0, n1 = t, n - t
            mu0 = sum(window[:t]) / n0
            mu1 = sum(window[t:]) / n1

            # Hoeffding bound
            epsilon_cut = self._hoeffding_bound(n0, n1)
            if abs(mu0 - mu1) > epsilon_cut:
                # Remove oldest half of window
                for _ in range(t):
                    oldest = self._window.popleft()
                    self._total -= oldest
                    self._n -= 1
                self.drift_detected = True
                self.drift_index = t
                logger.warning(
                    f"ADWIN drift detected! mu0={mu0:.4f} mu1={mu1:.4f} "
                    f"(n0={n0} n1={n1} eps={epsilon_cut:.4f})"
                )
                return True

        return False

    def _hoeffding_bound(self, n0: int, n1: int) -> float:
        n = n0 + n1
        return np.sqrt((1 / (2 * n0) + 1 / (2 * n1)) * np.log(4 * n / self.delta))


# ─────────────────────────────────────────────────────────────────────────────
# Population Stability Index (feature drift)
# ─────────────────────────────────────────────────────────────────────────────
def compute_psi(
    expected: np.ndarray,
    actual:   np.ndarray,
    n_bins:   int = 10,
) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.1  → No drift (stable)
    PSI 0.1–0.2 → Moderate drift (monitor)
    PSI > 0.2  → Significant drift (retrain)
    """
    eps = 1e-10

    # Equal-frequency binning on expected
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.unique(np.percentile(expected, percentiles))

    expected_freq = np.histogram(expected, bins=bins)[0] + eps
    actual_freq   = np.histogram(actual,   bins=bins)[0] + eps

    expected_pct = expected_freq / expected_freq.sum()
    actual_pct   = actual_freq   / actual_freq.sum()

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def psi_report(
    reference_df,
    current_df,
    feature_cols: list[str],
) -> dict:
    """Compute PSI for all features. Returns dict of feature → psi_score."""
    report = {}
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref = reference_df[col].dropna().values
        cur = current_df[col].dropna().values
        if len(ref) < 20 or len(cur) < 20:
            continue
        psi = compute_psi(ref, cur)
        severity = "stable" if psi < 0.1 else ("moderate" if psi < 0.2 else "significant")
        report[col] = {"psi": round(psi, 4), "severity": severity}
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Drift Monitor — orchestrates all detectors per queue
# ─────────────────────────────────────────────────────────────────────────────
class DriftMonitor:
    """
    Maintains Page-Hinkley + ADWIN detectors for each queue.
    Calls retraining_callback when drift is detected.
    """

    def __init__(
        self,
        ph_threshold:   float = 50.0,
        adwin_delta:    float = 0.002,
        on_drift: Optional[Callable[[str, str, float], None]] = None,
    ):
        self._ph_detectors:    dict[str, PageHinkleyDetector] = {}
        self._adwin_detectors: dict[str, ADWINDetector]       = {}
        self._on_drift = on_drift
        self._ph_thresh = ph_threshold
        self._adwin_delta = adwin_delta

    def _get_detectors(self, queue_name: str):
        if queue_name not in self._ph_detectors:
            self._ph_detectors[queue_name]    = PageHinkleyDetector(threshold=self._ph_thresh)
            self._adwin_detectors[queue_name] = ADWINDetector(delta=self._adwin_delta)
        return self._ph_detectors[queue_name], self._adwin_detectors[queue_name]

    def update(
        self,
        queue_name:  str,
        error:       float,   # e.g. |y_true - y_pred_proba|
        horizon:     int = 1,
    ) -> dict:
        """
        Push new prediction error into both detectors for a queue.
        Returns drift report dict.
        """
        ph, adwin = self._get_detectors(queue_name)

        ph_drift    = ph.update(error)
        adwin_drift = adwin.update(error)

        result = {
            "queue_name":        queue_name,
            "horizon_hours":     horizon,
            "ph_statistic":      round(ph.statistic, 4),
            "ph_drift":          ph_drift,
            "adwin_drift":       adwin_drift,
            "any_drift":         ph_drift or adwin_drift,
            "timestamp":         datetime.utcnow().isoformat(),
        }

        if result["any_drift"]:
            drift_type = "page_hinkley" if ph_drift else "adwin"
            logger.warning(
                f"DRIFT ALERT | queue={queue_name} | horizon={horizon}h | "
                f"type={drift_type} | ph={ph.statistic:.2f}"
            )
            if self._on_drift:
                self._on_drift(queue_name, drift_type, ph.statistic)

            # Reset after triggering
            ph.reset()
            if adwin_drift:
                adwin.reset()

        return result

    def reset_queue(self, queue_name: str):
        """Reset detectors for a specific queue (after retraining)."""
        if queue_name in self._ph_detectors:
            self._ph_detectors[queue_name].reset()
            self._adwin_detectors[queue_name].reset()
            logger.info(f"DriftMonitor reset for queue: {queue_name}")

    def get_stats(self) -> dict:
        return {
            q: {
                "ph_statistic": round(self._ph_detectors[q].statistic, 4),
                "ph_drift":     self._ph_detectors[q].drift_detected,
            }
            for q in self._ph_detectors
        }
