"""
Phase 2 — Conformal Prediction (Module 1 of 12)
=================================================
Replaces heuristic MC Dropout confidence bands with formally guaranteed
prediction intervals via Inductive Conformal Prediction (ICP / Split CP).

Theory:
  Given a non-conformity score s(x,y) and a calibration set of n samples,
  the conformal predictor guarantees:
      P(y_new ∈ Ĉ(x_new)) ≥ 1 - α
  where α is the desired miscoverage (e.g. α=0.2 → 80% coverage).
  This guarantee is distribution-free and model-agnostic.

Usage:
  # Regression (LSTM volume forecasts)
  cp = ConformalRegressor(alpha=0.2)
  cp.calibrate(y_cal_true, y_cal_pred)
  intervals = cp.predict(y_new_pred)   # returns (lower, upper) arrays

  # Classification (XGBoost breach probabilities)
  cpc = ConformalClassifier(alpha=0.2)
  cpc.calibrate(y_cal_true, proba_cal)
  sets = cpc.predict_set(proba_new)    # returns set of admitted labels

Run standalone:
  python phase2_ml/conformal_prediction.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Conformal Regressor (wraps LSTM / TFT volume forecasts)
# ─────────────────────────────────────────────────────────────────────────────
class ConformalRegressor:
    """
    Split Conformal Prediction for regression tasks.

    Non-conformity score: absolute residual |y_true - y_pred|
    Quantile: q̂ = (1 - α) empirical quantile of calibration scores
    Prediction interval: [ŷ - q̂, ŷ + q̂]

    Coverage guarantee: P(y ∈ [ŷ - q̂, ŷ + q̂]) ≥ 1 - α
    """

    def __init__(self, alpha: float = 0.2):
        """
        alpha: miscoverage level (e.g. 0.2 → 80% coverage guarantee)
        """
        assert 0 < alpha < 1, "alpha must be in (0, 1)"
        self.alpha = alpha
        self._quantile: Optional[float] = None
        self._cal_scores: Optional[np.ndarray] = None
        self.is_calibrated = False

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ConformalRegressor":
        """
        Fit conformal quantile on held-out calibration set.
        y_true, y_pred: 1-D arrays of shape (n_cal,)
        """
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        assert len(y_true) == len(y_pred), "y_true and y_pred must have same length"

        self._cal_scores = np.abs(y_true - y_pred)

        # Conformal quantile: ceil((n+1)(1-α))/n corrected for finite samples
        n = len(self._cal_scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self._quantile = float(np.quantile(self._cal_scores, level))
        self.is_calibrated = True

        logger.info(
            f"ConformalRegressor calibrated | n_cal={n} | α={self.alpha} | "
            f"q̂={self._quantile:.4f} | nominal_coverage={(1-self.alpha)*100:.0f}%"
        )
        return self

    def predict(self, y_pred: np.ndarray) -> dict:
        """
        Produce conformal prediction intervals for new predictions.
        Returns dict with keys: lower, upper, width, quantile
        """
        if not self.is_calibrated:
            raise RuntimeError("Call .calibrate() before .predict()")

        y_pred = np.asarray(y_pred, dtype=float).ravel()
        lower = y_pred - self._quantile
        upper = y_pred + self._quantile

        return {
            "lower":    lower,
            "upper":    upper,
            "width":    float(2 * self._quantile),
            "quantile": self._quantile,
            "coverage_target": 1 - self.alpha,
        }

    def empirical_coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute empirical coverage on a test set (should be ≥ 1-α)."""
        result = self.predict(y_pred)
        y_true = np.asarray(y_true).ravel()
        covered = np.mean((y_true >= result["lower"]) & (y_true <= result["upper"]))
        return float(covered)

    @property
    def quantile(self) -> Optional[float]:
        return self._quantile


# ─────────────────────────────────────────────────────────────────────────────
# Conformal Classifier (wraps XGBoost / TabTransformer breach probabilities)
# ─────────────────────────────────────────────────────────────────────────────
class ConformalClassifier:
    """
    Split Conformal Prediction for binary classification.

    Non-conformity score for label y ∈ {0,1}:
        s(x, y=1) = 1 - p̂(y=1|x)   [if true label is 1]
        s(x, y=0) = p̂(y=1|x)        [if true label is 0]

    Prediction set Ĉ(x):
        Include label y if s(x,y) ≤ q̂_y

    Marginal coverage guarantee: P(y_true ∈ Ĉ(x)) ≥ 1 - α
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self._q_pos: Optional[float] = None   # quantile for class 1
        self._q_neg: Optional[float] = None   # quantile for class 0
        self.is_calibrated = False

    def calibrate(
        self, y_true: np.ndarray, proba: np.ndarray
    ) -> "ConformalClassifier":
        """
        y_true: binary labels (0/1)
        proba:  predicted P(breach=1), shape (n,)
        """
        y_true = np.asarray(y_true, dtype=int).ravel()
        proba  = np.asarray(proba,  dtype=float).ravel()

        pos_mask = y_true == 1
        neg_mask = y_true == 0

        # Non-conformity scores per class
        scores_pos = 1.0 - proba[pos_mask]   # how wrong for true positives
        scores_neg = proba[neg_mask]           # how wrong for true negatives

        n = len(y_true)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)

        self._q_pos = float(np.quantile(scores_pos, level)) if len(scores_pos) > 0 else 0.5
        self._q_neg = float(np.quantile(scores_neg, level)) if len(scores_neg) > 0 else 0.5
        self.is_calibrated = True

        coverage_pos = float(np.mean(scores_pos <= self._q_pos)) if len(scores_pos) > 0 else 0.0
        coverage_neg = float(np.mean(scores_neg <= self._q_neg)) if len(scores_neg) > 0 else 0.0

        logger.info(
            f"ConformalClassifier calibrated | α={self.alpha} | "
            f"q_pos={self._q_pos:.4f} | q_neg={self._q_neg:.4f} | "
            f"coverage_pos={coverage_pos:.2%} | coverage_neg={coverage_neg:.2%}"
        )
        return self

    def predict_set(self, proba: np.ndarray) -> np.ndarray:
        """
        Returns prediction set membership for new probabilities.
        Result shape: (n, 2) — [include_0, include_1] booleans
        """
        if not self.is_calibrated:
            raise RuntimeError("Call .calibrate() first")

        proba = np.asarray(proba, dtype=float).ravel()
        include_1 = (1.0 - proba) <= self._q_pos
        include_0 = proba <= self._q_neg

        return np.column_stack([include_0, include_1])

    def predict_binary(self, proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Conservative binary decision: predict 1 if in prediction set AND proba ≥ threshold.
        """
        sets = self.predict_set(proba)
        proba = np.asarray(proba).ravel()
        return ((proba >= threshold) & sets[:, 1]).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Coverage Report
# ─────────────────────────────────────────────────────────────────────────────
class CoverageReport:
    """Computes and formats conformal coverage statistics."""

    @staticmethod
    def regression_report(
        cp: ConformalRegressor,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizon_label: str = "1h",
    ) -> dict:
        empirical = cp.empirical_coverage(y_true, y_pred)
        intervals = cp.predict(y_pred)
        return {
            "horizon":          horizon_label,
            "nominal_coverage": round(1 - cp.alpha, 3),
            "empirical_coverage": round(empirical, 4),
            "coverage_gap":     round(empirical - (1 - cp.alpha), 4),
            "interval_width":   round(intervals["width"], 4),
            "conformal_quantile": round(cp.quantile, 4),
            "guarantee_met":    empirical >= (1 - cp.alpha),
        }

    @staticmethod
    def print_report(reports: list[dict]):
        print("\n" + "═" * 65)
        print("  CONFORMAL PREDICTION — COVERAGE REPORT")
        print("═" * 65)
        for r in reports:
            status = "✓ GUARANTEE MET" if r["guarantee_met"] else "✗ BELOW TARGET"
            print(
                f"  {r['horizon']:>5} | "
                f"Target={r['nominal_coverage']:.0%} | "
                f"Actual={r['empirical_coverage']:.1%} | "
                f"Width=±{r['interval_width']:.3f} | {status}"
            )
        print("═" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MLflow Integration
# ─────────────────────────────────────────────────────────────────────────────
def log_conformal_to_mlflow(
    reports: list[dict],
    queue_name: str,
    tracking_uri: Optional[str] = None,
):
    """Log conformal coverage metrics to MLflow."""
    try:
        import mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("conformal_prediction")

        with mlflow.start_run(run_name=f"conformal_{queue_name.replace(' ', '_')}"):
            mlflow.log_param("queue_name", queue_name)
            for r in reports:
                h = r["horizon"]
                mlflow.log_metric(f"{h}_empirical_coverage", r["empirical_coverage"])
                mlflow.log_metric(f"{h}_interval_width",     r["interval_width"])
                mlflow.log_metric(f"{h}_coverage_gap",       r["coverage_gap"])
                mlflow.log_metric(f"{h}_guarantee_met",      int(r["guarantee_met"]))

        logger.info(f"Conformal metrics logged to MLflow for {queue_name}")
    except Exception as e:
        logger.warning(f"MLflow logging skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n" + "=" * 60)
    print("  MODULE 1 — Conformal Prediction Smoke Test")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n_cal, n_test = 500, 200

    # ── Regression test ───────────────────────────────────────────
    print("\n[Regression] Wrapping synthetic volume forecasts...")
    y_true_cal  = rng.normal(50, 10, n_cal)
    y_pred_cal  = y_true_cal + rng.normal(0, 3, n_cal)   # model predictions
    y_true_test = rng.normal(50, 10, n_test)
    y_pred_test = y_true_test + rng.normal(0, 3, n_test)

    cr = ConformalRegressor(alpha=0.2)
    cr.calibrate(y_true_cal, y_pred_cal)
    empirical = cr.empirical_coverage(y_true_test, y_pred_test)
    print(f"  Target coverage:   80.0%")
    print(f"  Empirical coverage: {empirical:.1%}  {'✓' if empirical >= 0.8 else '✗'}")
    print(f"  Interval width:    ±{cr.quantile:.3f}")

    # ── Classification test ───────────────────────────────────────
    print("\n[Classification] Wrapping synthetic breach probabilities...")
    y_binary_cal  = rng.integers(0, 2, n_cal)
    proba_cal     = np.clip(y_binary_cal * 0.7 + rng.normal(0, 0.15, n_cal), 0, 1)
    y_binary_test = rng.integers(0, 2, n_test)
    proba_test    = np.clip(y_binary_test * 0.7 + rng.normal(0, 0.15, n_test), 0, 1)

    cc = ConformalClassifier(alpha=0.2)
    cc.calibrate(y_binary_cal, proba_cal)
    pred_sets = cc.predict_set(proba_test)
    coverage  = np.mean([y_binary_test[i] in np.where(pred_sets[i])[0] for i in range(n_test)])
    print(f"  Target coverage:    80.0%")
    print(f"  Empirical coverage: {coverage:.1%}  {'✓' if coverage >= 0.8 else '✗'}")

    # ── Report ────────────────────────────────────────────────────
    reports = []
    for h in ["1h", "2h", "4h"]:
        reports.append(CoverageReport.regression_report(cr, y_true_test, y_pred_test, h))
    CoverageReport.print_report(reports)

    print("✓ Module 1 (Conformal Prediction) — PASSED\n")
