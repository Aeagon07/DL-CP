"""
Phase 2 — Probability Calibration
===================================
Ensures that "87% breach probability" really means 87 out of 100 cases breach.
Without calibration, discriminative models (XGBoost) produce poorly-scaled scores.

Implements:
  1. Isotonic Regression calibration (non-parametric, recommended for large data)
  2. Platt Scaling (logistic, good for small calibration sets)
  3. Temperature Scaling (post-hoc neural calibration — ported to sklearn API)
  4. Calibration diagnostics: Reliability diagram + ECE / MCE scores
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Isotonic Calibrator
# ─────────────────────────────────────────────────────────────────────────────
class IsotonicCalibrator:
    """
    Non-parametric probability calibrator using Isotonic Regression.
    Fits a monotone function mapping raw scores → calibrated probabilities.
    Best for ≥1,000 calibration samples.
    """

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.is_fitted = False

    def fit(self, raw_proba: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self.iso.fit(raw_proba, y_true)
        self.is_fitted = True
        # Diagnostics
        calibrated = self.iso.predict(raw_proba)
        brier = brier_score_loss(y_true, calibrated)
        ece   = _expected_calibration_error(y_true, calibrated)
        logger.info(f"Isotonic calibrator fitted | Brier={brier:.4f} | ECE={ece:.4f}")
        return self

    def calibrate(self, raw_proba: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() first")
        return np.clip(self.iso.predict(raw_proba), 0.0, 1.0)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "IsotonicCalibrator":
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Platt Scaler (logistic)
# ─────────────────────────────────────────────────────────────────────────────
class PlattScaler:
    """Classic Platt scaling — fits a logistic regression on raw scores."""

    def __init__(self):
        self.lr = LogisticRegression(C=1.0)
        self.is_fitted = False

    def fit(self, raw_proba: np.ndarray, y_true: np.ndarray) -> "PlattScaler":
        self.lr.fit(raw_proba.reshape(-1, 1), y_true)
        self.is_fitted = True
        calibrated = self.calibrate(raw_proba)
        brier = brier_score_loss(y_true, calibrated)
        logger.info(f"Platt scaler fitted | Brier={brier:.4f}")
        return self

    def calibrate(self, raw_proba: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() first")
        return self.lr.predict_proba(raw_proba.reshape(-1, 1))[:, 1]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "PlattScaler":
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Temperature Scaling (neural-style, implemented in numpy)
# ─────────────────────────────────────────────────────────────────────────────
class TemperatureScaler:
    """
    Post-hoc calibration via temperature scaling on logits.
    Equivalent to dividing logits by learned scalar T.
    Minimises NLL on a held-out calibration set.
    """

    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    def _nll(self, logits, y, T):
        p = self._sigmoid(logits / T)
        return -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))

    def fit(self, raw_proba: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        from scipy.optimize import minimize_scalar

        logits = self._logit(raw_proba)
        result = minimize_scalar(
            fun=lambda T: self._nll(logits, y_true, T),
            bounds=(0.1, 10.0),
            method="bounded",
        )
        self.temperature = float(result.x)
        self.is_fitted = True
        calibrated = self.calibrate(raw_proba)
        brier = brier_score_loss(y_true, calibrated)
        logger.info(
            f"Temperature scaler fitted | T={self.temperature:.4f} | Brier={brier:.4f}"
        )
        return self

    def calibrate(self, raw_proba: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() first")
        logits = self._logit(raw_proba)
        return np.clip(self._sigmoid(logits / self.temperature), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def _expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def _max_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Computes Maximum Calibration Error (MCE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    mce  = 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        mce  = max(mce, abs(acc - conf))

    return float(mce)


def calibration_report(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_cal: np.ndarray,
) -> dict:
    """Returns a full calibration diagnostic report."""
    return {
        "raw": {
            "brier_score": float(brier_score_loss(y_true, y_prob_raw)),
            "ece":         _expected_calibration_error(y_true, y_prob_raw),
            "mce":         _max_calibration_error(y_true, y_prob_raw),
        },
        "calibrated": {
            "brier_score": float(brier_score_loss(y_true, y_prob_cal)),
            "ece":         _expected_calibration_error(y_true, y_prob_cal),
            "mce":         _max_calibration_error(y_true, y_prob_cal),
        },
        "improvement_brier": float(
            brier_score_loss(y_true, y_prob_raw) - brier_score_loss(y_true, y_prob_cal)
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Combined Calibration Pipeline
# ─────────────────────────────────────────────────────────────────────────────
class CalibrationPipeline:
    """
    Recommended: IsotonicRegression by default (best empirical performance).
    Exposes a uniform .fit() / .calibrate() interface.
    """

    def __init__(self, method: Literal["isotonic", "platt", "temperature"] = "isotonic"):
        self.method = method
        if method == "isotonic":
            self._cal = IsotonicCalibrator()
        elif method == "platt":
            self._cal = PlattScaler()
        elif method == "temperature":
            self._cal = TemperatureScaler()
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def fit(self, raw_proba: np.ndarray, y_true: np.ndarray) -> "CalibrationPipeline":
        self._cal.fit(raw_proba, y_true)
        return self

    def calibrate(self, raw_proba: np.ndarray) -> np.ndarray:
        return self._cal.calibrate(raw_proba)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "CalibrationPipeline":
        with open(path, "rb") as f:
            return pickle.load(f)
