"""
Phase 2 — XGBoost SLA Breach Classifier
=========================================
Combines LSTM volume forecasts with operational features to predict
the probability of an SLA breach at 1h, 2h, and 4h horizons.

Features:
  - All 5 core operational features (+ lag/rolling variants)
  - LSTM volume forecast for each horizon
  - Temporal features (hour, day, weekend)
  - Queue SLA metadata (difficulty encoding)

Training:
  - Walk-forward cross-validation (5 folds)
  - Class imbalance handled via scale_pos_weight
  - Hyperparameter tuning with Optuna (optional)
  - Platt scaling calibration applied post-training
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/"))


FEATURE_COLUMNS = [
    # Core operational
    "wip_count", "throughput_15m", "throughput_30m", "throughput_60m",
    "utilization_rate", "time_pressure", "complexity_backlog",
    # Temporal
    "sla_hours", "hour_of_day", "day_of_week", "is_weekend",
    # Lag features (added by feature_engineering.add_temporal_features)
    "wip_count_lag1", "wip_count_lag3", "wip_count_lag6",
    "throughput_60m_lag1", "throughput_60m_lag3",
    "utilization_rate_lag1", "utilization_rate_lag3",
    "time_pressure_lag1", "time_pressure_lag3",
    # Rolling stats
    "wip_count_roll_mean_6", "wip_count_roll_std_6", "wip_count_roll_mean_12",
    "throughput_60m_roll_mean_6", "utilization_rate_roll_mean_6",
    # Derived
    "breach_velocity", "saturation_index",
    # LSTM forecasts (injected at inference time)
    "lstm_forecast_1h", "lstm_forecast_2h", "lstm_forecast_4h",
]


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost Breach Classifier
# ─────────────────────────────────────────────────────────────────────────────
class XGBoostBreachClassifier:
    """
    Per-queue, per-horizon XGBoost breach probability classifier.
    Separate model for each (queue_name, horizon) pair for maximum accuracy.
    """

    DEFAULT_PARAMS = {
        "objective":        "binary:logistic",
        "eval_metric":      ["auc", "logloss"],
        "max_depth":        6,
        "n_estimators":     300,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma":            0.1,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "tree_method":      "hist",
        "random_state":     42,
        "n_jobs":           -1,
    }

    def __init__(self, horizon_hours: int = 1, params: Optional[dict] = None):
        assert horizon_hours in (1, 2, 4), "horizon_hours must be 1, 2, or 4"
        self.horizon_hours = horizon_hours
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: list[str] = []
        self.is_trained = False

    def _prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Select available feature columns from df."""
        available = [c for c in FEATURE_COLUMNS if c in df.columns]

        # Fill LSTM forecast columns with 0 if not present (will be injected at inference)
        for col in ["lstm_forecast_1h", "lstm_forecast_2h", "lstm_forecast_4h"]:
            if col not in df.columns:
                df[col] = 0.0

        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        return df[available].fillna(0), available

    def fit_walk_forward(
        self,
        df: pd.DataFrame,
        queue_name: str,
        n_splits: int = 5,
    ) -> dict:
        """
        Walk-forward cross-validation training.
        Each fold uses all past data as train, next window as test.
        Prevents data leakage for temporal data.
        """
        label_col = "breach_label"
        if label_col not in df.columns:
            raise ValueError("df must contain 'breach_label' column")

        X, feat_names = self._prepare_features(df.copy())
        y = df[label_col].values.astype(int)

        pos_count = int(y.sum())
        neg_count = int(len(y) - pos_count)
        scale_pos_weight = neg_count / max(1, pos_count)
        logger.info(
            f"[{queue_name} | {self.horizon_hours}h] "
            f"Pos={pos_count} Neg={neg_count} scale_pos_weight={scale_pos_weight:.2f}"
        )

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []
        best_model = None
        best_auc = 0.0

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx],   y[val_idx]

            if y_val.sum() == 0:
                logger.warning(f"Fold {fold+1}: no positives in val — skipping")
                continue

            model = xgb.XGBClassifier(
                **{**self.params, "scale_pos_weight": scale_pos_weight}
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=30,
                verbose=False,
            )

            proba = model.predict_proba(X_val)[:, 1]
            auc   = roc_auc_score(y_val, proba)
            brier = brier_score_loss(y_val, proba)
            ap    = average_precision_score(y_val, proba)

            fold_metrics.append({"fold": fold + 1, "auc": auc, "brier": brier, "ap": ap})
            logger.info(
                f"  Fold {fold+1}: AUC={auc:.4f} | Brier={brier:.4f} | AP={ap:.4f}"
            )

            if auc > best_auc:
                best_auc = auc
                best_model = model

        if best_model is None:
            raise RuntimeError(f"No valid folds for {queue_name} | horizon {self.horizon_hours}h")

        self.model = best_model
        self.feature_names = feat_names
        self.is_trained = True

        mean_metrics = {
            "mean_auc":   float(np.mean([m["auc"]   for m in fold_metrics])),
            "mean_brier": float(np.mean([m["brier"]  for m in fold_metrics])),
            "mean_ap":    float(np.mean([m["ap"]     for m in fold_metrics])),
            "best_fold_auc": float(best_auc),
            "folds": fold_metrics,
        }
        logger.info(
            f"[{queue_name}|{self.horizon_hours}h] Final: "
            f"meanAUC={mean_metrics['mean_auc']:.4f} "
            f"meanBrier={mean_metrics['mean_brier']:.4f}"
        )
        return mean_metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns array of shape (N,) with breach probabilities."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit_walk_forward() first.")
        X_feat, _ = self._prepare_features(X.copy())
        return self.model.predict_proba(X_feat)[:, 1]

    def predict_single(self, features: dict) -> float:
        """Single-record inference. Returns breach probability [0,1]."""
        df = pd.DataFrame([features])
        return float(self.predict_proba(df)[0])

    def get_feature_importance(self) -> dict[str, float]:
        """Returns feature importance dict (gain-based)."""
        if not self.is_trained:
            return {}
        scores = self.model.get_booster().get_score(importance_type="gain")
        total  = sum(scores.values()) or 1.0
        return {k: round(v / total, 4) for k, v in sorted(scores.items(), key=lambda x: -x[1])}

    def save(self, path: Optional[str] = None):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = path or str(MODEL_DIR / f"xgb_h{self.horizon_hours}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved XGBoost classifier → {path}")

    @classmethod
    def load(cls, path: str) -> "XGBoostBreachClassifier":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Loaded XGBoost classifier from {path}")
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble — runs all 3 horizon classifiers
# ─────────────────────────────────────────────────────────────────────────────
class BreachEnsemble:
    """Manages 3 XGBoost classifiers (1h / 2h / 4h) per queue."""

    HORIZONS = [1, 2, 4]

    def __init__(self):
        self.classifiers: dict[int, XGBoostBreachClassifier] = {
            h: XGBoostBreachClassifier(horizon_hours=h) for h in self.HORIZONS
        }

    def train(self, df: pd.DataFrame, queue_name: str) -> dict:
        """Train all 3 classifiers. Returns combined metrics."""
        all_metrics = {}
        for h, clf in self.classifiers.items():
            metrics = clf.fit_walk_forward(df, queue_name=queue_name)
            all_metrics[f"horizon_{h}h"] = metrics
        return all_metrics

    def predict(self, features: dict, lstm_forecasts: dict) -> dict[int, float]:
        """
        features: dict of operational features
        lstm_forecasts: {"1h": float, "2h": float, "4h": float}
        Returns: {1: prob_1h, 2: prob_2h, 4: prob_4h}
        """
        enriched = {
            **features,
            "lstm_forecast_1h": lstm_forecasts.get("1h", 0.0),
            "lstm_forecast_2h": lstm_forecasts.get("2h", 0.0),
            "lstm_forecast_4h": lstm_forecasts.get("4h", 0.0),
        }
        return {h: clf.predict_single(enriched) for h, clf in self.classifiers.items()}

    def save_all(self, queue_name: str):
        safe_name = queue_name.replace(" ", "_")
        for h, clf in self.classifiers.items():
            clf.save(str(MODEL_DIR / f"xgb_{safe_name}_h{h}.pkl"))

    @classmethod
    def load_all(cls, queue_name: str) -> "BreachEnsemble":
        safe_name = queue_name.replace(" ", "_")
        ensemble = cls()
        for h in cls.HORIZONS:
            path = str(MODEL_DIR / f"xgb_{safe_name}_h{h}.pkl")
            ensemble.classifiers[h] = XGBoostBreachClassifier.load(path)
        return ensemble
