"""
Phase 2 — MLflow Experiment Tracking
=======================================
Centralised MLflow logging for all training runs, model versions,
calibration metrics, and drift events.

Registers:
  - LSTM training histories (loss curves)
  - XGBoost CV metrics (per fold + aggregate)
  - Calibration before/after metrics
  - Drift events
  - Feature importance plots
  - Saved model artifacts
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.xgboost
import mlflow.pytorch
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

TRACKING_URI     = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME  = os.getenv("MLFLOW_EXPERIMENT_NAME", "appian-breach-prediction")
MODEL_NAME       = os.getenv("MLFLOW_MODEL_NAME", "breach-classifier")

mlflow.set_tracking_uri(TRACKING_URI)


def _get_or_create_experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        logger.info(f"Created MLflow experiment: {name} (id={exp_id})")
        return exp_id
    return exp.experiment_id


# ─────────────────────────────────────────────────────────────────────────────
# MLflow Logger
# ─────────────────────────────────────────────────────────────────────────────
class MLflowLogger:
    """
    Thread-safe MLflow logger for all model training and evaluation events.
    Each training session starts a new run scoped to the experiment.
    """

    def __init__(self, experiment_name: str = EXPERIMENT_NAME):
        self.experiment_name = experiment_name
        self._run_id: Optional[str] = None

        try:
            mlflow.set_tracking_uri(TRACKING_URI)
            self.exp_id = _get_or_create_experiment(experiment_name)
            logger.info(f"MLflow tracking URI: {TRACKING_URI}")
        except Exception as e:
            logger.warning(f"MLflow unavailable ({e}) — tracking disabled")
            self.exp_id = None

    def start_run(
        self,
        run_name: str,
        tags: Optional[dict] = None,
    ) -> Optional[str]:
        """Start a new MLflow run. Returns run_id."""
        if self.exp_id is None:
            return None
        try:
            mlflow.set_experiment(self.experiment_name)
            run = mlflow.start_run(run_name=run_name, tags=tags or {})
            self._run_id = run.info.run_id
            logger.info(f"MLflow run started: {run_name} (id={self._run_id})")
            return self._run_id
        except Exception as e:
            logger.error(f"MLflow start_run error: {e}")
            return None

    def end_run(self, status: str = "FINISHED"):
        try:
            mlflow.end_run(status=status)
            logger.info(f"MLflow run ended: {self._run_id}")
        except Exception as e:
            logger.error(f"MLflow end_run error: {e}")

    def log_params(self, params: dict):
        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.error(f"MLflow log_params error: {e}")

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"MLflow log_metrics error: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.error(f"MLflow log_metric error: {e}")

    # ── LSTM Training ──────────────────────────────────────────────────────────
    def log_lstm_training(
        self,
        queue_name: str,
        params: dict,
        history: dict,
        model_path: Optional[str] = None,
    ):
        """Log LSTM training run."""
        run_name = f"lstm_{queue_name.replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
        self.start_run(run_name=run_name, tags={"model_type": "lstm", "queue": queue_name})

        try:
            self.log_params({
                "model_type":   "BiLSTM",
                "queue_name":   queue_name,
                **{f"lstm_{k}": v for k, v in params.items()},
            })

            # Log loss curves
            for epoch, (t_loss, v_loss) in enumerate(
                zip(history.get("train_loss", []), history.get("val_loss", []))
            ):
                mlflow.log_metrics({"train_loss": t_loss, "val_loss": v_loss}, step=epoch)

            mlflow.log_metrics({
                "best_epoch":   history.get("best_epoch", 0),
                "final_val_loss": history.get("val_loss", [0])[-1] if history.get("val_loss") else 0,
            })

            if model_path and Path(model_path).exists():
                mlflow.log_artifact(model_path, artifact_path="models")

        finally:
            self.end_run()

    # ── XGBoost Training ───────────────────────────────────────────────────────
    def log_xgboost_training(
        self,
        queue_name:    str,
        horizon_hours: int,
        params:        dict,
        metrics:       dict,
        model=None,
        feature_importance: Optional[dict] = None,
        calibration_report: Optional[dict] = None,
    ):
        """Log XGBoost classifier training run."""
        run_name = (
            f"xgb_{queue_name.replace(' ', '_')}_h{horizon_hours}_"
            f"{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
        )
        self.start_run(
            run_name=run_name,
            tags={
                "model_type":    "XGBoost",
                "queue":         queue_name,
                "horizon_hours": str(horizon_hours),
            }
        )

        try:
            self.log_params({
                "model_type":    "XGBoost",
                "queue_name":    queue_name,
                "horizon_hours": horizon_hours,
                **{f"xgb_{k}": v for k, v in params.items() if isinstance(v, (int, float, str, bool))},
            })

            # Aggregate metrics
            self.log_metrics({
                "mean_auc":          metrics.get("mean_auc", 0),
                "mean_brier":        metrics.get("mean_brier", 0),
                "mean_ap":           metrics.get("mean_ap", 0),
                "best_fold_auc":     metrics.get("best_fold_auc", 0),
            })

            # Per-fold metrics
            for fold_m in metrics.get("folds", []):
                fold = fold_m["fold"]
                mlflow.log_metrics({
                    f"fold_{fold}_auc":   fold_m["auc"],
                    f"fold_{fold}_brier": fold_m["brier"],
                    f"fold_{fold}_ap":    fold_m["ap"],
                }, step=fold)

            # Calibration
            if calibration_report:
                raw = calibration_report.get("raw", {})
                cal = calibration_report.get("calibrated", {})
                mlflow.log_metrics({
                    "brier_raw":       raw.get("brier_score", 0),
                    "brier_calibrated":cal.get("brier_score", 0),
                    "ece_raw":         raw.get("ece", 0),
                    "ece_calibrated":  cal.get("ece", 0),
                    "calibration_improvement": calibration_report.get("improvement_brier", 0),
                })

            # Feature importance table
            if feature_importance:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".csv", delete=False
                ) as f:
                    pd.DataFrame(
                        list(feature_importance.items()),
                        columns=["Feature", "Importance"],
                    ).to_csv(f.name, index=False)
                    mlflow.log_artifact(f.name, artifact_path="feature_importance")

            # Model artifact
            if model is not None:
                try:
                    mlflow.xgboost.log_model(
                        model.model,
                        artifact_path="xgboost_model",
                        registered_model_name=f"{MODEL_NAME}-{queue_name.replace(' ', '_')}-h{horizon_hours}",
                    )
                except Exception as e:
                    logger.warning(f"MLflow model registration failed: {e}")

        finally:
            self.end_run()

    # ── Drift Event ────────────────────────────────────────────────────────────
    def log_drift_event(
        self,
        queue_name: str,
        drift_type: str,
        drift_score: float,
        retrain_triggered: bool = False,
    ):
        """Log a detected drift event."""
        run_name = f"drift_{queue_name.replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.start_run(run_name=run_name, tags={"event_type": "drift", "queue": queue_name})
        try:
            self.log_params({
                "queue_name":       queue_name,
                "drift_type":       drift_type,
                "retrain_triggered": str(retrain_triggered),
            })
            self.log_metrics({
                "drift_score":       drift_score,
                "retrain_triggered": int(retrain_triggered),
            })
        finally:
            self.end_run()

    # ── System Health ──────────────────────────────────────────────────────────
    def log_evaluation(self, queue_name: str, horizon: int, metrics: dict):
        """Log a hold-out evaluation run."""
        run_name = f"eval_{queue_name.replace(' ', '_')}_h{horizon}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
        self.start_run(run_name=run_name, tags={"event_type": "evaluation", "queue": queue_name})
        try:
            self.log_params({"queue_name": queue_name, "horizon_hours": horizon})
            self.log_metrics(metrics)
        finally:
            self.end_run()
