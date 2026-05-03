"""
Phase 2 — Model Evaluation
============================
Evaluates trained models on a held-out test set.
Produces:
  - AUC-ROC per queue per horizon
  - Brier Score (before + after calibration)
  - Expected Calibration Error (ECE)
  - Average Precision Score
  - Confusion matrix at threshold 0.7
  - SHAP summary statistics
  - Prints a full formatted report

Usage:
  python phase2_ml/evaluate.py --data data/sample_data.csv
  python phase2_ml/evaluate.py --data data/sample_data.csv --queue "Payment Processing"
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, average_precision_score,
    confusion_matrix, classification_report, f1_score,
)

sys.path.insert(0, str(Path(__file__).parents[1]))

from phase1_pipeline.feature_engineering import add_temporal_features, prepare_sequences
from phase2_ml.xgboost_classifier import XGBoostBreachClassifier, BreachEnsemble
from phase2_ml.calibration import calibration_report, _expected_calibration_error
from phase2_ml.mlflow_tracking import MLflowLogger
from phase2_ml.drift_detection import psi_report

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/")

QUEUE_NAMES = [
    "Document Review", "Compliance Check", "Payment Processing",
    "Customer Onboarding", "Risk Assessment", "Audit Preparation",
]

BREACH_THRESHOLD = float(os.getenv("BREACH_THRESHOLD", "0.7"))


def _load_calibrator(queue_name: str, horizon: int):
    safe = queue_name.replace(" ", "_")
    path = MODEL_DIR / f"calibrator_{safe}_h{horizon}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _load_shap_explainer(queue_name: str, horizon: int):
    safe = queue_name.replace(" ", "_")
    path = MODEL_DIR / f"shap_{safe}_h{horizon}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation per Queue
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_queue(
    df: pd.DataFrame,
    queue_name: str,
    ml_logger: MLflowLogger,
) -> dict:
    """Evaluate all horizons for a single queue. Returns nested metrics dict."""
    q_df = df[df["queue_name"] == queue_name].copy().sort_values("snapshot_time")

    # Use last 15% as test set (strict temporal split)
    n = len(q_df)
    test_start = int(n * 0.85)
    train_df   = q_df.iloc[:test_start]
    test_df    = q_df.iloc[test_start:]

    if len(test_df) < 30:
        logger.warning(f"[{queue_name}] Test set too small ({len(test_df)}) — skipping evaluation")
        return {}

    # PSI drift report (train vs test distribution)
    feat_cols = [
        "wip_count", "throughput_60m", "utilization_rate",
        "time_pressure", "complexity_backlog",
    ]
    psi = psi_report(train_df, test_df, feat_cols)

    try:
        ensemble = BreachEnsemble.load_all(queue_name)
    except FileNotFoundError as e:
        logger.error(f"[{queue_name}] Model not found: {e}. Run train.py first.")
        return {}

    queue_results = {"psi_report": psi, "horizons": {}}
    all_ok = True

    for horizon in [1, 2, 4]:
        clf = ensemble.classifiers[horizon]
        X_test, feat_names = clf._prepare_features(test_df.copy())
        y_test = test_df["breach_label"].values.astype(int)

        if y_test.sum() == 0:
            logger.warning(f"[{queue_name}|{horizon}h] No positive labels in test set")
            continue

        # Raw predictions
        raw_proba = clf.model.predict_proba(X_test)[:, 1]

        # Calibrated predictions
        cal = _load_calibrator(queue_name, horizon)
        cal_proba = cal.calibrate(raw_proba) if cal else raw_proba

        # Classification at threshold
        y_pred = (cal_proba >= BREACH_THRESHOLD).astype(int)

        # Metrics
        auc   = roc_auc_score(y_test, cal_proba)
        brier = brier_score_loss(y_test, cal_proba)
        ap    = average_precision_score(y_test, cal_proba)
        ece   = _expected_calibration_error(y_test, cal_proba)
        f1    = f1_score(y_test, y_pred, zero_division=0)

        pass_auc   = auc   >= 0.88
        pass_brier = brier <= 0.12
        if not (pass_auc and pass_brier):
            all_ok = False

        cm = confusion_matrix(y_test, y_pred).tolist()

        # SHAP statistics
        shap_exp = _load_shap_explainer(queue_name, horizon)
        shap_stats = {}
        if shap_exp is not None:
            sample_size = min(50, len(X_test))
            sample_df = test_df.iloc[:sample_size].copy()
            explanations = shap_exp.explain(sample_df[feat_names], top_k=5)
            if explanations:
                all_shap = [e.get("all_shap_values", {}) for e in explanations]
                if all_shap and all_shap[0]:
                    feat_impact = {}
                    for feat in all_shap[0].keys():
                        vals = [e.get(feat, 0.0) for e in all_shap]
                        feat_impact[feat] = round(float(np.mean(np.abs(vals))), 4)
                    shap_stats = dict(
                        sorted(feat_impact.items(), key=lambda x: -x[1])[:5]
                    )

        cal_report = calibration_report(y_test, raw_proba, cal_proba)

        horizon_result = {
            "auc_roc":          round(float(auc),   4),
            "brier_score":      round(float(brier),  4),
            "average_precision": round(float(ap),    4),
            "ece":              round(float(ece),    4),
            "f1_score":         round(float(f1),     4),
            "confusion_matrix": cm,
            "n_test":           int(len(y_test)),
            "n_positive":       int(y_test.sum()),
            "breach_rate":      round(float(y_test.mean()), 4),
            "threshold":        BREACH_THRESHOLD,
            "pass_auc_target":  pass_auc,
            "pass_brier_target": pass_brier,
            "calibration":      cal_report,
            "top5_shap_features": shap_stats,
        }
        queue_results["horizons"][horizon] = horizon_result

        # Log to MLflow
        ml_logger.log_evaluation(
            queue_name = queue_name,
            horizon    = horizon,
            metrics    = {
                "auc_roc":    auc,
                "brier_score": brier,
                "avg_precision": ap,
                "ece":        ece,
                "f1_score":   f1,
            },
        )

    queue_results["all_targets_met"] = all_ok
    return queue_results


# ─────────────────────────────────────────────────────────────────────────────
# Report Printer
# ─────────────────────────────────────────────────────────────────────────────
def print_report(all_results: dict):
    """Pretty-print evaluation report."""
    SEP = "━" * 75

    print(f"\n{SEP}")
    print("  APPIAN BREACH CLASSIFIER — EVALUATION REPORT")
    print(f"{SEP}\n")
    print(f"  Target: AUC-ROC ≥ 0.88  |  Brier Score ≤ 0.12\n")

    overall_auc = []
    overall_brier = []

    for queue_name, result in all_results.items():
        if not result:
            continue

        print(f"\n  📋 Queue: {queue_name}")
        print(f"  {'─'*65}")

        horizons = result.get("horizons", {})
        for h, m in sorted(horizons.items()):
            auc_ok    = "✓" if m["pass_auc_target"]   else "✗"
            brier_ok  = "✓" if m["pass_brier_target"]  else "✗"
            print(
                f"    Horizon {h}h | "
                f"AUC={m['auc_roc']:.4f} {auc_ok} | "
                f"Brier={m['brier_score']:.4f} {brier_ok} | "
                f"AP={m['average_precision']:.4f} | "
                f"ECE={m['ece']:.4f} | "
                f"F1={m['f1_score']:.4f}"
            )
            overall_auc.append(m["auc_roc"])
            overall_brier.append(m["brier_score"])

            if m.get("top5_shap_features"):
                top_feat = list(m["top5_shap_features"].items())[0]
                print(f"       Top SHAP: {top_feat[0]} ({top_feat[1]:.4f})")

        psi = result.get("psi_report", {})
        if psi:
            high_psi = {k: v["psi"] for k, v in psi.items() if v["severity"] == "significant"}
            if high_psi:
                print(f"    ⚠ Feature drift (PSI): {high_psi}")
            else:
                print(f"    ✓ Feature distribution: stable")

    print(f"\n{SEP}")
    if overall_auc and overall_brier:
        print(f"  Overall Mean AUC:   {np.mean(overall_auc):.4f}")
        print(f"  Overall Mean Brier: {np.mean(overall_brier):.4f}")
        met = np.mean(overall_auc) >= 0.88 and np.mean(overall_brier) <= 0.12
        print(f"  Targets Met: {'✓ YES' if met else '✗ NO'}")
    print(f"{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Appian Model Evaluation")
    parser.add_argument("--data",  type=str, default="data/sample_data.csv")
    parser.add_argument("--queue", type=str, default=None, help="Single queue to evaluate")
    args = parser.parse_args()

    df     = pd.read_csv(args.data, parse_dates=["snapshot_time"])
    df     = add_temporal_features(df)
    queues = [args.queue] if args.queue else QUEUE_NAMES

    ml_logger    = MLflowLogger()
    all_results  = {}

    for queue_name in queues:
        if queue_name not in df["queue_name"].unique():
            logger.warning(f"Queue '{queue_name}' not in dataset")
            continue
        result = evaluate_queue(df, queue_name, ml_logger)
        all_results[queue_name] = result

    print_report(all_results)


if __name__ == "__main__":
    main()
