"""
drift_detector.py — Monitor production predictions for data & concept drift.

Runs as a scheduled job (Airflow calls this every hour).
Results are logged to MLflow and Prometheus push gateway.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PUSHGATEWAY = os.getenv("PROMETHEUS_PUSHGATEWAY", "pushgateway:9091")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))    # share of drifted features
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.80"))


def load_reference_data(path: str = "monitoring/reference_data.parquet") -> pd.DataFrame:
    if Path(path).exists():
        return pd.read_parquet(path)
    # Synthetic reference when no real data is available (for demo / CI)
    logger.warning("Reference data not found — generating synthetic baseline.")
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        "feature_brightness":    rng.normal(0.50, 0.10, n),
        "feature_saturation":    rng.normal(0.60, 0.12, n),
        "feature_aspect_ratio":  rng.normal(1.33, 0.20, n),
        "feature_edge_density":  rng.normal(0.30, 0.08, n),
        "prediction":            rng.integers(0, 8, n),
        "target":                rng.integers(0, 8, n),
    })


def load_current_data(path: str = "monitoring/current_data.parquet") -> pd.DataFrame:
    if Path(path).exists():
        return pd.read_parquet(path)
    # Simulate a drifted distribution (demo)
    rng = np.random.default_rng(99)
    n = 200
    return pd.DataFrame({
        "feature_brightness":    rng.normal(0.65, 0.15, n),   # shifted +0.15
        "feature_saturation":    rng.normal(0.55, 0.18, n),
        "feature_aspect_ratio":  rng.normal(1.50, 0.25, n),
        "feature_edge_density":  rng.normal(0.28, 0.09, n),
        "prediction":            rng.integers(0, 8, n),
        "target":                rng.integers(0, 8, n),
    })


def run_drift_analysis(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    feature_cols = [c for c in reference.columns if c.startswith("feature_")]

    column_mapping = ColumnMapping(
        prediction="prediction",
        target="target",
        numerical_features=feature_cols,
    )

    report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    report_dict = report.as_dict()

    # ── Extract drift metrics ─────────────────────────────────────────────────
    drift_results = {}
    for metric in report_dict.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            r = metric.get("result", {})
            drift_results["dataset_drift_detected"] = r.get("dataset_drift", False)
            drift_results["share_drifted_features"] = r.get("share_of_drifted_columns", 0.0)
            drift_results["n_drifted_features"]     = r.get("number_of_drifted_columns", 0)

        if metric.get("metric") == "ClassificationQualityMetric":
            r = metric.get("result", {}).get("current", {})
            drift_results["accuracy"] = r.get("accuracy", None)
            drift_results["f1"]       = r.get("f1",       None)

    # Save HTML report
    os.makedirs("monitoring/reports", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"monitoring/reports/drift_report_{ts}.html"
    report.save_html(report_path)
    drift_results["report_path"] = report_path

    return drift_results


def push_metrics(results: dict):
    """Push drift metrics to Prometheus Pushgateway."""
    try:
        registry = CollectorRegistry()
        g_drift  = Gauge("ml_data_drift_share",    "Share of drifted features", registry=registry)
        g_acc    = Gauge("ml_model_accuracy",       "Model accuracy on labelled data", registry=registry)
        g_alert  = Gauge("ml_drift_alert",          "1 if retraining triggered", registry=registry)

        g_drift.set(results.get("share_drifted_features", 0))
        g_acc.set(results.get("accuracy") or 0)
        g_alert.set(1 if results.get("retrain_triggered") else 0)

        push_to_gateway(PUSHGATEWAY, job="drift_detector", registry=registry)
        logger.info("Metrics pushed to Pushgateway.")
    except Exception as exc:
        logger.warning(f"Could not push to Pushgateway: {exc}")


def main():
    logger.info("=== Drift Detection Job Started ===")

    reference = load_reference_data()
    current   = load_current_data()

    results = run_drift_analysis(reference, current)

    # ── Decision logic ────────────────────────────────────────────────────────
    drift_share   = results.get("share_drifted_features", 0)
    accuracy      = results.get("accuracy")
    retrain_needed = (
        results.get("dataset_drift_detected", False)
        and drift_share >= DRIFT_THRESHOLD
    ) or (accuracy is not None and accuracy < ACCURACY_THRESHOLD)

    results["retrain_triggered"] = retrain_needed

    logger.info(f"Drift detected : {results.get('dataset_drift_detected')}")
    logger.info(f"Drifted features: {results.get('n_drifted_features')} ({drift_share:.0%})")
    logger.info(f"Accuracy       : {accuracy}")
    logger.info(f"Retrain needed : {retrain_needed}")

    if retrain_needed:
        logger.warning("⚠️  RETRAINING TRIGGERED — Airflow will schedule a new training run.")
        # In production: write a flag file / call Airflow REST API
        Path("monitoring/retrain_flag.txt").write_text(datetime.now().isoformat())

    push_metrics(results)

    # Persist results as JSON for Airflow XCom / downstream tasks
    out_path = "monitoring/latest_drift_results.json"
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in results.items() if k != "report_path"}, f, indent=2)

    logger.info(f"Report saved: {results['report_path']}")
    logger.info(f"Results saved: {out_path}")
    logger.info("=== Drift Detection Job Complete ===")
    return results


if __name__ == "__main__":
    main()
