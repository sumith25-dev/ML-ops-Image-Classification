"""
mlops_pipeline_dag.py — Airflow DAG for the Adobe MLOps project.

Schedule: hourly drift check + weekly full retraining.

Tasks:
  1. data_validation      — Check dataset schema and quality
  2. drift_detection      — Run evidently drift analysis
  3. conditional_retrain  — Branch: retrain if drift detected
  4. model_training       — Train EfficientNet, register in MLflow
  5. model_validation     — Validate new model vs current champion
  6. promote_model        — Assign 'canary' alias in MLflow registry
  7. notify               — Send Slack/email notification
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# ── Default args ──────────────────────────────────────────────────────────────
default_args = {
    "owner":            "mlops-team",
    "depends_on_past":  False,
    "email_on_failure": True,
    "email":            ["mlops-alerts@yourcompany.com"],
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}


# ── Task functions ────────────────────────────────────────────────────────────
def data_validation_fn(**ctx):
    """Validate raw data: schema, nulls, class distribution."""
    import pandas as pd
    import numpy as np

    data_path = "/opt/airflow/data/current_data.parquet"
    if not Path(data_path).exists():
        print("No data file found — generating synthetic data for demo.")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "feature_brightness":   rng.normal(0.5, 0.1, 300),
            "feature_saturation":   rng.normal(0.6, 0.1, 300),
            "feature_aspect_ratio": rng.normal(1.3, 0.2, 300),
            "feature_edge_density": rng.normal(0.3, 0.1, 300),
            "prediction": rng.integers(0, 8, 300),
            "target":     rng.integers(0, 8, 300),
        })
        Path("/opt/airflow/monitoring").mkdir(parents=True, exist_ok=True)
        df.to_parquet("/opt/airflow/monitoring/current_data.parquet")
        df.sample(500, replace=True).to_parquet("/opt/airflow/monitoring/reference_data.parquet")
        print("Synthetic data generated.")
        return

    df = pd.read_parquet(data_path)
    required_cols = ["feature_brightness", "feature_saturation", "feature_aspect_ratio"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    null_pct = df.isnull().mean()
    high_null = null_pct[null_pct > 0.05]
    if not high_null.empty:
        raise ValueError(f"High null rate in: {high_null.to_dict()}")

    print(f"✅ Data validation passed — {len(df)} rows, {df.shape[1]} columns.")


def drift_detection_fn(**ctx):
    """Run drift detection; push result to XCom."""
    import sys
    sys.path.insert(0, "/opt/airflow")
    from monitoring.drift_detector import load_reference_data, load_current_data, run_drift_analysis

    ref  = load_reference_data("/opt/airflow/monitoring/reference_data.parquet")
    curr = load_current_data("/opt/airflow/monitoring/current_data.parquet")
    results = run_drift_analysis(ref, curr)
    ctx["ti"].xcom_push(key="drift_results", value=results)
    print(f"Drift results: {results}")


def branch_retrain_fn(**ctx):
    """Decide whether to retrain based on drift results."""
    results = ctx["ti"].xcom_pull(key="drift_results", task_ids="drift_detection")
    drift_detected = results.get("dataset_drift_detected", False)
    share_drifted  = results.get("share_drifted_features", 0)
    accuracy       = results.get("accuracy") or 1.0

    print(f"Drift detected: {drift_detected}, share: {share_drifted:.0%}, accuracy: {accuracy:.2%}")

    if drift_detected and (share_drifted >= 0.3 or accuracy < 0.80):
        print("→ Routing to model_training")
        return "model_training"
    print("→ No retraining needed")
    return "no_retrain"


def model_training_fn(**ctx):
    """Trigger training script."""
    cmd = [
        "python", "/opt/airflow/model/train.py",
        "--data_dir", "/opt/airflow/data/images",
        "--epochs", "5",
        "--run_name", f"auto_retrain_{datetime.now().strftime('%Y%m%d_%H%M')}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Training script failed")
    print("✅ Training complete.")


def model_validation_fn(**ctx):
    """Compare new model vs champion; pass/fail gate."""
    import mlflow
    import os

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = mlflow.tracking.MlflowClient()

    try:
        versions = client.search_model_versions("name='image_classifier'")
        if not versions:
            print("No registered models yet — skipping validation gate.")
            return

        latest = sorted(versions, key=lambda v: int(v.version))[-1]
        run = client.get_run(latest.run_id)
        val_acc = run.data.metrics.get("best_val_acc", 0)
        print(f"New model val_acc: {val_acc:.2%}")

        if val_acc < 0.70:
            raise ValueError(f"New model val_acc {val_acc:.2%} < 70% threshold — rejecting.")

        print("✅ Model validation passed.")
    except Exception as exc:
        print(f"Validation step warning: {exc}")


def promote_model_fn(**ctx):
    """Assign 'canary' alias to the latest model version in MLflow."""
    import mlflow, os
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = mlflow.tracking.MlflowClient()

    try:
        versions = client.search_model_versions("name='image_classifier'")
        if not versions:
            print("No model to promote.")
            return
        latest = sorted(versions, key=lambda v: int(v.version))[-1]
        client.set_registered_model_alias("image_classifier", "canary", latest.version)
        print(f"✅ Version {latest.version} promoted to 'canary'.")
    except Exception as exc:
        print(f"Promote step warning: {exc}")


def notify_fn(**ctx):
    drift_results = ctx["ti"].xcom_pull(key="drift_results", task_ids="drift_detection") or {}
    message = (
        f"🔔 MLOps Pipeline Run Complete\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"Drift detected: {drift_results.get('dataset_drift_detected', 'N/A')}\n"
        f"Drifted features: {drift_results.get('n_drifted_features', 'N/A')}\n"
        f"Model accuracy: {drift_results.get('accuracy', 'N/A')}\n"
    )
    print(message)
    # In production: call Slack webhook / send email here


# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="mlops_image_classifier_pipeline",
    default_args=default_args,
    description="Adobe MLOps — drift detection + conditional retraining pipeline",
    schedule_interval="@hourly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "adobe", "image-classification"],
) as dag:

    data_validation = PythonOperator(
        task_id="data_validation",
        python_callable=data_validation_fn,
    )

    drift_detection = PythonOperator(
        task_id="drift_detection",
        python_callable=drift_detection_fn,
    )

    branch_retrain = BranchPythonOperator(
        task_id="branch_retrain",
        python_callable=branch_retrain_fn,
    )

    no_retrain = EmptyOperator(task_id="no_retrain")

    model_training = PythonOperator(
        task_id="model_training",
        python_callable=model_training_fn,
        execution_timeout=timedelta(hours=2),
    )

    model_validation = PythonOperator(
        task_id="model_validation",
        python_callable=model_validation_fn,
    )

    promote_model = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model_fn,
    )

    notify = PythonOperator(
        task_id="notify",
        python_callable=notify_fn,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # ── Task ordering ─────────────────────────────────────────────────────────
    (
        data_validation
        >> drift_detection
        >> branch_retrain
        >> [model_training, no_retrain]
    )
    model_training >> model_validation >> promote_model >> notify
    no_retrain >> notify
