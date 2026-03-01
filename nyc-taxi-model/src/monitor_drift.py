"""Drift monitoring with Evidently.

Compares reference data (training set) with current streaming data.
Pushes drift metrics to VictoriaMetrics, saves HTML report to MinIO.
"""

import io
import json
import logging
import os
import sys
from datetime import datetime

import boto3
import pandas as pd
import requests
from evidently import Report
from evidently.presets import DataDriftPreset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

PROCESSED_BUCKET = "taxi-processed"
STREAM_BUCKET = "taxi-streaming"
REPORTS_BUCKET = "taxi-reports"
VM_URL = os.environ.get("VM_URL", "http://victoriametrics.monitoring.svc.cluster.local:8428")
MAX_CURRENT_FILES = int(os.environ.get("MAX_CURRENT_FILES", "24"))
RETRAIN_THRESHOLD = float(os.environ.get("RETRAIN_THRESHOLD", "0.5"))
AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://airflow-webserver.airflow.svc.cluster.local:8080")
AIRFLOW_USER = os.environ.get("AIRFLOW_USER", "admin")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD", "admin")

FEATURE_COLS = [
    "trip_distance",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "passenger_count",
]


def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def ensure_bucket(s3, bucket):
    try:
        s3.head_bucket(Bucket=bucket)
    except s3.exceptions.ClientError:
        s3.create_bucket(Bucket=bucket)
        log.info("Created bucket: %s", bucket)


def load_reference(s3):
    """Load reference data (training set)."""
    log.info("Loading reference data from s3://%s/trips/train.parquet", PROCESSED_BUCKET)
    resp = s3.get_object(Bucket=PROCESSED_BUCKET, Key="trips/train.parquet")
    df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    return df


def load_current(s3):
    """Load latest streaming batches."""
    log.info("Loading current data from s3://%s/", STREAM_BUCKET)

    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=STREAM_BUCKET):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                keys.append(obj["Key"])

    if not keys:
        log.warning("No streaming data found in s3://%s/", STREAM_BUCKET)
        return None

    keys.sort()
    recent_keys = keys[-MAX_CURRENT_FILES:]
    log.info("Found %d total files, using latest %d", len(keys), len(recent_keys))

    frames = []
    for key in recent_keys:
        resp = s3.get_object(Bucket=STREAM_BUCKET, Key=key)
        df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def prepare_features(df):
    """Extract and engineer features for drift comparison.

    Handles both preprocessed data (already has pickup_hour etc.)
    and raw streaming data (needs extraction from tpep_pickup_datetime).
    Applies same outlier filters as preprocess.py for fair comparison.
    """
    df = df.copy()
    if "tpep_pickup_datetime" in df.columns and "pickup_hour" not in df.columns:
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
        df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek
        df["pickup_month"] = df["tpep_pickup_datetime"].dt.month

    # Same outlier filters as preprocess.py
    if "trip_distance" in df.columns:
        df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]
    if "passenger_count" in df.columns:
        df = df[(df["passenger_count"] >= 0) & (df["passenger_count"] <= 9)]

    cols = [c for c in FEATURE_COLS if c in df.columns]
    return df[cols].dropna()


def run_drift_report(reference, current):
    """Run Evidently DataDrift report."""
    report = Report([DataDriftPreset()])
    result = report.run(reference, current)
    return result


def extract_metrics(result):
    """Extract drift metrics from Evidently 0.7.x report result.

    Evidently 0.7.x JSON structure:
    - DriftedColumnsCount: value.count, value.share
    - ValueDrift per column: config.column, config.threshold, value (p-value)
    Drift detected when p-value < threshold.
    """
    result_json = json.loads(result.json())
    metrics_list = result_json.get("metrics", [])

    drift_share = 0.0
    n_drifted = 0
    n_columns = 0
    feature_scores = {}

    for metric in metrics_list:
        metric_type = metric.get("config", {}).get("type", "")

        if "DriftedColumnsCount" in metric_type:
            value = metric.get("value", {})
            n_drifted = int(value.get("count", 0))
            drift_share = float(value.get("share", 0))

        elif "ValueDrift" in metric_type:
            config = metric.get("config", {})
            column = config.get("column", "")
            threshold = float(config.get("threshold", 0.05))
            p_value = float(metric.get("value", 1.0))
            drifted = 1 if p_value < threshold else 0
            feature_scores[column] = {
                "drift_score": p_value,
                "drifted": drifted,
            }
            n_columns += 1

    dataset_drift = drift_share > 0.5

    result_metrics = {
        "evidently_dataset_drift": 1 if dataset_drift else 0,
        "evidently_drift_share": drift_share,
        "evidently_number_of_drifted_columns": n_drifted,
        "evidently_number_of_columns": n_columns,
    }

    return result_metrics, feature_scores


def push_to_vm(metrics, feature_metrics):
    """Push metrics to VictoriaMetrics in Prometheus text format."""
    lines = []
    for name, value in metrics.items():
        lines.append(f"{name} {value}")

    for feature, values in feature_metrics.items():
        lines.append(f'evidently_feature_drift_score{{feature="{feature}"}} {values["drift_score"]}')
        lines.append(f'evidently_feature_drifted{{feature="{feature}"}} {values["drifted"]}')

    body = "\n".join(lines) + "\n"
    log.info("Pushing %d metrics to VictoriaMetrics:\n%s", len(lines), body)

    resp = requests.post(
        f"{VM_URL}/api/v1/import/prometheus",
        data=body,
        headers={"Content-Type": "text/plain"},
        timeout=10,
    )
    resp.raise_for_status()
    log.info("Metrics pushed to VictoriaMetrics")


def save_report(s3, result):
    """Save HTML report to MinIO."""
    ensure_bucket(s3, REPORTS_BUCKET)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    key = f"drift/{timestamp}.html"

    html_path = f"/tmp/drift_report_{timestamp}.html"
    result.save_html(html_path)
    with open(html_path, "r") as f:
        html = f.read()

    s3.put_object(
        Bucket=REPORTS_BUCKET,
        Key=key,
        Body=html.encode(),
        ContentType="text/html",
    )
    log.info("Report saved to s3://%s/%s", REPORTS_BUCKET, key)


def trigger_retrain():
    """Trigger the auto-retrain DAG via Airflow REST API."""
    try:
        resp = requests.post(
            f"{AIRFLOW_URL}/api/v1/dags/taxi_auto_retrain/dagRuns",
            json={"conf": {"triggered_by": "drift_monitor"}},
            auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        log.info("Retrain DAG triggered successfully")
    except Exception as e:
        log.error("Failed to trigger retrain DAG: %s", e)


def main():
    s3 = get_s3()

    reference = load_reference(s3)
    ref_features = prepare_features(reference)
    log.info("Reference data: %d rows", len(ref_features))

    current_raw = load_current(s3)
    if current_raw is None or len(current_raw) == 0:
        log.warning("No current data, skipping drift check")
        return

    curr_features = prepare_features(current_raw)
    log.info("Current data: %d rows", len(curr_features))

    if len(curr_features) < 100:
        log.warning("Current data too small (%d rows), skipping", len(curr_features))
        return

    log.info("Running Evidently DataDrift report...")
    result = run_drift_report(ref_features, curr_features)

    metrics, feature_metrics = extract_metrics(result)
    log.info("Dataset drift: %s, drift share: %.2f",
             "YES" if metrics["evidently_dataset_drift"] else "NO",
             metrics["evidently_drift_share"])

    for feature, vals in feature_metrics.items():
        status = "DRIFTED" if vals["drifted"] else "ok"
        log.info("  %-25s score=%.4f  %s", feature, vals["drift_score"], status)

    push_to_vm(metrics, feature_metrics)
    save_report(s3, result)

    if metrics["evidently_drift_share"] >= RETRAIN_THRESHOLD:
        log.info("Drift share %.2f >= threshold %.2f, triggering retrain",
                 metrics["evidently_drift_share"], RETRAIN_THRESHOLD)
        trigger_retrain()
    else:
        log.info("Drift share %.2f < threshold %.2f, no retrain needed",
                 metrics["evidently_drift_share"], RETRAIN_THRESHOLD)

    log.info("Drift monitoring complete!")


if __name__ == "__main__":
    main()
