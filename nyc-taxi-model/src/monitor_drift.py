"""Drift monitoring with Evidently.

Compares reference data (silver/trips) with current streaming data (bronze/streaming).
Pushes drift metrics to VictoriaMetrics, saves HTML report to MinIO.
Triggers retrain DAG if drift exceeds threshold.
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

from lake import read_delta, table_uri

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

SILVER_TRIPS = table_uri("silver", "trips")
BRONZE_STREAMING = table_uri("bronze", "streaming")
REPORTS_BUCKET = "taxi-reports"
VM_URL = os.environ.get("VM_URL", "http://victoriametrics.monitoring.svc.cluster.local:8428")
RETRAIN_THRESHOLD = float(os.environ.get("RETRAIN_THRESHOLD", "0.5"))
AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://airflow-webserver.airflow.svc.cluster.local:8080")
AIRFLOW_USER = os.environ.get("AIRFLOW_USER", "admin")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD", "admin")

FEATURE_COLS = [
    "trip_distance", "pickup_hour", "pickup_day_of_week",
    "pickup_month", "passenger_count",
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


def prepare_features(df):
    """Extract features for drift comparison.

    Handles both preprocessed (silver) and raw (bronze) data.
    Applies same outlier filters as preprocess.py.
    """
    df = df.copy()
    if "tpep_pickup_datetime" in df.columns and "pickup_hour" not in df.columns:
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
        df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek
        df["pickup_month"] = df["tpep_pickup_datetime"].dt.month

    if "trip_distance" in df.columns:
        df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]
    if "passenger_count" in df.columns:
        df = df[(df["passenger_count"] >= 0) & (df["passenger_count"] <= 9)]

    cols = [c for c in FEATURE_COLS if c in df.columns]
    return df[cols].dropna()


def extract_metrics(result):
    """Extract drift metrics from Evidently 0.7.x report."""
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
            feature_scores[column] = {
                "drift_score": p_value,
                "drifted": 1 if p_value < threshold else 0,
            }
            n_columns += 1

    dataset_drift = drift_share > 0.5

    return {
        "evidently_dataset_drift": 1 if dataset_drift else 0,
        "evidently_drift_share": drift_share,
        "evidently_number_of_drifted_columns": n_drifted,
        "evidently_number_of_columns": n_columns,
    }, feature_scores


def push_to_vm(metrics, feature_metrics):
    lines = []
    for name, value in metrics.items():
        lines.append(f"{name} {value}")
    for feature, values in feature_metrics.items():
        lines.append(f'evidently_feature_drift_score{{feature="{feature}"}} {values["drift_score"]}')
        lines.append(f'evidently_feature_drifted{{feature="{feature}"}} {values["drifted"]}')

    body = "\n".join(lines) + "\n"
    resp = requests.post(
        f"{VM_URL}/api/v1/import/prometheus",
        data=body, headers={"Content-Type": "text/plain"}, timeout=10,
    )
    resp.raise_for_status()
    log.info("Metrics pushed to VictoriaMetrics")


def save_report(s3, result):
    ensure_bucket(s3, REPORTS_BUCKET)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    html_path = f"/tmp/drift_report_{timestamp}.html"
    result.save_html(html_path)
    with open(html_path, "r") as f:
        html = f.read()
    s3.put_object(
        Bucket=REPORTS_BUCKET, Key=f"drift/{timestamp}.html",
        Body=html.encode(), ContentType="text/html",
    )
    log.info("Report saved to s3://%s/drift/%s.html", REPORTS_BUCKET, timestamp)


def trigger_retrain():
    try:
        resp = requests.post(
            f"{AIRFLOW_URL}/api/v1/dags/taxi_auto_retrain/dagRuns",
            json={"conf": {"triggered_by": "drift_monitor"}},
            auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        log.info("Retrain DAG triggered")
    except Exception as e:
        log.error("Failed to trigger retrain: %s", e)


def main():
    s3 = get_s3()

    # Reference: silver/trips (training data)
    log.info("Loading reference from %s", SILVER_TRIPS)
    reference = read_delta(SILVER_TRIPS)
    ref_features = prepare_features(reference)
    log.info("Reference: %d rows", len(ref_features))

    # Current: bronze/streaming (live data)
    log.info("Loading current from %s", BRONZE_STREAMING)
    try:
        current_raw = read_delta(BRONZE_STREAMING)
    except Exception as e:
        log.warning("No streaming data yet: %s", e)
        return

    curr_features = prepare_features(current_raw)
    log.info("Current: %d rows", len(curr_features))

    if len(curr_features) < 100:
        log.warning("Current data too small (%d rows), skipping", len(curr_features))
        return

    report = Report([DataDriftPreset()])
    result = report.run(ref_features, curr_features)

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
        log.info("Drift share %.2f >= %.2f, triggering retrain",
                 metrics["evidently_drift_share"], RETRAIN_THRESHOLD)
        trigger_retrain()

    log.info("Drift monitoring complete!")


if __name__ == "__main__":
    main()
