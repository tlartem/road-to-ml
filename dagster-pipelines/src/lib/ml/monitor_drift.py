"""Drift monitoring with Evidently.

Compares reference (sample from silver/trips) vs current (bronze/streaming
preprocessed the same way) using the same feature columns.
Both datasets go through identical feature engineering so distributions
are comparable.
"""

import json
import logging
import os
from datetime import datetime

import boto3
import duckdb
import requests
from deltalake import DeltaTable
from evidently import Report
from evidently.presets import DataDriftPreset

from src.lib.lake import storage_options, table_uri, table_exists

log = logging.getLogger(__name__)

SILVER_TRIPS = table_uri("silver", "trips")
BRONZE_STREAMING = table_uri("bronze", "streaming")
REPORTS_BUCKET = "taxi-reports"
VM_URL = os.environ.get("VM_URL", "http://victoriametrics.monitoring.svc.cluster.local:8428")
RETRAIN_THRESHOLD = float(os.environ.get("RETRAIN_THRESHOLD", "0.5"))
REF_SAMPLE_ROWS = int(os.environ.get("REF_SAMPLE_ROWS", "10000"))

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


def _load_reference():
    """Load a random sample from silver/trips via DuckDB.

    silver_trips already has preprocessed features (pickup_hour, etc.)
    so we just sample the columns we need.
    """
    opts = storage_options()
    dt = DeltaTable(SILVER_TRIPS, storage_options=opts)
    ds = dt.to_pyarrow_dataset()

    con = duckdb.connect()
    con.register("trips", ds)

    cols = ", ".join(FEATURE_COLS)
    df = con.execute(f"""
        SELECT {cols}
        FROM trips
        WHERE trip_distance > 0 AND trip_distance < 100
          AND passenger_count >= 0 AND passenger_count <= 9
        USING SAMPLE {REF_SAMPLE_ROWS} ROWS
    """).fetchdf()

    con.close()

    # Fix dayofweek if needed (silver_trips already has correct values)
    return df.dropna()


def _load_current():
    """Load bronze/streaming and apply same feature engineering as preprocess.py.

    bronze_streaming has raw columns (tpep_pickup_datetime, PULocationID, etc.)
    We extract the same features as silver_trips to make distributions comparable.
    """
    opts = storage_options()
    dt = DeltaTable(BRONZE_STREAMING, storage_options=opts)
    ds = dt.to_pyarrow_dataset()

    con = duckdb.connect()
    con.register("streaming", ds)

    df = con.execute("""
        SELECT
            trip_distance,
            HOUR(CAST(tpep_pickup_datetime AS TIMESTAMP)) AS pickup_hour,
            DAYOFWEEK(CAST(tpep_pickup_datetime AS TIMESTAMP)) AS pickup_day_of_week,
            MONTH(CAST(tpep_pickup_datetime AS TIMESTAMP)) AS pickup_month,
            passenger_count
        FROM streaming
        WHERE trip_distance > 0 AND trip_distance < 100
          AND passenger_count >= 0 AND passenger_count <= 9
          AND "PULocationID" BETWEEN 1 AND 263
    """).fetchdf()

    con.close()

    # Fix dayofweek: DuckDB 0=Sunday → 0=Monday (same as preprocess.py)
    df["pickup_day_of_week"] = (df["pickup_day_of_week"] + 6) % 7

    return df.dropna()


def extract_metrics(result):
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
    try:
        resp = requests.post(
            f"{VM_URL}/api/v1/import/prometheus",
            data=body, headers={"Content-Type": "text/plain"}, timeout=10,
        )
        resp.raise_for_status()
        log.info("Metrics pushed to VictoriaMetrics")
    except Exception as e:
        log.warning("Failed to push drift metrics: %s", e)


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


def run():
    """Run drift monitoring. Returns (metrics_dict, drift_detected_bool)."""
    s3 = get_s3()

    log.info("Loading reference sample (%d rows) from %s", REF_SAMPLE_ROWS, SILVER_TRIPS)
    ref_features = _load_reference()
    log.info("Reference: %d rows, columns: %s", len(ref_features), list(ref_features.columns))

    if not table_exists(BRONZE_STREAMING):
        log.warning("No streaming data yet, skipping drift check")
        return {}, False

    log.info("Loading current from %s", BRONZE_STREAMING)
    try:
        curr_features = _load_current()
    except Exception as e:
        log.warning("Failed to load streaming data: %s", e)
        return {}, False

    log.info("Current: %d rows, columns: %s", len(curr_features), list(curr_features.columns))

    if len(curr_features) < 100:
        log.warning("Current data too small (%d rows), skipping", len(curr_features))
        return {}, False

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

    drift_detected = metrics["evidently_drift_share"] >= RETRAIN_THRESHOLD
    if drift_detected:
        log.info("Drift share %.2f >= %.2f, retrain needed",
                 metrics["evidently_drift_share"], RETRAIN_THRESHOLD)

    log.info("Drift monitoring complete!")
    return metrics, drift_detected
