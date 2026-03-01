"""Compute zone-level aggregated features from raw NYC Taxi data.

Reads raw Parquet from MinIO taxi-raw/, computes per-zone statistics,
saves to:
  - /feast-data/features/zone_stats.parquet (PVC, for Feast)
  - s3://taxi-processed/feast/zone_stats.parquet (MinIO, for training)
"""

import io
import logging
import os
import sys
from datetime import datetime

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

RAW_BUCKET = "taxi-raw"
PROCESSED_BUCKET = "taxi-processed"
FEAST_DATA_DIR = "/feast-data/features"
TRAIN_MONTHS = os.environ.get("TRAIN_MONTHS", "2024-01,2024-02").split(",")


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


def load_raw_data(s3):
    frames = []
    for month in TRAIN_MONTHS:
        key = f"yellow_tripdata_{month}.parquet"
        log.info("Loading s3://%s/%s ...", RAW_BUCKET, key)
        resp = s3.get_object(Bucket=RAW_BUCKET, Key=key)
        df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
        log.info("  %d rows", len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_zone_stats(df):
    """Compute per-zone aggregated statistics."""
    df = df.copy()
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    df["duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

    # Filter reasonable trips
    mask = (
        (df["duration_min"] >= 1) & (df["duration_min"] <= 180)
        & (df["trip_distance"] > 0) & (df["trip_distance"] < 100)
        & (df["total_amount"] > 0) & (df["total_amount"] < 500)
    )
    df = df[mask]
    log.info("Filtered to %d reasonable trips", len(df))

    # Aggregate by pickup zone
    stats = df.groupby("PULocationID").agg(
        zone_avg_fare=("total_amount", "mean"),
        zone_avg_duration_min=("duration_min", "mean"),
        zone_avg_distance=("trip_distance", "mean"),
        zone_trip_count=("PULocationID", "count"),
    ).reset_index()

    stats = stats.rename(columns={"PULocationID": "zone_id"})

    # Add event_timestamp (end of training period)
    stats["event_timestamp"] = pd.Timestamp(
        df["tpep_pickup_datetime"].max().date()
    )

    # Round floats
    for col in ["zone_avg_fare", "zone_avg_duration_min", "zone_avg_distance"]:
        stats[col] = stats[col].round(2)

    log.info("Computed stats for %d zones", len(stats))
    log.info("Sample:\n%s", stats.head())
    return stats


def main():
    s3 = get_s3()
    ensure_bucket(s3, PROCESSED_BUCKET)

    log.info("Loading raw data for months: %s", TRAIN_MONTHS)
    df = load_raw_data(s3)
    log.info("Total raw rows: %d", len(df))

    zone_stats = compute_zone_stats(df)

    # Save to PVC (for Feast)
    os.makedirs(FEAST_DATA_DIR, exist_ok=True)
    pvc_path = os.path.join(FEAST_DATA_DIR, "zone_stats.parquet")
    zone_stats.to_parquet(pvc_path, index=False)
    log.info("Saved to PVC: %s", pvc_path)

    # Save to MinIO (for training scripts)
    buf = io.BytesIO()
    table = pa.Table.from_pandas(zone_stats)
    pq.write_table(table, buf)
    buf.seek(0)
    s3.put_object(Bucket=PROCESSED_BUCKET, Key="feast/zone_stats.parquet", Body=buf.getvalue())
    log.info("Saved to s3://%s/feast/zone_stats.parquet", PROCESSED_BUCKET)

    log.info("Feature engineering complete!")


if __name__ == "__main__":
    main()
