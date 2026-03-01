"""Feature engineering for NYC Taxi trip prediction models.

Reads raw Parquet from MinIO taxi-raw/, creates features, removes outliers,
saves train/test to taxi-processed/ for duration and fare models.
"""

import io
import logging
import os
import sys

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

RAW_BUCKET = "taxi-raw"
PROCESSED_BUCKET = "taxi-processed"
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


def engineer_features(df):
    """Create features from raw taxi data. Keeps both duration and fare targets."""
    df = df.copy()

    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    # Targets
    df["duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

    # Time features
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["tpep_pickup_datetime"].dt.month

    df = df.rename(columns={
        "PULocationID": "pickup_zone_id",
        "DOLocationID": "dropoff_zone_id",
    })

    feature_cols = [
        "pickup_zone_id",
        "dropoff_zone_id",
        "trip_distance",
        "pickup_hour",
        "pickup_day_of_week",
        "pickup_month",
        "passenger_count",
    ]
    target_cols = ["duration_min", "total_amount"]

    keep_cols = feature_cols + target_cols
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    return df, feature_cols, target_cols


def remove_outliers(df):
    before = len(df)

    df = df[(df["duration_min"] >= 1) & (df["duration_min"] <= 180)]
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]
    df = df[(df["total_amount"] >= 1) & (df["total_amount"] <= 500)]

    if "passenger_count" in df.columns:
        df = df[(df["passenger_count"] >= 0) & (df["passenger_count"] <= 9)]

    after = len(df)
    log.info("Removed %d outliers (%.1f%%), kept %d rows", before - after, 100 * (before - after) / before, after)
    return df


def upload_parquet(s3, df, key):
    buf = io.BytesIO()
    table = pa.Table.from_pandas(df)
    pq.write_table(table, buf)
    buf.seek(0)
    s3.put_object(Bucket=PROCESSED_BUCKET, Key=key, Body=buf.getvalue())
    log.info("Uploaded s3://%s/%s", PROCESSED_BUCKET, key)


def main():
    s3 = get_s3()
    ensure_bucket(s3, PROCESSED_BUCKET)

    log.info("Loading raw data for months: %s", TRAIN_MONTHS)
    df = load_raw_data(s3)
    log.info("Total raw rows: %d", len(df))

    log.info("Engineering features...")
    df, feature_cols, target_cols = engineer_features(df)

    df = df.dropna(subset=feature_cols + target_cols)
    log.info("After dropping NaN: %d rows", len(df))

    log.info("Removing outliers...")
    df = remove_outliers(df)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    log.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    # Save shared dataset (used by both duration and fare models)
    for name, data in [("train.parquet", train_df), ("test.parquet", test_df)]:
        upload_parquet(s3, data, f"trips/{name}")
    log.info("Saved to s3://%s/trips/", PROCESSED_BUCKET)

    # Also save to legacy path for backward compat
    for name, data in [("train.parquet", train_df), ("test.parquet", test_df)]:
        upload_parquet(s3, data, f"duration/{name}")

    log.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
