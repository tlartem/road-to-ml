"""Feature engineering for NYC Taxi trip duration prediction.

Reads raw Parquet from MinIO taxi-streaming/ (or taxi-raw/ for initial training),
creates features, removes outliers, saves train/test to taxi-processed/.
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
# Use first 2 months for training by default
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
    """Load raw Parquet files for training months."""
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
    """Create features from raw taxi data."""
    df = df.copy()

    # Parse datetime
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    # Target: trip duration in minutes
    df["duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

    # Time features
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["tpep_pickup_datetime"].dt.month

    # Rename zone columns for clarity
    df = df.rename(columns={
        "PULocationID": "pickup_zone_id",
        "DOLocationID": "dropoff_zone_id",
    })

    # Select features + target
    feature_cols = [
        "pickup_zone_id",
        "dropoff_zone_id",
        "trip_distance",
        "pickup_hour",
        "pickup_day_of_week",
        "pickup_month",
        "passenger_count",
    ]
    target_col = "duration_min"

    # Keep only needed columns
    keep_cols = feature_cols + [target_col]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    return df, feature_cols, target_col


def remove_outliers(df):
    """Remove unrealistic records."""
    before = len(df)

    # Duration: 1 min to 180 min (3 hours)
    df = df[(df["duration_min"] >= 1) & (df["duration_min"] <= 180)]

    # Distance: > 0 and < 100 miles
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]

    # Passenger count: 0-9 (0 is valid — recorded trips with no passenger count)
    if "passenger_count" in df.columns:
        df = df[(df["passenger_count"] >= 0) & (df["passenger_count"] <= 9)]

    after = len(df)
    log.info("Removed %d outliers (%.1f%%), kept %d rows", before - after, 100 * (before - after) / before, after)
    return df


def main():
    s3 = get_s3()
    ensure_bucket(s3, PROCESSED_BUCKET)

    log.info("Loading raw data for months: %s", TRAIN_MONTHS)
    df = load_raw_data(s3)
    log.info("Total raw rows: %d", len(df))

    log.info("Engineering features...")
    df, feature_cols, target_col = engineer_features(df)

    # Drop rows with NaN in features or target
    df = df.dropna(subset=feature_cols + [target_col])
    log.info("After dropping NaN: %d rows", len(df))

    log.info("Removing outliers...")
    df = remove_outliers(df)

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    log.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    # Upload as Parquet
    for name, data in [("train.parquet", train_df), ("test.parquet", test_df)]:
        buf = io.BytesIO()
        table = pa.Table.from_pandas(data)
        pq.write_table(table, buf)
        buf.seek(0)
        s3.put_object(Bucket=PROCESSED_BUCKET, Key=f"duration/{name}", Body=buf.getvalue())
        log.info("Uploaded s3://%s/duration/%s", PROCESSED_BUCKET, name)

    log.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
