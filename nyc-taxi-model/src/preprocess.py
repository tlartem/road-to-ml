"""Feature engineering for NYC Taxi trip prediction models.

Reads from Delta Lake bronze → cleans, engineers features, removes outliers →
writes to Delta Lake silver/trips.

DATA_SOURCE env var:
  "raw"       — bronze/raw only (initial training)
  "streaming" — bronze/streaming only (retrain on fresh data)
  "combined"  — bronze/raw + bronze/streaming (retrain with full history)
"""

import logging
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from lake import read_delta, table_uri, write_delta
from quality import validate_and_push

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

DATA_SOURCE = os.environ.get("DATA_SOURCE", "raw")

BRONZE_RAW = table_uri("bronze", "raw")
BRONZE_STREAMING = table_uri("bronze", "streaming")
SILVER_TRIPS = table_uri("silver", "trips")


def load_source_data():
    """Load data from the configured source."""
    if DATA_SOURCE == "combined":
        log.info("Loading combined: bronze/raw + bronze/streaming")
        raw = read_delta(BRONZE_RAW)
        streaming = read_delta(BRONZE_STREAMING)
        return pd.concat([raw, streaming], ignore_index=True)
    elif DATA_SOURCE == "streaming":
        log.info("Loading: bronze/streaming")
        return read_delta(BRONZE_STREAMING)
    else:
        log.info("Loading: bronze/raw")
        return read_delta(BRONZE_RAW)


def engineer_features(df):
    """Create features from raw taxi data."""
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
        "pickup_zone_id", "dropoff_zone_id", "trip_distance",
        "pickup_hour", "pickup_day_of_week", "pickup_month",
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
    log.info("Removed %d outliers (%.1f%%), kept %d rows",
             before - after, 100 * (before - after) / before, after)
    return df


def main():
    log.info("Data source: %s", DATA_SOURCE)
    df = load_source_data()
    log.info("Total rows: %d", len(df))

    log.info("Engineering features...")
    df, feature_cols, target_cols = engineer_features(df)

    df = df.dropna(subset=feature_cols + target_cols)
    log.info("After dropping NaN: %d rows", len(df))

    log.info("Removing outliers...")
    df = remove_outliers(df)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    log.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    # Validate silver data
    validate_and_push(train_df, "silver_trips")

    # Write to Delta Lake silver/trips (overwrite — full rebuild)
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined["split"] = ["train"] * len(train_df) + ["test"] * len(test_df)
    write_delta(combined, SILVER_TRIPS, mode="overwrite")

    log.info("Preprocessing complete → %s", SILVER_TRIPS)


if __name__ == "__main__":
    main()
