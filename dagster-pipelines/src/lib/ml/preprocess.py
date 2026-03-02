"""Feature engineering for NYC Taxi trip prediction models."""

import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.lib.lake import read_delta, table_uri, write_delta
from src.lib.quality import validate_and_push

log = logging.getLogger(__name__)

BRONZE_RAW = table_uri("bronze", "raw")
BRONZE_STREAMING = table_uri("bronze", "streaming")
SILVER_TRIPS = table_uri("silver", "trips")


def load_source_data(data_source="raw"):
    if data_source == "combined":
        log.info("Loading combined: bronze/raw + bronze/streaming")
        raw = read_delta(BRONZE_RAW)
        streaming = read_delta(BRONZE_STREAMING)
        return pd.concat([raw, streaming], ignore_index=True)
    elif data_source == "streaming":
        log.info("Loading: bronze/streaming")
        return read_delta(BRONZE_STREAMING)
    else:
        log.info("Loading: bronze/raw")
        return read_delta(BRONZE_RAW)


def engineer_features(df):
    df = df.copy()
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    df["duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

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


def run(data_source="raw"):
    log.info("Data source: %s", data_source)
    df = load_source_data(data_source)
    log.info("Total rows: %d", len(df))

    log.info("Engineering features...")
    df, feature_cols, target_cols = engineer_features(df)
    df = df.dropna(subset=feature_cols + target_cols)
    log.info("After dropping NaN: %d rows", len(df))

    log.info("Removing outliers...")
    df = remove_outliers(df)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    log.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    validate_and_push(train_df, "silver_trips")

    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined["split"] = ["train"] * len(train_df) + ["test"] * len(test_df)
    write_delta(combined, SILVER_TRIPS, mode="overwrite")

    log.info("Preprocessing complete → %s", SILVER_TRIPS)
    return combined
