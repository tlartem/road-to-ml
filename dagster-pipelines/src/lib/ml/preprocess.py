"""Feature engineering for NYC Taxi trip prediction models.

Uses DuckDB to read and transform Delta tables without loading all raw data into pandas.
"""

import logging

import duckdb
import pandas as pd
from deltalake import DeltaTable
from sklearn.model_selection import train_test_split

from src.lib.lake import storage_options, table_uri, write_delta, table_exists
from src.lib.quality import validate_and_push

log = logging.getLogger(__name__)

BRONZE_RAW = table_uri("bronze", "raw")
BRONZE_STREAMING = table_uri("bronze", "streaming")
SILVER_TRIPS = table_uri("silver", "trips")


def _read_via_duckdb(uri):
    """Read Delta table and do feature engineering + outlier removal via DuckDB."""
    dt = DeltaTable(uri, storage_options=storage_options())
    arrow_ds = dt.to_pyarrow_dataset()

    con = duckdb.connect()
    con.register("src", arrow_ds)

    df = con.execute("""
        SELECT
            "PULocationID" AS pickup_zone_id,
            "DOLocationID" AS dropoff_zone_id,
            trip_distance,
            HOUR(CAST(tpep_pickup_datetime AS TIMESTAMP)) AS pickup_hour,
            DAYOFWEEK(CAST(tpep_pickup_datetime AS TIMESTAMP)) AS pickup_day_of_week,
            MONTH(CAST(tpep_pickup_datetime AS TIMESTAMP)) AS pickup_month,
            passenger_count,
            EPOCH(CAST(tpep_dropoff_datetime AS TIMESTAMP) - CAST(tpep_pickup_datetime AS TIMESTAMP)) / 60.0 AS duration_min,
            total_amount
        FROM src
        WHERE trip_distance > 0 AND trip_distance < 100
          AND total_amount >= 1 AND total_amount <= 500
          AND EPOCH(CAST(tpep_dropoff_datetime AS TIMESTAMP) - CAST(tpep_pickup_datetime AS TIMESTAMP)) / 60.0 >= 1
          AND EPOCH(CAST(tpep_dropoff_datetime AS TIMESTAMP) - CAST(tpep_pickup_datetime AS TIMESTAMP)) / 60.0 <= 180
          AND "PULocationID" BETWEEN 1 AND 263
          AND "DOLocationID" BETWEEN 1 AND 263
    """).fetchdf()

    con.close()

    # Fix dayofweek: DuckDB returns 0=Sunday, pandas uses 0=Monday
    df["pickup_day_of_week"] = (df["pickup_day_of_week"] + 6) % 7

    if "passenger_count" in df.columns:
        df = df[(df["passenger_count"] >= 0) & (df["passenger_count"] <= 9)]

    df = df.dropna()
    return df


def run(data_source="raw"):
    log.info("Data source: %s", data_source)

    frames = []
    if data_source in ("raw", "combined"):
        log.info("Reading bronze/raw via DuckDB...")
        frames.append(_read_via_duckdb(BRONZE_RAW))

    if data_source in ("streaming", "combined"):
        if table_exists(BRONZE_STREAMING):
            log.info("Reading bronze/streaming via DuckDB...")
            frames.append(_read_via_duckdb(BRONZE_STREAMING))

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    log.info("Total cleaned rows: %d", len(df))

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    log.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    validate_and_push(train_df, "silver_trips")

    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined["split"] = ["train"] * len(train_df) + ["test"] * len(test_df)
    write_delta(combined, SILVER_TRIPS, mode="overwrite")

    log.info("Preprocessing complete → %s", SILVER_TRIPS)
    return combined
