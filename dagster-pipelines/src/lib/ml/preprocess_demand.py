"""Preprocess data for demand prediction model.

Uses DuckDB to aggregate directly from Delta tables.
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
SILVER_DEMAND = table_uri("silver", "demand")


def _aggregate_via_duckdb(uri):
    """Aggregate demand (zone, date, hour) → trip_count via DuckDB."""
    dt = DeltaTable(uri, storage_options=storage_options())
    arrow_ds = dt.to_pyarrow_dataset()

    con = duckdb.connect()
    con.register("src", arrow_ds)

    df = con.execute("""
        SELECT
            "PULocationID" AS zone_id,
            HOUR(CAST(tpep_pickup_datetime AS TIMESTAMP)) AS pickup_hour,
            DAYOFWEEK(CAST(tpep_pickup_datetime AS TIMESTAMP)) AS day_of_week,
            MONTH(CAST(tpep_pickup_datetime AS TIMESTAMP)) AS month,
            COUNT(*) AS trip_count
        FROM src
        WHERE "PULocationID" BETWEEN 1 AND 263
        GROUP BY zone_id,
                 HOUR(CAST(tpep_pickup_datetime AS TIMESTAMP)),
                 DAYOFWEEK(CAST(tpep_pickup_datetime AS TIMESTAMP)),
                 MONTH(CAST(tpep_pickup_datetime AS TIMESTAMP)),
                 CAST(tpep_pickup_datetime AS DATE)
    """).fetchdf()

    con.close()

    # Fix dayofweek: DuckDB returns 0=Sunday, pandas uses 0=Monday
    df["day_of_week"] = (df["day_of_week"] + 6) % 7

    return df


def run(data_source="raw"):
    log.info("Data source: %s", data_source)

    frames = []
    if data_source in ("raw", "combined"):
        log.info("Aggregating demand from bronze/raw via DuckDB...")
        frames.append(_aggregate_via_duckdb(BRONZE_RAW))

    if data_source in ("streaming", "combined"):
        if table_exists(BRONZE_STREAMING):
            log.info("Aggregating demand from bronze/streaming via DuckDB...")
            frames.append(_aggregate_via_duckdb(BRONZE_STREAMING))

    demand = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    # Re-aggregate if combined (same zone-date-hour from both sources)
    if len(frames) > 1:
        demand = demand.groupby(["zone_id", "pickup_hour", "day_of_week", "month"]).agg(
            trip_count=("trip_count", "sum")
        ).reset_index()

    log.info("Demand data: %d rows", len(demand))

    train_df, test_df = train_test_split(demand, test_size=0.2, random_state=42)
    log.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    validate_and_push(train_df, "silver_demand")

    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined["split"] = ["train"] * len(train_df) + ["test"] * len(test_df)
    write_delta(combined, SILVER_DEMAND, mode="overwrite")

    log.info("Demand preprocessing complete → %s", SILVER_DEMAND)
    return combined
