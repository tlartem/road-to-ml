"""Compute zone-level aggregated features from bronze data.

Uses DuckDB to aggregate directly from Delta table without loading all rows into pandas.
"""

import logging
import os

import duckdb
import pandas as pd
from deltalake import DeltaTable

from src.lib.lake import storage_options, table_uri, write_delta
from src.lib.quality import validate_and_push

log = logging.getLogger(__name__)

BRONZE_RAW = table_uri("bronze", "raw")
GOLD_ZONE_STATS = table_uri("gold", "zone_stats")
FEAST_DATA_DIR = "/feast-data/features"


def run():
    log.info("Computing zone stats from %s via DuckDB", BRONZE_RAW)

    dt = DeltaTable(BRONZE_RAW, storage_options=storage_options())
    arrow_dataset = dt.to_pyarrow_dataset()

    con = duckdb.connect()
    con.register("bronze_raw", arrow_dataset)

    zone_stats = con.execute("""
        WITH trips AS (
            SELECT
                "PULocationID" AS zone_id,
                total_amount,
                trip_distance,
                EPOCH(CAST(tpep_dropoff_datetime AS TIMESTAMP) - CAST(tpep_pickup_datetime AS TIMESTAMP)) / 60.0 AS duration_min,
                tpep_pickup_datetime
            FROM bronze_raw
            WHERE trip_distance > 0 AND trip_distance < 100
              AND total_amount > 0 AND total_amount < 500
              AND "PULocationID" >= 1 AND "PULocationID" <= 263
        ),
        filtered AS (
            SELECT * FROM trips
            WHERE duration_min >= 1 AND duration_min <= 180
        )
        SELECT
            zone_id,
            ROUND(AVG(total_amount), 2) AS zone_avg_fare,
            ROUND(AVG(duration_min), 2) AS zone_avg_duration_min,
            ROUND(AVG(trip_distance), 2) AS zone_avg_distance,
            COUNT(*) AS zone_trip_count,
            MAX(tpep_pickup_datetime)::DATE::TIMESTAMP AS event_timestamp
        FROM filtered
        GROUP BY zone_id
        ORDER BY zone_id
    """).fetchdf()

    con.close()

    log.info("Computed stats for %d zones", len(zone_stats))

    validate_and_push(zone_stats, "gold_zone_stats")
    write_delta(zone_stats, GOLD_ZONE_STATS, mode="overwrite")

    os.makedirs(FEAST_DATA_DIR, exist_ok=True)
    pvc_path = os.path.join(FEAST_DATA_DIR, "zone_stats.parquet")
    zone_stats.to_parquet(pvc_path, index=False)
    log.info("Saved to PVC: %s", pvc_path)

    log.info("Feature engineering complete → %s", GOLD_ZONE_STATS)
    return zone_stats
