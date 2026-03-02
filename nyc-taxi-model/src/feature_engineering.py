"""Compute zone-level aggregated features from bronze data.

Reads from Delta Lake bronze/raw → computes per-zone statistics →
writes to Delta Lake gold/zone_stats + PVC (for Feast).
"""

import logging
import os
import sys

import pandas as pd

from lake import read_delta, table_uri, write_delta
from quality import validate_and_push

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

BRONZE_RAW = table_uri("bronze", "raw")
GOLD_ZONE_STATS = table_uri("gold", "zone_stats")
FEAST_DATA_DIR = "/feast-data/features"


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
        & (df["PULocationID"] >= 1) & (df["PULocationID"] <= 263)
    )
    df = df[mask]
    log.info("Filtered to %d reasonable trips", len(df))

    stats = df.groupby("PULocationID").agg(
        zone_avg_fare=("total_amount", "mean"),
        zone_avg_duration_min=("duration_min", "mean"),
        zone_avg_distance=("trip_distance", "mean"),
        zone_trip_count=("PULocationID", "count"),
    ).reset_index()

    stats = stats.rename(columns={"PULocationID": "zone_id"})

    stats["event_timestamp"] = pd.Timestamp(
        df["tpep_pickup_datetime"].max().date()
    )

    for col in ["zone_avg_fare", "zone_avg_duration_min", "zone_avg_distance"]:
        stats[col] = stats[col].round(2)

    log.info("Computed stats for %d zones", len(stats))
    return stats


def main():
    log.info("Loading data from %s", BRONZE_RAW)
    df = read_delta(BRONZE_RAW)
    log.info("Total raw rows: %d", len(df))

    zone_stats = compute_zone_stats(df)

    # Validate
    validate_and_push(zone_stats, "gold_zone_stats")

    # Write to Delta Lake gold/zone_stats
    write_delta(zone_stats, GOLD_ZONE_STATS, mode="overwrite")

    # Also save to PVC (for Feast server)
    os.makedirs(FEAST_DATA_DIR, exist_ok=True)
    pvc_path = os.path.join(FEAST_DATA_DIR, "zone_stats.parquet")
    zone_stats.to_parquet(pvc_path, index=False)
    log.info("Saved to PVC: %s", pvc_path)

    log.info("Feature engineering complete → %s", GOLD_ZONE_STATS)


if __name__ == "__main__":
    main()
