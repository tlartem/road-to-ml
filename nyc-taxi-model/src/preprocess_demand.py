"""Preprocess data for demand prediction model.

Reads from Delta Lake bronze → aggregates by (zone, date, hour) →
writes to Delta Lake silver/demand.

DATA_SOURCE: "raw", "streaming", or "combined"
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
SILVER_DEMAND = table_uri("silver", "demand")


def load_source_data():
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


def aggregate_demand(df):
    """Aggregate trips by (zone, date, hour) → trip_count."""
    df = df.copy()
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour

    demand = (
        df.groupby(["PULocationID", "pickup_date", "pickup_hour"])
        .size()
        .reset_index(name="trip_count")
    )
    demand = demand.rename(columns={"PULocationID": "zone_id"})

    # Time features
    demand["pickup_date"] = pd.to_datetime(demand["pickup_date"])
    demand["day_of_week"] = demand["pickup_date"].dt.dayofweek
    demand["month"] = demand["pickup_date"].dt.month

    demand = demand.drop("pickup_date", axis=1)

    log.info("Demand data: %d rows (zone-hour combinations)", len(demand))
    return demand


def main():
    log.info("Data source: %s", DATA_SOURCE)
    df = load_source_data()
    log.info("Total raw rows: %d", len(df))

    log.info("Aggregating demand...")
    demand = aggregate_demand(df)

    train_df, test_df = train_test_split(demand, test_size=0.2, random_state=42)
    log.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    # Validate
    validate_and_push(train_df, "silver_demand")

    # Write to Delta Lake
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined["split"] = ["train"] * len(train_df) + ["test"] * len(test_df)
    write_delta(combined, SILVER_DEMAND, mode="overwrite")

    log.info("Demand preprocessing complete → %s", SILVER_DEMAND)


if __name__ == "__main__":
    main()
