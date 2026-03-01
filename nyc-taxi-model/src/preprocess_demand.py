"""Preprocess data for demand prediction model.

Aggregates trips by (zone_id, date, hour) → trip_count.
Features: zone_id, hour, day_of_week, month.
Target: trip_count.
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

    # Drop date (not a model feature)
    demand = demand.drop("pickup_date", axis=1)

    log.info("Demand data: %d rows (zone-hour combinations)", len(demand))
    log.info("Trip count stats:\n%s", demand["trip_count"].describe())
    return demand


def main():
    s3 = get_s3()

    log.info("Loading raw data for months: %s", TRAIN_MONTHS)
    df = load_raw_data(s3)
    log.info("Total raw rows: %d", len(df))

    log.info("Aggregating demand...")
    demand = aggregate_demand(df)

    train_df, test_df = train_test_split(demand, test_size=0.2, random_state=42)
    log.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    for name, data in [("train.parquet", train_df), ("test.parquet", test_df)]:
        buf = io.BytesIO()
        table = pa.Table.from_pandas(data)
        pq.write_table(table, buf)
        buf.seek(0)
        s3.put_object(Bucket=PROCESSED_BUCKET, Key=f"demand/{name}", Body=buf.getvalue())
        log.info("Uploaded s3://%s/demand/%s", PROCESSED_BUCKET, name)

    log.info("Demand preprocessing complete!")


if __name__ == "__main__":
    main()
