"""Simulate streaming data: read from bronze/raw → append to bronze/streaming.

Each invocation advances the virtual clock by 1 hour:
  1. Read state.json from MinIO (current date + hour offset)
  2. Select rows matching that virtual hour from bronze/raw
  3. Append batch to bronze/streaming Delta table (partitioned by date)
  4. Update state.json
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta

import boto3
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
BRONZE_STREAMING = table_uri("bronze", "streaming")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1000"))
STATE_BUCKET = "taxi-lake"
STATE_KEY = "state.json"


def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def load_state(s3):
    try:
        resp = s3.get_object(Bucket=STATE_BUCKET, Key=STATE_KEY)
        state = json.loads(resp["Body"].read().decode())
        log.info("Loaded state: %s", state)
        return state
    except s3.exceptions.ClientError:
        return None


def save_state(s3, state):
    body = json.dumps(state, indent=2).encode()
    s3.put_object(Bucket=STATE_BUCKET, Key=STATE_KEY, Body=body)
    log.info("Saved state: %s", state)


def push_freshness(table_name):
    """Push freshness metric (seconds since epoch) to VictoriaMetrics."""
    import requests
    vm_url = os.environ.get("VM_URL", "http://victoriametrics.monitoring.svc.cluster.local:8428")
    try:
        requests.post(
            f"{vm_url}/api/v1/import/prometheus",
            data=f'data_freshness_timestamp{{table="{table_name}"}} {int(datetime.utcnow().timestamp())}\n',
            headers={"Content-Type": "text/plain"},
            timeout=5,
        )
    except Exception as e:
        log.warning("Failed to push freshness metric: %s", e)


def main():
    s3 = get_s3()

    # Determine simulation date
    jump_date = os.environ.get("SIMULATION_DATE", "")
    state = load_state(s3)

    if jump_date:
        current_date = jump_date
        hour_offset = 0
        log.info("Jump to date: %s", current_date)
    elif state:
        current_date = state["current_date"]
        hour_offset = state["hour_offset"]
    else:
        current_date = "2024-01-01"
        hour_offset = 0
        log.info("First run, starting from %s", current_date)

    # Virtual datetime
    base_dt = datetime.strptime(current_date, "%Y-%m-%d")
    virtual_dt = base_dt + timedelta(hours=hour_offset)
    log.info("Virtual datetime: %s (hour_offset=%d)", virtual_dt, hour_offset)

    # Read from bronze/raw Delta table
    log.info("Reading from %s", BRONZE_RAW)
    df = read_delta(BRONZE_RAW)

    # Filter rows for this virtual hour
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    target_date = virtual_dt.date()
    target_hour = virtual_dt.hour

    mask = (
        (df["tpep_pickup_datetime"].dt.date == target_date)
        & (df["tpep_pickup_datetime"].dt.hour == target_hour)
    )
    batch = df[mask].head(BATCH_SIZE).copy()

    if len(batch) == 0:
        log.warning("No data for %s hour %d, advancing to next hour", target_date, target_hour)
    else:
        log.info("Selected %d rows for %s %02d:00", len(batch), target_date, target_hour)

        # Add partition column
        batch["date"] = virtual_dt.strftime("%Y-%m-%d")

        # Validate batch
        validate_and_push(batch, "bronze_streaming")

        # Append to bronze/streaming Delta table
        write_delta(batch, BRONZE_STREAMING, mode="append", partition_by=["date"])

        push_freshness("bronze.streaming")

    # Advance state
    next_offset = hour_offset + 1
    next_dt = base_dt + timedelta(hours=next_offset)
    new_state = {
        "current_date": current_date,
        "hour_offset": next_offset,
        "last_virtual_dt": next_dt.isoformat(),
    }
    save_state(s3, new_state)

    log.info("Simulator step complete. Next virtual datetime: %s", next_dt)


if __name__ == "__main__":
    main()
