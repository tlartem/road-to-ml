"""Simulate streaming data by reading from taxi-raw/ and writing batches to taxi-streaming/.

Environment variables:
  SIMULATION_DATE  — date to simulate, e.g. "2024-01-15" (read from state or env)
  BATCH_SIZE       — rows per batch (default 1000)

Each invocation advances the virtual clock by 1 hour:
  - Reads state.json from taxi-streaming/ (current date + hour offset)
  - Selects rows matching that virtual hour from the source Parquet
  - Writes batch to taxi-streaming/YYYY/MM/DD/HH-mm.parquet
  - Updates state.json
"""

import io
import json
import logging
import os
import sys
from datetime import datetime, timedelta

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

RAW_BUCKET = "taxi-raw"
STREAM_BUCKET = "taxi-streaming"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1000"))
STATE_KEY = "state.json"


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


def load_state(s3):
    """Load simulator state from MinIO. Returns dict with current_date and hour_offset."""
    try:
        resp = s3.get_object(Bucket=STREAM_BUCKET, Key=STATE_KEY)
        state = json.loads(resp["Body"].read().decode())
        log.info("Loaded state: %s", state)
        return state
    except s3.exceptions.ClientError:
        return None


def save_state(s3, state):
    body = json.dumps(state, indent=2).encode()
    s3.put_object(Bucket=STREAM_BUCKET, Key=STATE_KEY, Body=body)
    log.info("Saved state: %s", state)


def main():
    s3 = get_s3()
    ensure_bucket(s3, STREAM_BUCKET)

    # Determine simulation date: env override (for "jump to drift") or from state
    jump_date = os.environ.get("SIMULATION_DATE", "")
    state = load_state(s3)

    if jump_date:
        # Jump: reset to new date at hour 0
        current_date = jump_date
        hour_offset = 0
        log.info("Jump to date: %s", current_date)
    elif state:
        current_date = state["current_date"]
        hour_offset = state["hour_offset"]
    else:
        # First run: default to 2024-01-01
        current_date = "2024-01-01"
        hour_offset = 0
        log.info("First run, starting from %s", current_date)

    # Calculate virtual datetime
    base_dt = datetime.strptime(current_date, "%Y-%m-%d")
    virtual_dt = base_dt + timedelta(hours=hour_offset)
    log.info("Virtual datetime: %s (hour_offset=%d)", virtual_dt, hour_offset)

    # Find the source Parquet file for this month
    source_month = virtual_dt.strftime("%Y-%m")
    source_key = f"yellow_tripdata_{source_month}.parquet"

    try:
        s3.head_object(Bucket=RAW_BUCKET, Key=source_key)
    except s3.exceptions.ClientError:
        log.error("Source file not found: s3://%s/%s", RAW_BUCKET, source_key)
        log.error("Run download_data.py with MONTHS=%s first", source_month)
        sys.exit(1)

    log.info("Reading source: s3://%s/%s", RAW_BUCKET, source_key)
    resp = s3.get_object(Bucket=RAW_BUCKET, Key=source_key)
    df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    log.info("Source data: %d rows", len(df))

    # Filter rows for this virtual hour
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    target_date = virtual_dt.date()
    target_hour = virtual_dt.hour

    mask = (
        (df["tpep_pickup_datetime"].dt.date == target_date)
        & (df["tpep_pickup_datetime"].dt.hour == target_hour)
    )
    batch = df[mask].head(BATCH_SIZE)

    if len(batch) == 0:
        log.warning("No data for %s hour %d, advancing to next hour", target_date, target_hour)
    else:
        log.info("Selected %d rows for %s %02d:00", len(batch), target_date, target_hour)

        # Write batch to taxi-streaming/YYYY/MM/DD/HH-00.parquet
        output_key = virtual_dt.strftime("%Y/%m/%d/%H-00.parquet")
        buf = io.BytesIO()
        table = pa.Table.from_pandas(batch)
        pq.write_table(table, buf)
        buf.seek(0)

        s3.put_object(Bucket=STREAM_BUCKET, Key=output_key, Body=buf.getvalue())
        log.info("Written s3://%s/%s (%d rows)", STREAM_BUCKET, output_key, len(batch))

    # Advance state by 1 hour
    next_offset = hour_offset + 1
    # If we've passed midnight into a new day, reset offset
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
