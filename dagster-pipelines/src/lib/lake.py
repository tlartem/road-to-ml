"""Delta Lake utilities for NYC Taxi data lake.

All tables stored in s3://taxi-lake/ with Medallion architecture:
  bronze/ — raw ingested data
  silver/ — cleaned, typed, validated
  gold/   — aggregated features for ML
"""

import logging
import os

import boto3
import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

log = logging.getLogger(__name__)

LAKE_BUCKET = "taxi-lake"


def storage_options():
    """S3/MinIO credentials for delta-rs."""
    return {
        "AWS_ENDPOINT_URL": os.environ["MLFLOW_S3_ENDPOINT_URL"],
        "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
        "AWS_REGION": "us-east-1",
        "AWS_ALLOW_HTTP": "true",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }


def table_uri(layer, name):
    """Build S3 URI: s3://taxi-lake/{layer}/{name}."""
    return f"s3://{LAKE_BUCKET}/{layer}/{name}"


def ensure_bucket():
    """Create taxi-lake bucket if it doesn't exist."""
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    try:
        s3.head_bucket(Bucket=LAKE_BUCKET)
    except s3.exceptions.ClientError:
        s3.create_bucket(Bucket=LAKE_BUCKET)
        log.info("Created bucket: %s", LAKE_BUCKET)


def read_delta(uri):
    """Read Delta table → pandas DataFrame."""
    dt = DeltaTable(uri, storage_options=storage_options())
    df = dt.to_pandas()
    log.info("Read %d rows from %s (version %d)", len(df), uri, dt.version())
    return df


def read_delta_version(uri, version):
    """Read specific version of Delta table (time travel)."""
    dt = DeltaTable(uri, storage_options=storage_options(), version=version)
    df = dt.to_pandas()
    log.info("Read %d rows from %s (version %d)", len(df), uri, version)
    return df


def write_delta(df, uri, mode="append", partition_by=None, schema=None):
    """Write DataFrame to Delta table."""
    ensure_bucket()
    table = pa.Table.from_pandas(df, preserve_index=False)
    if schema:
        table = table.cast(schema)

    write_deltalake(
        uri,
        table,
        mode=mode,
        partition_by=partition_by,
        storage_options=storage_options(),
    )
    log.info("Wrote %d rows to %s (mode=%s)", len(df), uri, mode)


def compact(uri):
    """OPTIMIZE: merge small files in Delta table."""
    dt = DeltaTable(uri, storage_options=storage_options())
    before = len(dt.files())
    result = dt.optimize.compact()
    after = len(dt.files())
    log.info("Compacted %s: %d → %d files", uri, before, after)
    return result


def history(uri):
    """Get Delta table history (list of versions)."""
    dt = DeltaTable(uri, storage_options=storage_options())
    return dt.history()


def table_exists(uri):
    """Check if a Delta table exists at the given URI."""
    try:
        DeltaTable(uri, storage_options=storage_options())
        return True
    except Exception:
        return False
