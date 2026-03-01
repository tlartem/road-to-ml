"""Step 1: Validate raw data from MinIO."""

import json
import logging
import os
import sys

import boto3
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def main():
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    log.info("Downloading dataset from MinIO...")
    s3.download_file("datasets", "water_quality/water_potability.csv", "/tmp/data.csv")

    df = pd.read_csv("/tmp/data.csv")
    log.info("Dataset loaded: %d rows, %d columns", len(df), len(df.columns))

    missing = df.isnull().sum()
    missing_total = missing.sum()
    log.info("Missing values total: %d", missing_total)
    for col in missing[missing > 0].index:
        log.warning("  Column '%s': %d missing (%.1f%%)", col, missing[col], 100 * missing[col] / len(df))

    target_dist = df["Potability"].value_counts()
    log.info("Target distribution:")
    log.info("  Not potable (0): %d (%.1f%%)", target_dist[0], 100 * target_dist[0] / len(df))
    log.info("  Potable (1):     %d (%.1f%%)", target_dist[1], 100 * target_dist[1] / len(df))

    assert len(df) > 0, "Dataset is empty"
    assert "Potability" in df.columns, "Target column 'Potability' missing"
    assert len(df.columns) == 10, f"Expected 10 columns, got {len(df.columns)}"

    log.info("Validation passed!")


if __name__ == "__main__":
    main()
