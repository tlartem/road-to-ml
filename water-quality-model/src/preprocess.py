"""Step 2: Preprocess data — handle missing values, split into train/test."""

import logging
import os
import sys

import boto3
import pandas as pd
from sklearn.model_selection import train_test_split

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

    log.info("Downloading raw dataset from MinIO...")
    s3.download_file("datasets", "water_quality/water_potability.csv", "/tmp/data.csv")

    df = pd.read_csv("/tmp/data.csv")
    log.info("Raw data shape: %s", df.shape)

    missing_before = df.isnull().sum().sum()
    df = df.fillna(df.median())
    log.info("Filled %d missing values with median", missing_before)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Potability"],
    )
    log.info("Train set: %d rows", len(train_df))
    log.info("Test set:  %d rows", len(test_df))

    for name, data in [("train.csv", train_df), ("test.csv", test_df)]:
        path = f"/tmp/{name}"
        data.to_csv(path, index=False)
        s3.upload_file(path, "datasets", f"water_quality/processed/{name}")
        log.info("Uploaded s3://datasets/water_quality/processed/%s", name)

    log.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
