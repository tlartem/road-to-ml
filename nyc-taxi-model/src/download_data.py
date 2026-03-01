"""Download NYC Taxi Parquet files from TLC website → MinIO taxi-raw/."""

import logging
import os
import sys
import urllib.request

import boto3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
MONTHS = os.environ.get("MONTHS", "2024-01,2024-02,2024-03").split(",")
BUCKET = "taxi-raw"


def main():
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    # Ensure bucket exists
    try:
        s3.head_bucket(Bucket=BUCKET)
    except s3.exceptions.ClientError:
        s3.create_bucket(Bucket=BUCKET)
        log.info("Created bucket: %s", BUCKET)

    for month in MONTHS:
        filename = f"yellow_tripdata_{month}.parquet"
        url = f"{BASE_URL}/{filename}"
        local_path = f"/tmp/{filename}"
        s3_key = filename

        # Check if already uploaded
        try:
            s3.head_object(Bucket=BUCKET, Key=s3_key)
            log.info("Already exists in MinIO: s3://%s/%s, skipping", BUCKET, s3_key)
            continue
        except s3.exceptions.ClientError:
            pass

        log.info("Downloading %s ...", url)
        urllib.request.urlretrieve(url, local_path)
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        log.info("Downloaded %.1f MB", size_mb)

        log.info("Uploading to s3://%s/%s ...", BUCKET, s3_key)
        s3.upload_file(local_path, BUCKET, s3_key)
        log.info("Uploaded successfully")

        os.remove(local_path)

    log.info("All files downloaded and uploaded to MinIO!")


if __name__ == "__main__":
    main()
