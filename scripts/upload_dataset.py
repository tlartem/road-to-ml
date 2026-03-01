"""Download Water Quality dataset and upload to MinIO."""

import boto3
import requests
import io

MINIO_ENDPOINT = "http://minio-api.local:30448"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "datasets"
DATASET_URL = "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/water_potability.csv"

s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
)

# Create bucket
existing = [b["Name"] for b in s3.list_buckets()["Buckets"]]
if BUCKET_NAME not in existing:
    s3.create_bucket(Bucket=BUCKET_NAME)
    print(f"Bucket '{BUCKET_NAME}' created")

# Download dataset
print(f"Downloading from {DATASET_URL}...")
response = requests.get(DATASET_URL)
response.raise_for_status()
print(f"Downloaded {len(response.content)} bytes")

# Upload to MinIO
s3.upload_fileobj(
    io.BytesIO(response.content),
    BUCKET_NAME,
    "water_quality/water_potability.csv",
)
print("Uploaded to s3://datasets/water_quality/water_potability.csv")

# Verify
obj = s3.head_object(Bucket=BUCKET_NAME, Key="water_quality/water_potability.csv")
print(f"Verified: {obj['ContentLength']} bytes in MinIO")
