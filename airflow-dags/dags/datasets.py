"""Airflow Dataset definitions for data-aware scheduling.

Datasets enable lineage tracking and trigger-based scheduling:
  simulator writes → BRONZE_STREAMING outlet
  preprocess triggers on BRONZE_STREAMING → writes SILVER_TRIPS outlet
  drift monitor triggers on BRONZE_STREAMING
"""

from airflow.datasets import Dataset

BRONZE_RAW = Dataset("s3://taxi-lake/bronze/raw")
BRONZE_STREAMING = Dataset("s3://taxi-lake/bronze/streaming")
SILVER_TRIPS = Dataset("s3://taxi-lake/silver/trips")
SILVER_DEMAND = Dataset("s3://taxi-lake/silver/demand")
GOLD_ZONE_STATS = Dataset("s3://taxi-lake/gold/zone_stats")
