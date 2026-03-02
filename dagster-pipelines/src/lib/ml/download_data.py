"""Download NYC Taxi Parquet files from TLC website → Delta Lake bronze/raw."""

import logging
import os
import urllib.request

import pandas as pd

from src.lib.lake import table_uri, write_delta
from src.lib.quality import validate_and_push

log = logging.getLogger(__name__)

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
BRONZE_RAW = table_uri("bronze", "raw")


def run(months: str = "2024-01,2024-02,2024-03"):
    months_list = months.split(",")
    all_frames = []

    for month in months_list:
        filename = f"yellow_tripdata_{month}.parquet"
        url = f"{BASE_URL}/{filename}"
        local_path = f"/tmp/{filename}"

        log.info("Downloading %s ...", url)
        urllib.request.urlretrieve(url, local_path)
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        log.info("Downloaded %.1f MB", size_mb)

        df = pd.read_parquet(local_path)
        log.info("  %d rows", len(df))
        all_frames.append(df)
        os.remove(local_path)

    combined = pd.concat(all_frames, ignore_index=True)
    log.info("Total: %d rows from %d months", len(combined), len(months_list))

    validate_and_push(combined, "bronze_raw")
    write_delta(combined, BRONZE_RAW, mode="overwrite")

    log.info("All data loaded into %s", BRONZE_RAW)
    return combined
