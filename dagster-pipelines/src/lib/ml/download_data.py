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
    total_rows = 0

    for i, month in enumerate(months_list):
        filename = f"yellow_tripdata_{month}.parquet"
        url = f"{BASE_URL}/{filename}"
        local_path = f"/tmp/{filename}"

        log.info("Downloading %s ...", url)
        urllib.request.urlretrieve(url, local_path)
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        log.info("Downloaded %.1f MB", size_mb)

        df = pd.read_parquet(local_path)
        os.remove(local_path)
        log.info("  %s: %d rows", month, len(df))

        validate_and_push(df, "bronze_raw", fail_on_error=False)

        # First month: overwrite, rest: append
        mode = "overwrite" if i == 0 else "append"
        write_delta(df, BRONZE_RAW, mode=mode)
        total_rows += len(df)

        del df  # free memory before next month

    log.info("Total: %d rows from %d months → %s", total_rows, len(months_list), BRONZE_RAW)
    return total_rows
