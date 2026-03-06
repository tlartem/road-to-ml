"""Bronze layer assets — raw and streaming data."""

import os

from dagster import AssetExecutionContext, asset

from src.resources import MinIOResource, VictoriaMetricsResource


@asset(
    group_name="bronze",
    compute_kind="python",
    description="Raw NYC TLC yellow taxi trip data, downloaded from TLC website",
)
def bronze_raw(
    context: AssetExecutionContext,
    minio: MinIOResource,
    vm: VictoriaMetricsResource,
):
    minio.setup_env()
    vm.setup_env()

    from src.lib.ml.download_data import run

    months = os.environ.get("MONTHS", "2024-01,2024-02,2024-03")
    total_rows = run(months=months)
    context.log.info("bronze_raw: %d rows loaded", total_rows)
    return None


@asset(
    group_name="bronze",
    compute_kind="python",
    deps=["bronze_raw"],
    description="Simulated streaming batches (hourly), partitioned by date",
)
def bronze_streaming(
    context: AssetExecutionContext,
    minio: MinIOResource,
    vm: VictoriaMetricsResource,
):
    minio.setup_env()
    vm.setup_env()

    from src.lib.ml.simulator import run

    rows = run()
    context.log.info("bronze_streaming: %d rows written", rows)
    return None
