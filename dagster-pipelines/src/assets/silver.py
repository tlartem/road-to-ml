"""Silver layer assets — cleaned and preprocessed data."""

import os

from dagster import AssetExecutionContext, asset

from src.resources import MinIOResource, VictoriaMetricsResource


@asset(
    group_name="silver",
    compute_kind="python",
    deps=["bronze_raw"],
    description="Cleaned trip data with engineered features, outliers removed",
)
def silver_trips(
    context: AssetExecutionContext,
    minio: MinIOResource,
    vm: VictoriaMetricsResource,
):
    minio.setup_env()
    vm.setup_env()

    from src.lib.ml.preprocess import run

    data_source = os.environ.get("DATA_SOURCE", "raw")
    df = run(data_source=data_source)
    context.log.info("silver_trips: %d rows", len(df))
    return None


@asset(
    group_name="silver",
    compute_kind="python",
    deps=["bronze_raw"],
    description="Aggregated trip counts per zone per hour",
)
def silver_demand(
    context: AssetExecutionContext,
    minio: MinIOResource,
    vm: VictoriaMetricsResource,
):
    minio.setup_env()
    vm.setup_env()

    from src.lib.ml.preprocess_demand import run

    data_source = os.environ.get("DATA_SOURCE", "raw")
    df = run(data_source=data_source)
    context.log.info("silver_demand: %d rows", len(df))
    return None
