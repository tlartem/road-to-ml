"""Asset checks — data quality validation using Pandera."""

from dagster import AssetCheckExecutionContext, AssetCheckResult, asset_check

from src.resources import MinIOResource, VictoriaMetricsResource


@asset_check(asset="bronze_raw", description="Validate bronze_raw with Pandera checks")
def check_bronze_raw(context: AssetCheckExecutionContext, minio: MinIOResource, vm: VictoriaMetricsResource):
    minio.setup_env()
    vm.setup_env()
    from src.lib.lake import read_delta, table_uri
    from src.lib.quality import validate, push_quality_metrics

    df = read_delta(table_uri("bronze", "raw"))
    result = validate(df, "bronze_raw")
    push_quality_metrics(result)
    return AssetCheckResult(
        passed=result.success,
        metadata={"rows": result.rows, "passed": result.passed, "failed": result.failed},
    )


@asset_check(asset="bronze_streaming", description="Validate bronze_streaming with Pandera checks")
def check_bronze_streaming(context: AssetCheckExecutionContext, minio: MinIOResource, vm: VictoriaMetricsResource):
    minio.setup_env()
    vm.setup_env()
    from src.lib.lake import read_delta, table_uri
    from src.lib.quality import validate, push_quality_metrics

    df = read_delta(table_uri("bronze", "streaming"))
    result = validate(df, "bronze_streaming")
    push_quality_metrics(result)
    return AssetCheckResult(
        passed=result.success,
        metadata={"rows": result.rows, "passed": result.passed, "failed": result.failed},
    )


@asset_check(asset="silver_trips", description="Validate silver_trips with Pandera checks")
def check_silver_trips(context: AssetCheckExecutionContext, minio: MinIOResource, vm: VictoriaMetricsResource):
    minio.setup_env()
    vm.setup_env()
    from src.lib.lake import read_delta, table_uri
    from src.lib.quality import validate, push_quality_metrics

    df = read_delta(table_uri("silver", "trips"))
    result = validate(df, "silver_trips")
    push_quality_metrics(result)
    return AssetCheckResult(
        passed=result.success,
        metadata={"rows": result.rows, "passed": result.passed, "failed": result.failed},
    )


@asset_check(asset="silver_demand", description="Validate silver_demand with Pandera checks")
def check_silver_demand(context: AssetCheckExecutionContext, minio: MinIOResource, vm: VictoriaMetricsResource):
    minio.setup_env()
    vm.setup_env()
    from src.lib.lake import read_delta, table_uri
    from src.lib.quality import validate, push_quality_metrics

    df = read_delta(table_uri("silver", "demand"))
    result = validate(df, "silver_demand")
    push_quality_metrics(result)
    return AssetCheckResult(
        passed=result.success,
        metadata={"rows": result.rows, "passed": result.passed, "failed": result.failed},
    )


@asset_check(asset="gold_zone_stats", description="Validate gold_zone_stats with Pandera checks")
def check_gold_zone_stats(context: AssetCheckExecutionContext, minio: MinIOResource, vm: VictoriaMetricsResource):
    minio.setup_env()
    vm.setup_env()
    from src.lib.lake import read_delta, table_uri
    from src.lib.quality import validate, push_quality_metrics

    df = read_delta(table_uri("gold", "zone_stats"))
    result = validate(df, "gold_zone_stats")
    push_quality_metrics(result)
    return AssetCheckResult(
        passed=result.success,
        metadata={"rows": result.rows, "passed": result.passed, "failed": result.failed},
    )
