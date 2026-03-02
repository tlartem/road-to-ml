"""Gold layer assets — aggregated features and Feast online store."""

from dagster import AssetExecutionContext, asset

from src.resources import MinIOResource, VictoriaMetricsResource


@asset(
    group_name="gold",
    compute_kind="python",
    deps=["bronze_raw"],
    description="Zone-level aggregated statistics for Feature Store",
)
def gold_zone_stats(
    context: AssetExecutionContext,
    minio: MinIOResource,
    vm: VictoriaMetricsResource,
):
    minio.setup_env()
    vm.setup_env()

    from src.lib.ml.feature_engineering import run

    zone_stats = run()
    context.log.info("gold_zone_stats: %d zones", len(zone_stats))
    return None


@asset(
    group_name="gold",
    compute_kind="python",
    deps=["gold_zone_stats"],
    description="Feast online store materialized from zone stats",
)
def feast_online(
    context: AssetExecutionContext,
    minio: MinIOResource,
):
    minio.setup_env()

    from src.lib.ml.feast_materialize import run

    run()
    context.log.info("feast_online: materialization complete")
    return None
