"""Monitoring assets — drift detection and compaction."""

from dagster import AssetExecutionContext, asset

from src.resources import MinIOResource, VictoriaMetricsResource


@asset(
    group_name="monitoring",
    compute_kind="evidently",
    deps=["silver_trips", "bronze_streaming"],
    description="Evidently drift report comparing training vs streaming data",
)
def drift_report(
    context: AssetExecutionContext,
    minio: MinIOResource,
    vm: VictoriaMetricsResource,
):
    minio.setup_env()
    vm.setup_env()

    from src.lib.ml.monitor_drift import run

    metrics, drift_detected = run()
    if metrics:
        context.log.info(
            "drift_report: drift_share=%.2f, detected=%s",
            metrics.get("evidently_drift_share", 0),
            drift_detected,
        )
    return None


@asset(
    group_name="monitoring",
    compute_kind="python",
    deps=["bronze_streaming"],
    description="Compact small files in bronze/streaming Delta table",
)
def compact_streaming(
    context: AssetExecutionContext,
    minio: MinIOResource,
):
    minio.setup_env()

    from src.lib.lake import compact, table_uri

    uri = table_uri("bronze", "streaming")
    result = compact(uri)
    context.log.info("compact_streaming: done")
    return None
