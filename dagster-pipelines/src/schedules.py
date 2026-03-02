"""Schedules and sensors for automated pipeline execution."""

from dagster import (
    AssetSelection,
    RunRequest,
    ScheduleDefinition,
    SensorEvaluationContext,
    define_asset_job,
    sensor,
)

from src.resources import MinIOResource, VictoriaMetricsResource

# -- Jobs (used by schedules/sensors) --

simulator_job = define_asset_job(
    name="simulator_job",
    selection=AssetSelection.assets("bronze_streaming"),
    description="Run streaming simulator (1 batch)",
)

compact_job = define_asset_job(
    name="compact_job",
    selection=AssetSelection.assets("compact_streaming"),
    description="Compact bronze/streaming Delta table",
)

drift_job = define_asset_job(
    name="drift_job",
    selection=AssetSelection.assets("drift_report"),
    description="Run drift monitoring",
)

retrain_job = define_asset_job(
    name="retrain_job",
    selection=AssetSelection.assets(
        "silver_trips", "silver_demand",
        "model_duration", "model_fare", "model_demand",
    ),
    description="Retrain all models on combined data",
)

# -- Schedules --

simulator_schedule = ScheduleDefinition(
    job=simulator_job,
    cron_schedule="*/1 * * * *",
)

compact_schedule = ScheduleDefinition(
    job=compact_job,
    cron_schedule="0 * * * *",
)

drift_schedule = ScheduleDefinition(
    job=drift_job,
    cron_schedule="0 */6 * * *",
)


# -- Sensors --

@sensor(job=retrain_job, minimum_interval_seconds=300)
def drift_retrain_sensor(context: SensorEvaluationContext, minio: MinIOResource, vm: VictoriaMetricsResource):
    """Check VictoriaMetrics for drift metrics, trigger retrain if drift detected."""
    import os
    import requests

    vm.setup_env()
    minio.setup_env()
    vm_url = os.environ.get("VM_URL", "http://victoriametrics.monitoring.svc.cluster.local:8428")

    try:
        resp = requests.get(
            f"{vm_url}/api/v1/query",
            params={"query": "evidently_drift_share"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("data", {}).get("result", [])
        if not results:
            context.log.info("No drift metrics found yet")
            return

        drift_share = float(results[0].get("value", [0, "0"])[1])
        threshold = float(os.environ.get("RETRAIN_THRESHOLD", "0.5"))
        context.log.info("Current drift_share=%.2f, threshold=%.2f", drift_share, threshold)

        if drift_share >= threshold:
            cursor = context.cursor or "0"
            new_cursor = str(int(cursor) + 1)
            context.update_cursor(new_cursor)

            if cursor != new_cursor:
                context.log.info("Drift detected! Triggering retrain (run #%s)", new_cursor)
                yield RunRequest(
                    run_key=f"drift_retrain_{new_cursor}",
                )
    except Exception as e:
        context.log.warning("Failed to check drift metrics: %s", e)
