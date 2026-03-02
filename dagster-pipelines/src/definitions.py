"""Dagster Definitions — entry point for the user code server."""

from dagster import Definitions, load_assets_from_modules

from src.assets import bronze, gold, monitoring, silver, training
from src.checks import (
    check_bronze_raw,
    check_bronze_streaming,
    check_gold_zone_stats,
    check_silver_demand,
    check_silver_trips,
)
from src.jobs import full_pipeline_job, training_pipeline_job
from src.resources import MLflowResource, MinIOResource, VictoriaMetricsResource
from src.schedules import (
    compact_job,
    compact_schedule,
    drift_job,
    drift_retrain_sensor,
    drift_schedule,
    retrain_job,
    simulator_job,
    simulator_schedule,
)

all_assets = load_assets_from_modules([bronze, silver, gold, training, monitoring])

all_checks = [
    check_bronze_raw,
    check_bronze_streaming,
    check_silver_trips,
    check_silver_demand,
    check_gold_zone_stats,
]

all_jobs = [
    simulator_job,
    compact_job,
    drift_job,
    retrain_job,
    full_pipeline_job,
    training_pipeline_job,
]

all_schedules = [
    simulator_schedule,
    compact_schedule,
    drift_schedule,
]

all_sensors = [
    drift_retrain_sensor,
]

defs = Definitions(
    assets=all_assets,
    asset_checks=all_checks,
    jobs=all_jobs,
    schedules=all_schedules,
    sensors=all_sensors,
    resources={
        "minio": MinIOResource(),
        "mlflow_res": MLflowResource(),
        "vm": VictoriaMetricsResource(),
    },
)
