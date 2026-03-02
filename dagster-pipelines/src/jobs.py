"""Dagster job definitions for manual triggering."""

from dagster import AssetSelection, define_asset_job

full_pipeline_job = define_asset_job(
    name="full_pipeline_job",
    selection=AssetSelection.all(),
    description="Materialize all assets (full pipeline)",
)

training_pipeline_job = define_asset_job(
    name="training_pipeline_job",
    selection=AssetSelection.assets(
        "silver_trips", "silver_demand", "gold_zone_stats",
        "model_duration", "model_fare", "model_demand",
    ),
    description="Preprocess + train all models",
)
