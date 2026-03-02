"""Training assets — ML model training and registration."""

from dagster import AssetExecutionContext, asset

from src.resources import MLflowResource, MinIOResource


@asset(
    group_name="training",
    compute_kind="lightgbm",
    deps=["silver_trips", "gold_zone_stats"],
    description="LightGBM model for trip duration prediction",
)
def model_duration(
    context: AssetExecutionContext,
    minio: MinIOResource,
    mlflow_res: MLflowResource,
):
    minio.setup_env()
    mlflow_res.setup_env()

    from src.lib.ml.train_duration import run

    metrics = run()
    context.log.info("model_duration: MAE=%.4f, R2=%.4f", metrics["mae"], metrics["r2"])
    return None


@asset(
    group_name="training",
    compute_kind="lightgbm",
    deps=["silver_trips", "gold_zone_stats"],
    description="LightGBM model for fare prediction",
)
def model_fare(
    context: AssetExecutionContext,
    minio: MinIOResource,
    mlflow_res: MLflowResource,
):
    minio.setup_env()
    mlflow_res.setup_env()

    from src.lib.ml.train_fare import run

    metrics = run()
    context.log.info("model_fare: MAE=%.4f, R2=%.4f", metrics["mae"], metrics["r2"])
    return None


@asset(
    group_name="training",
    compute_kind="lightgbm",
    deps=["silver_demand", "gold_zone_stats"],
    description="LightGBM model for demand prediction",
)
def model_demand(
    context: AssetExecutionContext,
    minio: MinIOResource,
    mlflow_res: MLflowResource,
):
    minio.setup_env()
    mlflow_res.setup_env()

    from src.lib.ml.train_demand import run

    metrics = run()
    context.log.info("model_demand: MAE=%.4f, R2=%.4f", metrics["mae"], metrics["r2"])
    return None
