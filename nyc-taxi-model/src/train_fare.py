"""Train fare prediction model (LightGBM) and register in MLflow.

Uses same trip features + zone stats as duration model. Target: total_amount.
"""

import io
import logging
import os
import sys

import boto3
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

PROCESSED_BUCKET = "taxi-processed"

TRIP_FEATURES = [
    "pickup_zone_id",
    "dropoff_zone_id",
    "trip_distance",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "passenger_count",
]

ZONE_FEATURES = [
    "pu_zone_avg_fare",
    "pu_zone_avg_duration_min",
    "pu_zone_avg_distance",
    "do_zone_avg_fare",
    "do_zone_avg_duration_min",
    "do_zone_avg_distance",
]

FEATURE_COLS = TRIP_FEATURES + ZONE_FEATURES
TARGET_COL = "total_amount"


def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def load_data(s3, key):
    resp = s3.get_object(Bucket=PROCESSED_BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(resp["Body"].read()))


def enrich_with_zone_stats(df, zone_stats):
    zone_cols = ["zone_id", "zone_avg_fare", "zone_avg_duration_min", "zone_avg_distance"]
    zs = zone_stats[zone_cols]

    df = df.merge(zs, left_on="pickup_zone_id", right_on="zone_id", how="left").drop("zone_id", axis=1)
    df = df.rename(columns={
        "zone_avg_fare": "pu_zone_avg_fare",
        "zone_avg_duration_min": "pu_zone_avg_duration_min",
        "zone_avg_distance": "pu_zone_avg_distance",
    })

    df = df.merge(zs, left_on="dropoff_zone_id", right_on="zone_id", how="left").drop("zone_id", axis=1)
    df = df.rename(columns={
        "zone_avg_fare": "do_zone_avg_fare",
        "zone_avg_duration_min": "do_zone_avg_duration_min",
        "zone_avg_distance": "do_zone_avg_distance",
    })

    for col in ZONE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def promote_if_better(new_mae, model_info):
    client = mlflow.tracking.MlflowClient()
    model_name = "taxi-fare"

    new_version = None
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.run_id == model_info.run_id:
            new_version = mv.version
            break

    if new_version is None:
        log.warning("Could not find new model version, skipping promotion")
        return

    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
        champion_run = client.get_run(champion.run_id)
        champion_mae = champion_run.data.metrics.get("mae")
        log.info("Current champion: v%s (MAE=%.4f)", champion.version, champion_mae)

        if new_mae < champion_mae:
            log.info("New model is better (MAE %.4f < %.4f), promoting", new_mae, champion_mae)
            client.set_registered_model_alias(model_name, "champion", new_version)
        else:
            log.info("Current champion still better, keeping v%s", champion.version)
    except Exception:
        log.info("No champion yet, setting v%s as first champion", new_version)
        client.set_registered_model_alias(model_name, "champion", new_version)


def main():
    s3 = get_s3()

    log.info("Loading processed data...")
    train_df = load_data(s3, "trips/train.parquet")
    test_df = load_data(s3, "trips/test.parquet")
    log.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    log.info("Loading zone stats from Feature Store...")
    zone_stats = load_data(s3, "feast/zone_stats.parquet")
    train_df = enrich_with_zone_stats(train_df, zone_stats)
    test_df = enrich_with_zone_stats(test_df, zone_stats)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("nyc-taxi-fare")

    params = {
        "objective": "regression",
        "metric": "mae",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "random_state": 42,
        "verbose": -1,
    }

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("features", FEATURE_COLS)
        mlflow.log_param("feature_store", "zone_stats")

        log.info("Training LightGBM for fare prediction...")
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": r2_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)

        log.info("Metrics:")
        for name, value in metrics.items():
            log.info("  %-12s %.4f", name, value)

        result = mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name="taxi-fare",
        )
        log.info("Model registered: taxi-fare")
        promote_if_better(metrics["mae"], result)


if __name__ == "__main__":
    main()
