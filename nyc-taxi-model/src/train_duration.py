"""Train trip duration prediction model (LightGBM) and register in MLflow."""

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
FEATURE_COLS = [
    "pickup_zone_id",
    "dropoff_zone_id",
    "trip_distance",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "passenger_count",
]
TARGET_COL = "duration_min"


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


def main():
    s3 = get_s3()

    log.info("Loading processed data...")
    train_df = load_data(s3, "duration/train.parquet")
    test_df = load_data(s3, "duration/test.parquet")
    log.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("nyc-taxi-duration")

    params = {
        "objective": "regression",
        "metric": "mae",
        "num_leaves": int(os.environ.get("NUM_LEAVES", "31")),
        "learning_rate": float(os.environ.get("LEARNING_RATE", "0.1")),
        "n_estimators": int(os.environ.get("N_ESTIMATORS", "200")),
        "random_state": 42,
        "verbose": -1,
    }

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("features", FEATURE_COLS)

        log.info("Training LightGBM with params: %s", params)
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        log.info("Training complete")

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

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="taxi-duration",
        )
        log.info("Model registered: taxi-duration")


if __name__ == "__main__":
    main()
