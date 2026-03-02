"""Train demand prediction model (LightGBM) and register in MLflow.

Reads from Delta Lake silver/demand + gold/zone_stats. Target: trip_count.
"""

import logging
import os
import sys

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from lake import read_delta, table_uri

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

SILVER_DEMAND = table_uri("silver", "demand")
GOLD_ZONE_STATS = table_uri("gold", "zone_stats")

BASE_FEATURES = ["zone_id", "pickup_hour", "day_of_week", "month"]
ZONE_FEATURES = [
    "zone_avg_fare", "zone_avg_duration_min", "zone_avg_distance", "zone_trip_count",
]
FEATURE_COLS = BASE_FEATURES + ZONE_FEATURES
TARGET_COL = "trip_count"


def enrich_with_zone_stats(df, zone_stats):
    zs = zone_stats[["zone_id", "zone_avg_fare", "zone_avg_duration_min",
                      "zone_avg_distance", "zone_trip_count"]]
    df = df.merge(zs, on="zone_id", how="left")
    for col in ZONE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def promote_if_better(new_mae, model_info):
    client = mlflow.tracking.MlflowClient()
    model_name = "taxi-demand"

    new_version = None
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.run_id == model_info.run_id:
            new_version = mv.version
            break

    if new_version is None:
        return

    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
        champion_run = client.get_run(champion.run_id)
        champion_mae = champion_run.data.metrics.get("mae")

        if new_mae < champion_mae:
            log.info("New model better (MAE %.4f < %.4f), promoting", new_mae, champion_mae)
            client.set_registered_model_alias(model_name, "champion", new_version)
        else:
            log.info("Champion still better (MAE %.4f >= %.4f)", new_mae, champion_mae)
    except Exception:
        log.info("No champion yet, setting v%s as first champion", new_version)
        client.set_registered_model_alias(model_name, "champion", new_version)


def main():
    log.info("Loading silver/demand from Delta Lake...")
    all_data = read_delta(SILVER_DEMAND)
    train_df = all_data[all_data["split"] == "train"].drop("split", axis=1)
    test_df = all_data[all_data["split"] == "test"].drop("split", axis=1)
    log.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    log.info("Loading gold/zone_stats from Delta Lake...")
    zone_stats = read_delta(GOLD_ZONE_STATS)
    train_df = enrich_with_zone_stats(train_df, zone_stats)
    test_df = enrich_with_zone_stats(test_df, zone_stats)

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("nyc-taxi-demand")

    params = {
        "objective": "regression", "metric": "mae",
        "num_leaves": 31, "learning_rate": 0.1, "n_estimators": 200,
        "random_state": 42, "verbose": -1,
    }

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("data_source", "delta_lake")

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": r2_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)
        for name, value in metrics.items():
            log.info("  %-12s %.4f", name, value)

        result = mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name="taxi-demand",
        )
        promote_if_better(metrics["mae"], result)


if __name__ == "__main__":
    main()
