"""Train trip duration prediction model (LightGBM) and register in MLflow.

Reads from Delta Lake silver/trips + gold/zone_stats.
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

SILVER_TRIPS = table_uri("silver", "trips")
GOLD_ZONE_STATS = table_uri("gold", "zone_stats")

TRIP_FEATURES = [
    "pickup_zone_id", "dropoff_zone_id", "trip_distance",
    "pickup_hour", "pickup_day_of_week", "pickup_month",
    "passenger_count",
]

ZONE_FEATURES = [
    "pu_zone_avg_fare", "pu_zone_avg_duration_min", "pu_zone_avg_distance",
    "do_zone_avg_fare", "do_zone_avg_duration_min", "do_zone_avg_distance",
]

FEATURE_COLS = TRIP_FEATURES + ZONE_FEATURES
TARGET_COL = "duration_min"


def enrich_with_zone_stats(df, zone_stats):
    """Join zone-level features for pickup and dropoff zones."""
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


def main():
    # Read from Delta Lake
    log.info("Loading silver/trips from Delta Lake...")
    all_data = read_delta(SILVER_TRIPS)
    train_df = all_data[all_data["split"] == "train"].drop("split", axis=1)
    test_df = all_data[all_data["split"] == "test"].drop("split", axis=1)
    log.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    log.info("Loading gold/zone_stats from Delta Lake...")
    zone_stats = read_delta(GOLD_ZONE_STATS)
    log.info("Zone stats: %d zones", len(zone_stats))

    train_df = enrich_with_zone_stats(train_df, zone_stats)
    test_df = enrich_with_zone_stats(test_df, zone_stats)

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
            model, artifact_path="model", registered_model_name="taxi-duration",
        )
        promote_if_better(metrics["mae"], result)


def promote_if_better(new_mae, model_info):
    """Set alias 'champion' on new model version if MAE improved."""
    client = mlflow.tracking.MlflowClient()
    model_name = "taxi-duration"

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
            log.info("New model better (MAE %.4f < %.4f), promoting", new_mae, champion_mae)
            client.set_registered_model_alias(model_name, "champion", new_version)
        else:
            log.info("Champion still better (MAE %.4f >= %.4f)", new_mae, champion_mae)
    except Exception:
        log.info("No champion yet, setting v%s as first champion", new_version)
        client.set_registered_model_alias(model_name, "champion", new_version)


if __name__ == "__main__":
    main()
