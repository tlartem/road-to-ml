"""Train fare prediction model (LightGBM) and register in MLflow.

Uses DuckDB to JOIN silver/trips with gold/zone_stats without loading all data into pandas.
"""

import logging
import os
import time

import duckdb
import lightgbm as lgb
import mlflow
import numpy as np
import requests
from deltalake import DeltaTable
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.lib.lake import storage_options, table_uri

log = logging.getLogger(__name__)

SILVER_TRIPS = table_uri("silver", "trips")
GOLD_ZONE_STATS = table_uri("gold", "zone_stats")
VM_URL = os.environ.get("VM_URL", "http://victoriametrics.monitoring.svc.cluster.local:8428")

FEATURE_COLS = [
    "pickup_zone_id", "dropoff_zone_id", "trip_distance",
    "pickup_hour", "pickup_day_of_week", "pickup_month",
    "passenger_count",
    "pu_zone_avg_fare", "pu_zone_avg_duration_min", "pu_zone_avg_distance",
    "do_zone_avg_fare", "do_zone_avg_duration_min", "do_zone_avg_distance",
]
TARGET_COL = "total_amount"

MAX_SAMPLE_ROWS = int(os.environ.get("MAX_SAMPLE_ROWS", "1000000"))


def _load_enriched(split_value, max_rows=None):
    """Load trips enriched with zone stats via DuckDB, filtered by split.

    Uses USING SAMPLE for random sampling so all time periods
    are represented equally, including the most recent data.
    """
    opts = storage_options()
    trips_ds = DeltaTable(SILVER_TRIPS, storage_options=opts).to_pyarrow_dataset()
    zones_ds = DeltaTable(GOLD_ZONE_STATS, storage_options=opts).to_pyarrow_dataset()

    con = duckdb.connect()
    con.register("trips", trips_ds)
    con.register("zones", zones_ds)

    sample_clause = f"USING SAMPLE {max_rows} ROWS" if max_rows else ""

    df = con.execute(f"""
        SELECT
            t.pickup_zone_id, t.dropoff_zone_id, t.trip_distance,
            t.pickup_hour, t.pickup_day_of_week, t.pickup_month,
            t.passenger_count, t.total_amount,
            COALESCE(pu.zone_avg_fare, 0) AS pu_zone_avg_fare,
            COALESCE(pu.zone_avg_duration_min, 0) AS pu_zone_avg_duration_min,
            COALESCE(pu.zone_avg_distance, 0) AS pu_zone_avg_distance,
            COALESCE(dz.zone_avg_fare, 0) AS do_zone_avg_fare,
            COALESCE(dz.zone_avg_duration_min, 0) AS do_zone_avg_duration_min,
            COALESCE(dz.zone_avg_distance, 0) AS do_zone_avg_distance
        FROM trips t
        LEFT JOIN zones pu ON t.pickup_zone_id = pu.zone_id
        LEFT JOIN zones dz ON t.dropoff_zone_id = dz.zone_id
        WHERE t.split = '{split_value}'
        {sample_clause}
    """).fetchdf()

    con.close()
    return df


def _push_model_metrics(model_name, metrics, train_rows):
    """Push model performance metrics to VictoriaMetrics."""
    lines = [
        f'model_mae{{model="{model_name}"}} {metrics["mae"]:.4f}',
        f'model_rmse{{model="{model_name}"}} {metrics["rmse"]:.4f}',
        f'model_r2{{model="{model_name}"}} {metrics["r2"]:.4f}',
        f'model_train_rows{{model="{model_name}"}} {train_rows}',
        f'model_retrain_timestamp{{model="{model_name}"}} {time.time():.0f}',
    ]
    try:
        requests.post(
            f"{VM_URL}/api/v1/import/prometheus",
            data="\n".join(lines) + "\n",
            headers={"Content-Type": "text/plain"}, timeout=10,
        )
        log.info("Model metrics pushed to VictoriaMetrics for %s", model_name)
    except Exception as e:
        log.warning("Failed to push model metrics: %s", e)


def promote_if_better(new_mae, model_info):
    client = mlflow.tracking.MlflowClient()
    model_name = "taxi-fare"

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


def run():
    log.info("Loading train split via DuckDB (sample %d rows)...", MAX_SAMPLE_ROWS)
    train_df = _load_enriched("train", max_rows=MAX_SAMPLE_ROWS)
    log.info("Train: %d rows", len(train_df))

    log.info("Loading test split via DuckDB (sample %d rows)...", MAX_SAMPLE_ROWS // 4)
    test_df = _load_enriched("test", max_rows=MAX_SAMPLE_ROWS // 4)
    log.info("Test: %d rows", len(test_df))

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("nyc-taxi-fare")

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
        _push_model_metrics("taxi-fare", metrics, len(train_df))
        for name, value in metrics.items():
            log.info("  %-12s %.4f", name, value)

        result = mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name="taxi-fare",
        )
        promote_if_better(metrics["mae"], result)

    return metrics
