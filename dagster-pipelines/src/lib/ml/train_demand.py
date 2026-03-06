"""Train demand prediction model (LightGBM) and register in MLflow.

Uses DuckDB to JOIN silver/demand with gold/zone_stats without loading all data into pandas.
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

SILVER_DEMAND = table_uri("silver", "demand")
GOLD_ZONE_STATS = table_uri("gold", "zone_stats")
VM_URL = os.environ.get("VM_URL", "http://victoriametrics.monitoring.svc.cluster.local:8428")

FEATURE_COLS = [
    "zone_id", "pickup_hour", "day_of_week", "month",
    "zone_avg_fare", "zone_avg_duration_min", "zone_avg_distance", "zone_trip_count",
]
TARGET_COL = "trip_count"


def _load_enriched(split_value):
    """Load demand enriched with zone stats via DuckDB, filtered by split."""
    opts = storage_options()
    demand_ds = DeltaTable(SILVER_DEMAND, storage_options=opts).to_pyarrow_dataset()
    zones_ds = DeltaTable(GOLD_ZONE_STATS, storage_options=opts).to_pyarrow_dataset()

    con = duckdb.connect()
    con.register("demand", demand_ds)
    con.register("zones", zones_ds)

    df = con.execute(f"""
        SELECT
            d.zone_id, d.pickup_hour, d.day_of_week, d.month, d.trip_count,
            COALESCE(z.zone_avg_fare, 0) AS zone_avg_fare,
            COALESCE(z.zone_avg_duration_min, 0) AS zone_avg_duration_min,
            COALESCE(z.zone_avg_distance, 0) AS zone_avg_distance,
            COALESCE(z.zone_trip_count, 0) AS zone_trip_count
        FROM demand d
        LEFT JOIN zones z ON d.zone_id = z.zone_id
        WHERE d.split = '{split_value}'
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


def run():
    log.info("Loading train split via DuckDB (demand + zone_stats JOIN)...")
    train_df = _load_enriched("train")
    log.info("Train: %d rows", len(train_df))

    log.info("Loading test split via DuckDB...")
    test_df = _load_enriched("test")
    log.info("Test: %d rows", len(test_df))

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
        _push_model_metrics("taxi-demand", metrics, len(train_df))
        for name, value in metrics.items():
            log.info("  %-12s %.4f", name, value)

        result = mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name="taxi-demand",
        )
        promote_if_better(metrics["mae"], result)

    return metrics
