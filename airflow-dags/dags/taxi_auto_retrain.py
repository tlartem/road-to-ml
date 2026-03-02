"""
NYC Taxi Auto-Retrain.

Triggered by drift monitor when drift exceeds threshold.
Retrains all 3 models on combined data (bronze/raw + bronze/streaming).
"""

from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

from datasets import SILVER_TRIPS, SILVER_DEMAND

ENV_VARS = [
    k8s.V1EnvVar(name="MLFLOW_S3_ENDPOINT_URL", value="http://minio.minio.svc.cluster.local:9000"),
    k8s.V1EnvVar(name="AWS_ACCESS_KEY_ID", value="minioadmin"),
    k8s.V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value="minioadmin"),
    k8s.V1EnvVar(name="MLFLOW_TRACKING_URI", value="http://mlflow.mlflow.svc.cluster.local:5000"),
    k8s.V1EnvVar(name="DATA_SOURCE", value="combined"),
]

with DAG(
    dag_id="taxi_auto_retrain",
    description="Auto-retrain all models on combined data when drift detected",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["taxi", "retrain", "auto"],
) as dag:

    preprocess = KubernetesPodOperator(
        task_id="preprocess_combined",
        name="taxi-preprocess-combined",
        image="nyc-taxi-model:latest",
        image_pull_policy="Never",
        cmds=[".venv/bin/python", "src/preprocess.py"],
        env_vars=ENV_VARS,
        namespace="training",
        service_account_name="workflow-sa",
        get_logs=True,
        outlets=[SILVER_TRIPS],
    )

    preprocess_demand = KubernetesPodOperator(
        task_id="preprocess_demand_combined",
        name="taxi-preprocess-demand-combined",
        image="nyc-taxi-model:latest",
        image_pull_policy="Never",
        cmds=[".venv/bin/python", "src/preprocess_demand.py"],
        env_vars=ENV_VARS,
        namespace="training",
        service_account_name="workflow-sa",
        get_logs=True,
        outlets=[SILVER_DEMAND],
    )

    train_duration = KubernetesPodOperator(
        task_id="train_duration",
        name="taxi-train-duration",
        image="nyc-taxi-model:latest",
        image_pull_policy="Never",
        cmds=[".venv/bin/python", "src/train_duration.py"],
        env_vars=ENV_VARS,
        namespace="training",
        service_account_name="workflow-sa",
        get_logs=True,
    )

    train_fare = KubernetesPodOperator(
        task_id="train_fare",
        name="taxi-train-fare",
        image="nyc-taxi-model:latest",
        image_pull_policy="Never",
        cmds=[".venv/bin/python", "src/train_fare.py"],
        env_vars=ENV_VARS,
        namespace="training",
        service_account_name="workflow-sa",
        get_logs=True,
    )

    train_demand = KubernetesPodOperator(
        task_id="train_demand",
        name="taxi-train-demand",
        image="nyc-taxi-model:latest",
        image_pull_policy="Never",
        cmds=[".venv/bin/python", "src/train_demand.py"],
        env_vars=ENV_VARS,
        namespace="training",
        service_account_name="workflow-sa",
        get_logs=True,
    )

    preprocess >> [train_duration, train_fare]
    preprocess_demand >> train_demand
