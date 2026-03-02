"""
NYC Taxi Demand Model Training Pipeline.

DAG: preprocess_demand → train_demand
Manual trigger. Reads/writes Delta Lake.
"""

from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

from datasets import SILVER_DEMAND

ENV_VARS = [
    k8s.V1EnvVar(name="MLFLOW_TRACKING_URI", value="http://mlflow.mlflow.svc.cluster.local:5000"),
    k8s.V1EnvVar(name="MLFLOW_S3_ENDPOINT_URL", value="http://minio.minio.svc.cluster.local:9000"),
    k8s.V1EnvVar(name="AWS_ACCESS_KEY_ID", value="minioadmin"),
    k8s.V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value="minioadmin"),
    k8s.V1EnvVar(name="VM_URL", value="http://victoriametrics.monitoring.svc.cluster.local:8428"),
]

with DAG(
    dag_id="taxi_train_demand",
    description="Train NYC Taxi demand prediction model",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["taxi", "training", "ml"],
) as dag:

    preprocess = KubernetesPodOperator(
        task_id="preprocess_demand",
        name="taxi-preprocess-demand",
        image="nyc-taxi-model:latest",
        image_pull_policy="Never",
        cmds=[".venv/bin/python", "src/preprocess_demand.py"],
        env_vars=ENV_VARS,
        namespace="training",
        service_account_name="workflow-sa",
        get_logs=True,
        outlets=[SILVER_DEMAND],
    )

    train = KubernetesPodOperator(
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

    preprocess >> train
