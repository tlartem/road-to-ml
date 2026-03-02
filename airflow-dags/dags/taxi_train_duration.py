"""
NYC Taxi Duration Model Training Pipeline.

DAG: preprocess → train_duration
Manual trigger. Reads/writes Delta Lake.
"""

from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

from datasets import SILVER_TRIPS

ENV_VARS = [
    k8s.V1EnvVar(name="MLFLOW_TRACKING_URI", value="http://mlflow.mlflow.svc.cluster.local:5000"),
    k8s.V1EnvVar(name="MLFLOW_S3_ENDPOINT_URL", value="http://minio.minio.svc.cluster.local:9000"),
    k8s.V1EnvVar(name="AWS_ACCESS_KEY_ID", value="minioadmin"),
    k8s.V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value="minioadmin"),
    k8s.V1EnvVar(name="VM_URL", value="http://victoriametrics.monitoring.svc.cluster.local:8428"),
]

POD_RESOURCES = k8s.V1ResourceRequirements(
    requests={"memory": "512Mi", "cpu": "250m"},
    limits={"memory": "3Gi"},
)

with DAG(
    dag_id="taxi_train_duration",
    description="Train NYC Taxi trip duration prediction model",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["taxi", "training", "ml"],
) as dag:

    preprocess = KubernetesPodOperator(
        task_id="preprocess",
        name="taxi-preprocess",
        image="nyc-taxi-model:latest",
        image_pull_policy="Never",
        cmds=[".venv/bin/python", "src/preprocess.py"],
        env_vars=ENV_VARS,
        namespace="training",
        service_account_name="workflow-sa",
        get_logs=True,
        container_resources=POD_RESOURCES,
        outlets=[SILVER_TRIPS],
    )

    train = KubernetesPodOperator(
        task_id="train_duration",
        name="taxi-train-duration",
        image="nyc-taxi-model:latest",
        image_pull_policy="Never",
        cmds=[".venv/bin/python", "src/train_duration.py"],
        env_vars=ENV_VARS,
        namespace="training",
        service_account_name="workflow-sa",
        get_logs=True,
        container_resources=POD_RESOURCES,
    )

    preprocess >> train
