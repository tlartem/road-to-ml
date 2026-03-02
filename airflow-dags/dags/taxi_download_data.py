"""
NYC Taxi Raw Data Download.

DAG: download_data
Manual trigger. Downloads TLC Parquet → Delta Lake bronze/raw.
"""

from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

from datasets import BRONZE_RAW

ENV_VARS = [
    k8s.V1EnvVar(name="MLFLOW_S3_ENDPOINT_URL", value="http://minio.minio.svc.cluster.local:9000"),
    k8s.V1EnvVar(name="AWS_ACCESS_KEY_ID", value="minioadmin"),
    k8s.V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value="minioadmin"),
    k8s.V1EnvVar(name="VM_URL", value="http://victoriametrics.monitoring.svc.cluster.local:8428"),
    k8s.V1EnvVar(name="MONTHS", value="2024-01,2024-02,2024-03"),
]

POD_RESOURCES = k8s.V1ResourceRequirements(
    requests={"memory": "512Mi", "cpu": "250m"},
    limits={"memory": "3Gi"},
)

with DAG(
    dag_id="taxi_download_data",
    description="Download NYC Taxi raw data to Delta Lake bronze/raw",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["taxi", "data", "bronze"],
) as dag:

    download = KubernetesPodOperator(
        task_id="download_data",
        name="taxi-download-data",
        image="nyc-taxi-model:latest",
        image_pull_policy="Never",
        cmds=[".venv/bin/python", "src/download_data.py"],
        env_vars=ENV_VARS,
        namespace="training",
        service_account_name="workflow-sa",
        get_logs=True,
        container_resources=POD_RESOURCES,
        outlets=[BRONZE_RAW],
    )
