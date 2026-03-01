"""
Water Quality Model Training Pipeline.

DAG: validate → preprocess → train
Каждый шаг запускается как отдельный K8s Pod с образом water-quality-model.
"""

from datetime import datetime
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

# Общие env-переменные для всех шагов
ENV_VARS = [
    k8s.V1EnvVar(name="MLFLOW_TRACKING_URI", value="http://mlflow.mlflow.svc.cluster.local:5000"),
    k8s.V1EnvVar(name="MLFLOW_S3_ENDPOINT_URL", value="http://minio.minio.svc.cluster.local:9000"),
    k8s.V1EnvVar(name="AWS_ACCESS_KEY_ID", value="minioadmin"),
    k8s.V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value="minioadmin"),
]

with DAG(
    dag_id="water_quality_train",
    description="Train water quality classification model",
    schedule=None,                    # ручной запуск (можно поставить cron: "0 3 * * 1")
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "training"],
) as dag:

    validate = KubernetesPodOperator(
        task_id="validate_data",
        name="validate-data",
        image="water-quality-model:latest",
        image_pull_policy="Never",
        cmds=["python", "src/validate.py"],
        env_vars=ENV_VARS,
        namespace="training",
        service_account_name="workflow-sa",
        get_logs=True,
    )

    preprocess = KubernetesPodOperator(
        task_id="preprocess",
        name="preprocess",
        image="water-quality-model:latest",
        image_pull_policy="Never",
        cmds=["python", "src/preprocess.py"],
        env_vars=ENV_VARS,
        namespace="training",
        service_account_name="workflow-sa",
        get_logs=True,
    )

    train = KubernetesPodOperator(
        task_id="train_model",
        name="train-model",
        image="water-quality-model:latest",
        image_pull_policy="Never",
        cmds=["python", "src/train.py"],
        env_vars=ENV_VARS,
        namespace="training",
        service_account_name="workflow-sa",
        get_logs=True,
    )

    validate >> preprocess >> train
