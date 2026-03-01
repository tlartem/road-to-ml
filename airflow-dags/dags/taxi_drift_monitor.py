"""
NYC Taxi Drift Monitoring.

Runs every 5 minutes. Compares reference (training) data with latest
streaming batches using Evidently. Pushes drift metrics to VictoriaMetrics.
"""

from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

ENV_VARS = [
    k8s.V1EnvVar(name="MLFLOW_S3_ENDPOINT_URL", value="http://minio.minio.svc.cluster.local:9000"),
    k8s.V1EnvVar(name="AWS_ACCESS_KEY_ID", value="minioadmin"),
    k8s.V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value="minioadmin"),
    k8s.V1EnvVar(name="VM_URL", value="http://victoriametrics.monitoring.svc.cluster.local:8428"),
]

with DAG(
    dag_id="taxi_drift_monitor",
    description="Monitor data drift with Evidently, push metrics to VictoriaMetrics",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["taxi", "monitoring", "drift"],
) as dag:

    monitor = KubernetesPodOperator(
        task_id="check_drift",
        name="taxi-drift-monitor",
        image="nyc-taxi-model:latest",
        image_pull_policy="Never",
        cmds=["python", "src/monitor_drift.py"],
        env_vars=ENV_VARS,
        namespace="training",
        service_account_name="workflow-sa",
        get_logs=True,
    )
