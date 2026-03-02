"""
NYC Taxi Delta Lake Compaction.

Runs hourly. Merges small files in bronze/streaming Delta table
to improve read performance.
"""

from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

ENV_VARS = [
    k8s.V1EnvVar(name="MLFLOW_S3_ENDPOINT_URL", value="http://minio.minio.svc.cluster.local:9000"),
    k8s.V1EnvVar(name="AWS_ACCESS_KEY_ID", value="minioadmin"),
    k8s.V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value="minioadmin"),
]

COMPACT_SCRIPT = """
from lake import compact, table_uri
compact(table_uri("bronze", "streaming"))
print("Compaction complete")
"""

with DAG(
    dag_id="taxi_compact",
    description="Compact bronze/streaming Delta table (merge small files)",
    schedule="0 * * * *",  # every hour
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["taxi", "maintenance", "delta"],
) as dag:

    compact = KubernetesPodOperator(
        task_id="compact_streaming",
        name="taxi-compact",
        image="nyc-taxi-model:latest",
        image_pull_policy="Never",
        cmds=[".venv/bin/python", "-c", COMPACT_SCRIPT],
        env_vars=ENV_VARS,
        namespace="training",
        service_account_name="workflow-sa",
        get_logs=True,
    )
