"""
NYC Taxi Data Simulator.

Runs every minute, each invocation simulates 1 hour of taxi data.
Reads raw Parquet from MinIO taxi-raw/, writes batches to taxi-streaming/.

To "jump" to a different date (e.g., July for drift demo):
  Airflow UI → Admin → Variables → set taxi_simulation_date = "2024-07-01"
Clear the variable to continue from the last saved state.
"""

from datetime import datetime

from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

ENV_VARS = [
    k8s.V1EnvVar(name="MLFLOW_S3_ENDPOINT_URL", value="http://minio.minio.svc.cluster.local:9000"),
    k8s.V1EnvVar(name="AWS_ACCESS_KEY_ID", value="minioadmin"),
    k8s.V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value="minioadmin"),
]

with DAG(
    dag_id="taxi_simulator",
    description="Simulate NYC Taxi streaming data (1 hour per run)",
    schedule="*/1 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["taxi", "simulator"],
) as dag:

    # Read jump-to-date from Airflow Variable (empty = continue from state)
    simulation_date = Variable.get("taxi_simulation_date", default_var="")

    env = list(ENV_VARS)
    if simulation_date:
        env.append(k8s.V1EnvVar(name="SIMULATION_DATE", value=simulation_date))

    simulate = KubernetesPodOperator(
        task_id="simulate_batch",
        name="taxi-simulator",
        image="nyc-taxi-model:latest",
        image_pull_policy="Never",
        cmds=["python", "src/simulator.py"],
        env_vars=env,
        namespace="training",
        service_account_name="workflow-sa",
        get_logs=True,
    )
