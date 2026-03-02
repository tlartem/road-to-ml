"""
NYC Taxi Feature Engineering + Feast Materialization.

DAG: feature_engineering → feast_materialize
Manual trigger. Computes zone-level stats and pushes to Feast online store.
Pods run in namespace 'feast' to access the feast-data PVC.
"""

from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

from datasets import GOLD_ZONE_STATS

ENV_VARS = [
    k8s.V1EnvVar(name="MLFLOW_S3_ENDPOINT_URL", value="http://minio.minio.svc.cluster.local:9000"),
    k8s.V1EnvVar(name="AWS_ACCESS_KEY_ID", value="minioadmin"),
    k8s.V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value="minioadmin"),
]

FEAST_VOLUME = k8s.V1Volume(
    name="feast-data",
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
        claim_name="feast-data",
    ),
)

FEAST_MOUNT = k8s.V1VolumeMount(
    name="feast-data",
    mount_path="/feast-data",
)

with DAG(
    dag_id="taxi_feature_engineering",
    description="Compute zone stats and materialize to Feast online store",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["taxi", "feast", "features"],
) as dag:

    feature_eng = KubernetesPodOperator(
        task_id="feature_engineering",
        name="taxi-feature-engineering",
        image="nyc-taxi-model:latest",
        image_pull_policy="Never",
        cmds=[".venv/bin/python", "src/feature_engineering.py"],
        env_vars=ENV_VARS,
        namespace="feast",
        service_account_name="workflow-sa",
        volumes=[FEAST_VOLUME],
        volume_mounts=[FEAST_MOUNT],
        get_logs=True,
        outlets=[GOLD_ZONE_STATS],
    )

    feast_materialize = KubernetesPodOperator(
        task_id="feast_materialize",
        name="taxi-feast-materialize",
        image="nyc-taxi-model:latest",
        image_pull_policy="Never",
        cmds=[".venv/bin/python", "src/feast_materialize.py"],
        namespace="feast",
        service_account_name="workflow-sa",
        volumes=[FEAST_VOLUME],
        volume_mounts=[FEAST_MOUNT],
        get_logs=True,
    )

    feature_eng >> feast_materialize
