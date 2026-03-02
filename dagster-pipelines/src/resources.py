"""Dagster resources — dependency injection for MinIO, MLflow, VictoriaMetrics."""

import os

from dagster import ConfigurableResource


class MinIOResource(ConfigurableResource):
    """S3-compatible MinIO connection. Sets env vars for lake.py."""
    endpoint_url: str = "http://minio.minio.svc.cluster.local:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"

    def setup_env(self):
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.endpoint_url
        os.environ["AWS_ACCESS_KEY_ID"] = self.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.secret_key


class MLflowResource(ConfigurableResource):
    """MLflow tracking server connection."""
    tracking_uri: str = "http://mlflow.mlflow.svc.cluster.local:5000"

    def setup_env(self):
        os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get(
            "MLFLOW_S3_ENDPOINT_URL", "http://minio.minio.svc.cluster.local:9000"
        )


class VictoriaMetricsResource(ConfigurableResource):
    """VictoriaMetrics for pushing metrics."""
    url: str = "http://victoriametrics.monitoring.svc.cluster.local:8428"

    def setup_env(self):
        os.environ["VM_URL"] = self.url
