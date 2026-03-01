"""Step 3: Train model and log to MLflow."""

import logging
import os
import sys

import boto3
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def main():
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    log.info("Downloading processed data from MinIO...")
    for name in ["train.csv", "test.csv"]:
        s3.download_file("datasets", f"water_quality/processed/{name}", f"/tmp/{name}")

    train_df = pd.read_csv("/tmp/train.csv")
    test_df = pd.read_csv("/tmp/test.csv")
    log.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    X_train = train_df.drop("Potability", axis=1)
    y_train = train_df["Potability"]
    X_test = test_df.drop("Potability", axis=1)
    y_test = test_df["Potability"]

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("water-quality")

    params = {
        "n_estimators": int(os.environ.get("N_ESTIMATORS", "100")),
        "max_depth": int(os.environ.get("MAX_DEPTH", "10")),
        "random_state": 42,
    }

    with mlflow.start_run():
        mlflow.log_params(params)
        log.info("Training RandomForestClassifier with params: %s", params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        log.info("Training complete")

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)

        log.info("Metrics:")
        for name, value in metrics.items():
            log.info("  %-12s %.4f", name, value)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="water-quality-classifier",
        )
        log.info("Model registered: water-quality-classifier")


if __name__ == "__main__":
    main()
