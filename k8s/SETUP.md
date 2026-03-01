# K8s Cluster Setup

## 1. Ingress Controller (nginx)

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.12.0/deploy/static/provider/cloud/deploy.yaml
```

## 2. /etc/hosts

```bash
sudo sh -c 'echo "127.0.0.1 minio.local minio-api.local mlflow.local airflow.local" >> /etc/hosts'
```

## 3. MinIO

```bash
kubectl apply -f k8s/minio/namespace.yaml
kubectl apply -f k8s/minio/secret.yaml
kubectl apply -f k8s/minio/minio.yaml
kubectl apply -f k8s/minio/ingress.yaml
```

## 4. MLflow

```bash
kubectl apply -f k8s/mlflow/namespace.yaml
kubectl apply -f k8s/mlflow/secret.yaml
kubectl apply -f k8s/mlflow/postgres.yaml
kubectl apply -f k8s/mlflow/mlflow.yaml
kubectl apply -f k8s/mlflow/ingress.yaml
```

## 5. Airflow

```bash
kubectl apply -f k8s/airflow/namespace.yaml
kubectl apply -f k8s/airflow/helmchart.yaml
# Подождать ~2 мин пока поднимется
kubectl apply -f k8s/airflow/ingress.yaml
```

UI: http://airflow.local:30448 (admin / admin)

## 6. Запуск пайплайна

```bash
# Собрать ML-образ (из репо water-quality-model)
docker build -t water-quality-model:latest ~/water-quality-model/

# DAGs лежат в ~/airflow-dags/dags/ (монтируются в Airflow через hostPath)
# Запуск через Airflow UI или CLI
```
