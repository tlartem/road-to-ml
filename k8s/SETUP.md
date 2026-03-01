# MLOps Platform Bootstrap

## Prerequisites

- OrbStack (K8s)
- Docker / docker-compose
- `/etc/hosts`:
```
127.0.0.1 minio.local minio-api.local mlflow.local airflow.local model.local gitea.local argocd.local
```

## 1. Gitea (docker-compose — вне K8s)

```bash
cd ~/road-to-ml
docker compose up -d
```

UI: http://gitea.local:3000 (root / root)

## 2. Ingress Controller (nginx)

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.12.0/deploy/static/provider/cloud/deploy.yaml
```

## 3. ArgoCD

```bash
kubectl apply -f k8s/argocd/namespace.yaml
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml --server-side --force-conflicts
kubectl delete networkpolicy --all -n argocd

# Отключить HTTPS-редирект (для ingress без TLS)
kubectl -n argocd patch configmap argocd-cmd-params-cm --type merge -p '{"data":{"server.insecure":"true"}}'
kubectl -n argocd rollout restart deployment argocd-server

kubectl apply -f k8s/argocd/ingress.yaml
```

Получить admin пароль:
```bash
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d; echo
```

UI: http://argocd.local:30448 (admin / <пароль выше>)

## 4. Secrets (вручную, не в git)

```bash
kubectl apply -f k8s/minio/secret.yaml
kubectl apply -f k8s/mlflow/secret.yaml
kubectl apply -f k8s/argocd/repo-secret.yaml
```

## 5. App-of-apps (bootstrap ArgoCD)

```bash
kubectl apply -f k8s/argocd/app-of-apps.yaml
```

ArgoCD автоматически синхронизирует: minio, mlflow, airflow, serving.

## 6. ML-образы (локальные)

```bash
docker build -t water-quality-model:latest ~/water-quality-model/
docker build -t water-quality-serving:latest -f ~/water-quality-model/Dockerfile.serving ~/water-quality-model/
docker build -t mlflow-custom:v2.19.0 ~/road-to-ml/k8s/mlflow/
```

## GitOps workflow

1. Изменить манифест локально
2. Commit + push в Gitea (`git push gitea main`)
3. ArgoCD автоматически обнаружит изменение и применит его
