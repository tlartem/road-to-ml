# MLOps Platform Bootstrap

## Prerequisites

- OrbStack (K8s)
- Docker / docker-compose
- `kubeseal` CLI (`brew install kubeseal`)
- `/etc/hosts`:
```
127.0.0.1 minio.local minio-api.local mlflow.local airflow.local model.local gitea.local argocd.local
```

## 1. Gitea + Act Runner (docker-compose — вне K8s)

```bash
# Первый запуск: получить registration token из Gitea UI (Admin → Runners)
# или через API, и передать в RUNNER_TOKEN
RUNNER_TOKEN=<token> docker compose up -d
```

UI: http://gitea.local:3000 (root / root)

Act runner автоматически зарегистрируется при первом запуске.
Для repo `water-quality-model` — включить Actions в Settings → Advanced.

## 2. Ingress Controller (nginx)

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.12.0/deploy/static/provider/cloud/deploy.yaml
```

## 3. ArgoCD

```bash
kubectl apply -f argocd/namespace.yaml
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml --server-side --force-conflicts
kubectl delete networkpolicy --all -n argocd

# Отключить HTTPS-редирект (для ingress без TLS)
kubectl -n argocd patch configmap argocd-cmd-params-cm --type merge -p '{"data":{"server.insecure":"true"}}'
kubectl -n argocd rollout restart deployment argocd-server

kubectl apply -f argocd/ingress.yaml
```

Получить admin пароль:
```bash
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d; echo
```

UI: http://argocd.local:30448 (admin / <пароль выше>)

## 4. Bootstrap secrets (один раз вручную)

ArgoCD нужен доступ к Gitea для первой синхронизации, а SealedSecret для repo-creds
нужно применить вручную (argocd/ не управляется ArgoCD — bootstrap-директория):
```bash
kubectl apply -f argocd/repo-secret.yaml
kubectl apply -f argocd/sealed-repo-secret.yaml
```

После этого контроллер создаст Secret из SealedSecret, и ручной repo-secret можно удалить.

## 5. App-of-apps (bootstrap ArgoCD)

```bash
kubectl apply -f argocd/app-of-apps.yaml
```

ArgoCD автоматически синхронизирует: sealed-secrets, minio, mlflow, airflow, serving.
Sealed Secrets контроллер расшифрует SealedSecrets → создаст обычные Secrets → поды подхватят.

## 6. ML-образы

`water-quality-model` и `water-quality-serving` собираются автоматически через Gitea Actions
при push в main ветку `root/water-quality-model`. MLflow-образ — вручную:
```bash
docker build -t mlflow-custom:v2.19.0 <path-to-mlflow-dockerfile>/
```

## GitOps workflow

1. Изменить манифест в клоне этого репозитория
2. Commit + push в Gitea
3. ArgoCD автоматически обнаружит изменение и применит его

## Ротация секретов

При необходимости изменить значения секретов:
```bash
# Создать обычный Secret → зашифровать → сохранить в репо
kubectl create secret generic <name> --namespace=<ns> \
  --from-literal=key=value \
  --dry-run=client -o yaml | kubeseal --format yaml > <path>/sealed-secret.yaml

# Commit + push — ArgoCD синхронизирует, контроллер расшифрует
```

При ротации ключа контроллера (re-seal все секреты):
```bash
kubeseal --fetch-cert > /tmp/sealed-secrets-cert.pem
# Пересоздать каждый SealedSecret с новым сертификатом
```
