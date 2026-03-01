# MLOps Platform Bootstrap

## Prerequisites

- OrbStack (K8s)
- Docker / docker-compose
- `kubeseal` CLI (`brew install kubeseal`)
- `/etc/hosts`:
```
127.0.0.1 minio.local minio-api.local mlflow.local airflow.local model.local gitea.local argocd.local grafana.local vm.local
```

## 1. Gitea + Act Runner (docker-compose — вне K8s)

```bash
# Получить registration token через API:
curl -s -u root:root 'http://gitea.local:3000/api/v1/user/actions/runners/registration-token' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])"

# Записать в .env и запустить:
echo "RUNNER_TOKEN=<token>" > .env
docker compose up -d
```

UI: http://gitea.local:3000 (root / root)

Act runner автоматически зарегистрируется при первом запуске.
Для repo `water-quality-model` — включить Actions в Settings → Advanced.

**Важно**: `container.network` в конфиге act-runner НЕ использовать — вызывает зависание
`docker create` на OrbStack/Docker 28.x. Вместо этого используется `--add-host gitea:host-gateway`
в `container.options`, а runner создаёт временную сеть для каждого job.

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

## 6. ML-образы и CI/CD

**Автоматическая сборка**: push в main `root/water-quality-model` → Gitea Actions
собирает `water-quality-model:sha-XXXXXXX` и `water-quality-serving:sha-XXXXXXX` (+ `:latest`).

**Деплой**: кнопка "Run Workflow" на `deploy.yaml` в Gitea UI
(или API: `POST /api/v1/repos/root/water-quality-model/actions/workflows/deploy.yaml/dispatches`)
→ обновляет image tag в `serving/serving.yaml` → ArgoCD синхронизирует.

**Настройка CI (один раз)**:
```bash
# Создать API token
curl -s -u root:root -X POST 'http://gitea.local:3000/api/v1/users/root/tokens' \
  -H 'Content-Type: application/json' -d '{"name":"ci-deploy","scopes":["write:repository"]}'
# Добавить token как secret DEPLOY_TOKEN в repo water-quality-model:
# Gitea UI → Settings → Actions → Secrets
```

MLflow-образ — вручную:
```bash
docker build -t mlflow-custom:v2.19.0 <path-to-mlflow-dockerfile>/
```

## 7. Мониторинг (VictoriaMetrics + Grafana)

Манифесты в `monitoring/` — ArgoCD синхронизирует автоматически через `argocd/apps/monitoring.yaml`.

Добавить в `/etc/hosts` (если ещё не добавлены):
```
127.0.0.1 grafana.local vm.local
```

Компоненты:
- **VictoriaMetrics** — TSDB, Prometheus-совместимый, scrape метрик
- **Grafana** — дашборды (admin / admin)
- **node-exporter** — метрики ноды (CPU, RAM, disk)
- **kube-state-metrics** — метрики объектов K8s (pods, deployments)

UI:
- http://grafana.local:30448 (admin / admin)
- http://vm.local:30448 (VictoriaMetrics UI, PromQL запросы)

Проверка:
```bash
kubectl get pods -n monitoring
# Все поды должны быть Running

# VictoriaMetrics — запрос up показывает scrape targets
curl -s 'http://vm.local:30448/api/v1/query?query=up' | python3 -c "import sys,json; [print(r['metric']['job'], r['value'][1]) for r in json.load(sys.stdin)['data']['result']]"
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
