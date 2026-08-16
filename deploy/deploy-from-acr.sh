#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
COMPOSE_FILE="${COMPOSE_FILE:-$REPO_ROOT/docker-compose.yml}"
DEPLOY_TARGET="${1:-all}"

load_env_file() {
  local path="$1"

  [[ -f "$path" ]] || return 0

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue

    local key="${line%%=*}"
    local value="${line#*=}"
    value="${value%$'\r'}"
    key="${key%"${key##*[![:space:]]}"}"

    [[ -z "$key" ]] && continue

    if [[ -z "${!key:-}" ]]; then
      if [[ ${#value} -ge 2 ]]; then
        if [[ "${value:0:1}" == "\"" && "${value: -1}" == "\"" ]]; then
          value="${value:1:${#value}-2}"
        elif [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
          value="${value:1:${#value}-2}"
        fi
      fi

      export "$key=$value"
    fi
  done < "$path"
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
}

# Warn if leftover celery-queue messages may still be PDF extract tasks.
# Never blocks deploy. Runs redis-cli inside the redis container.
check_celery_queue_backlog() {
  local llen
  llen=$(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" exec -T redis redis-cli -n 1 LLEN celery 2>/dev/null) || true

  if [[ -n "$llen" && "$llen" =~ ^[0-9]+$ && "$llen" -gt 0 ]]; then
    printf '\033[33m%s\033[0m\n' "检测到 celery 队列仍有 ${llen} 条遗留消息。遗留 celery 队列中的旧 PDF 提取任务会被主 worker 转交 extract 队列执行，extract worker 内存充足，一般不会 OOM；仍建议发布前确认队列无 in-flight 遗留任务。"
  fi
  return 0
}

load_env_file "$ENV_FILE"

require_env "ACR_REGISTRY"
require_env "IMAGE_NAMESPACE"

case "$DEPLOY_TARGET" in
  all)
    UP_SERVICES=()
    ;;
  app)
    UP_SERVICES=(backend frontend celery-worker celery-extract-worker nginx)
    ;;
  -h|--help|help)
    cat <<'EOF'
Usage: bash deploy/deploy-from-acr.sh [all|app]

  all  Recreate and deploy the full compose stack.
  app  Recreate only application services (backend, frontend, celery-worker, celery-extract-worker, nginx).
EOF
    exit 0
    ;;
  *)
    echo "Unsupported deploy target: $DEPLOY_TARGET" >&2
    echo "Use one of: all, app" >&2
    exit 1
    ;;
esac

IMAGE_TAG="${IMAGE_TAG:-latest}"
BACKEND_IMAGE="${ACR_REGISTRY}/${IMAGE_NAMESPACE}/backend:${IMAGE_TAG}"
FRONTEND_IMAGE="${ACR_REGISTRY}/${IMAGE_NAMESPACE}/frontend:${IMAGE_TAG}"

OVERRIDE_FILE="$(mktemp "${TMPDIR:-/tmp}/researchdock-acr-compose.XXXXXX.yml")"
cleanup() {
  rm -f "$OVERRIDE_FILE"
}
trap cleanup EXIT

cat > "$OVERRIDE_FILE" <<EOF
services:
  backend:
    image: ${BACKEND_IMAGE}
    build: null
  celery-worker:
    image: ${BACKEND_IMAGE}
    build: null
  celery-extract-worker:
    image: ${BACKEND_IMAGE}
    build: null
  frontend:
    image: ${FRONTEND_IMAGE}
    build: null
EOF

echo "Pulling images"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" -f "$OVERRIDE_FILE" pull backend celery-worker celery-extract-worker frontend

check_celery_queue_backlog

if [[ "$DEPLOY_TARGET" == "all" ]]; then
  echo "Starting full stack"
else
  echo "Starting application services"
fi
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" -f "$OVERRIDE_FILE" up -d --remove-orphans --no-build --force-recreate "${UP_SERVICES[@]}"

echo "Restarting internal nginx"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" -f "$OVERRIDE_FILE" restart nginx

echo "Done."
