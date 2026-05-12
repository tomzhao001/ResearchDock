#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
COMPOSE_FILE="${COMPOSE_FILE:-$REPO_ROOT/docker-compose.yml}"

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

load_env_file "$ENV_FILE"

require_env "ACR_REGISTRY"
require_env "IMAGE_NAMESPACE"

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
  frontend:
    image: ${FRONTEND_IMAGE}
    build: null
EOF

echo "Pulling images"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" -f "$OVERRIDE_FILE" pull backend celery-worker frontend

echo "Starting services"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" -f "$OVERRIDE_FILE" up -d --remove-orphans --no-build

echo "Done."
