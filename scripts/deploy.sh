#!/usr/bin/env bash
set -euo pipefail

# 인자로 환경변수를 받는다 (GitHub Actions에서 secrets로 전달)
MFDS_PILL_SERVICE_KEY="${1:?MFDS_PILL_SERVICE_KEY is required}"
GEMINI_API_KEY="${2:-}"
MYSQL_HOST="${3:-127.0.0.1}"
MYSQL_PORT="${4:-3306}"
MYSQL_USER="${5:-root}"
MYSQL_PASSWORD="${6:-}"
MYSQL_DB="${7:-pillcare}"

IMAGE_NAME="pill-ai"
CONTAINER_NAME="pill-ai"

echo "=== git pull ==="
git pull origin main

echo "=== Docker build ==="
docker build -t "$IMAGE_NAME" .

echo "=== Stop existing container ==="
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

echo "=== Start new container ==="
docker run -d \
  --name "$CONTAINER_NAME" \
  -p 8000:8000 \
  --restart unless-stopped \
  -e MFDS_PILL_SERVICE_KEY="$MFDS_PILL_SERVICE_KEY" \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  -e MYSQL_HOST="$MYSQL_HOST" \
  -e MYSQL_PORT="$MYSQL_PORT" \
  -e MYSQL_USER="$MYSQL_USER" \
  -e MYSQL_PASSWORD="$MYSQL_PASSWORD" \
  -e MYSQL_DB="$MYSQL_DB" \
  "$IMAGE_NAME"

echo "=== Health check ==="
for i in $(seq 1 10); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "Health check passed!"
    break
  fi
  if [ "$i" -eq 10 ]; then
    echo "Health check failed after 10 attempts"
    docker logs "$CONTAINER_NAME"
    exit 1
  fi
  echo "Waiting for app to start... ($i/10)"
  sleep 3
done

echo "=== Clean up old images ==="
docker image prune -f

echo "=== Deploy complete ==="
