#!/usr/bin/env bash
# Build (Python-only image) + start business console. Run from repo root or this dir.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$HERE"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

echo "=== business-console preflight ==="
"${HERE}/ensure_docker_compose.sh"
for p in \
  /opt/quant-engine/.env \
  /opt/quant-engine/live/shared_feature_bus \
  /opt/quant-engine/live/highcap \
  /opt/quant-engine/live/highcap/data \
  /opt/quant-engine/data \
  /opt/quant-engine/results; do
  [[ -e "$p" ]] || fail "missing path: $p"
done
[[ -r /opt/quant-engine/.env ]] || fail "/opt/quant-engine/.env not readable (chown deploy user; chmod 600)"

if [[ ! -f "$ROOT/src/mlbot_console/static/dist/index.html" ]]; then
  fail "missing frontend dist — CI should run npm run build before pack; local: make frontend-build"
fi

if ! docker info >/dev/null 2>&1; then
  fail "cannot access docker daemon (add $(whoami) to group docker, or run deploy SSH as root)"
fi

echo "=== docker compose up --build ==="
docker compose up -d --build --remove-orphans

sleep 3
if ! docker ps --filter name=mlbot-business-console --filter status=running -q | grep -q .; then
  echo "--- container not running; logs ---" >&2
  docker logs mlbot-business-console --tail 100 2>&1 || true
  fail "mlbot-business-console is not running"
fi

if ! curl -fsS http://127.0.0.1:8800/api/health >/dev/null; then
  docker logs mlbot-business-console --tail 100 2>&1 || true
  fail "health check failed on http://127.0.0.1:8800/api/health"
fi

echo "OK: http://127.0.0.1:8800/trade-map"
curl -fsS http://127.0.0.1:8800/api/health | head -c 400
echo ""
