#!/usr/bin/env bash
# Run rebalance cockpit check inside quant-engine:latest (every 4h timer).
set -euo pipefail

IMAGE="${MLBOT_MONITOR_IMAGE:-quant-engine:latest}"
ENV_FILE="${MLBOT_REBALANCE_ENV_FILE:-/opt/quant-engine/.env}"

exec docker run --rm \
  --name "mlbot-rebalance-cockpit-$$" \
  --env-file "${ENV_FILE}" \
  -e MLBOT_FEATURE_BUS_ROOT=/app/live/shared_feature_bus \
  -e MLBOT_CONSOLE_STRATEGIES_ROOT=/app/live/highcap/config/strategies \
  -e MLBOT_CONSOLE_LIVE_ROOT=/app/live/highcap \
  -e MLBOT_MONITORING_ENV=/opt/quant-engine/monitoring/.env \
  -v /opt/quant-engine/live/shared_feature_bus:/app/live/shared_feature_bus:ro \
  -v /opt/quant-engine/results:/app/results \
  -v /opt/quant-engine/live/highcap/config:/app/live/highcap/config:ro \
  -v /opt/quant-engine/config/monitoring:/app/config/monitoring:ro \
  -v /opt/quant-engine/monitoring/.env:/opt/quant-engine/monitoring/.env:ro \
  "${IMAGE}" \
  python scripts/monitoring/rebalance_cockpit_check.py "$@"
