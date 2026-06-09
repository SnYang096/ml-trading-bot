#!/usr/bin/env bash
# Run mlbot monitor schedule inside quant-engine:latest (remote /opt/quant-engine).
set -euo pipefail

CADENCE="${1:?usage: mlbot-monitor-docker-run.sh <cadence>}"
IMAGE="${MLBOT_MONITOR_IMAGE:-quant-engine:latest}"

exec docker run --rm \
  --name "mlbot-monitor-${CADENCE}-$$" \
  -e MLBOT_FEATURE_BUS_ROOT=/app/live/shared_feature_bus \
  -v /opt/quant-engine/live/shared_feature_bus:/app/live/shared_feature_bus:ro \
  -v /opt/quant-engine/results:/app/results \
  -v /opt/quant-engine/live/highcap/config:/app/live/highcap/config:ro \
  -v /opt/quant-engine/config/monitoring:/app/config/monitoring:ro \
  "${IMAGE}" \
  python scripts/monitoring/monitor_scheduler.py --cadence "${CADENCE}"
