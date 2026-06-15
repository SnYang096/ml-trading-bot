#!/usr/bin/env bash
# Run mlbot monitor schedule inside quant-engine:latest (remote /opt/quant-engine).
# For daily cadence, also runs truth sync health check after the main monitor.
set -euo pipefail

CADENCE="${1:?usage: mlbot-monitor-docker-run.sh <cadence>}"
IMAGE="${MLBOT_MONITOR_IMAGE:-quant-engine:latest}"

# Common volume mounts
VOLUMES=(
  -v /opt/quant-engine/live/shared_feature_bus:/app/live/shared_feature_bus:ro
  -v /opt/quant-engine/results:/app/results
  -v /opt/quant-engine/live/highcap/config:/app/live/highcap/config:ro
  -v /opt/quant-engine/config/monitoring:/app/config/monitoring:ro
)

# Truth sync health check needs data/ and live/*/position_trackers/
if [[ "${CADENCE}" == "daily" ]]; then
  VOLUMES+=(
    -v /opt/quant-engine/data:/app/data:ro
    -v /opt/quant-engine/live:/app/live:ro
  )
fi

docker run --rm \
  --name "mlbot-monitor-${CADENCE}-$$" \
  -e MLBOT_FEATURE_BUS_ROOT=/app/live/shared_feature_bus \
  "${VOLUMES[@]}" \
  "${IMAGE}" \
  python scripts/monitoring/monitor_scheduler.py --cadence "${CADENCE}"

# Run truth sync health check for daily cadence
if [[ "${CADENCE}" == "daily" ]]; then
  echo "--- Truth Sync Health Check ---"
  docker run --rm \
    --name "mlbot-monitor-truth-sync-$$" \
    -v /opt/quant-engine/data:/app/data:ro \
    -v /opt/quant-engine/live:/app/live:ro \
    "${IMAGE}" \
    python scripts/check_truth_sync_health.py --days 1 || {
    echo "WARNING: truth sync health check failed"
    exit 1
  }
fi
