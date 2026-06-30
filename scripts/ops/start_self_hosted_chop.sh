#!/usr/bin/env bash
# Self-hosted chop_grid: shadow | feature-bus | multileg (testnet/mainnet).
# See docs/strategy/自建服务器部署_chop_grid_CN.md
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEPLOY_ROOT="${DEPLOY_ROOT:-/opt/quant-engine}"
IMAGE="${QUANT_IMAGE:-quant-engine:latest}"
ENV_FILE="${ENV_FILE:-$DEPLOY_ROOT/live/binance_mainnet.env}"

usage() {
  cat <<'EOF'
Usage:
  DEPLOY_ROOT=/opt/quant-engine ./scripts/ops/start_self_hosted_chop.sh shadow
  DEPLOY_ROOT=/opt/quant-engine ./scripts/ops/start_self_hosted_chop.sh feature-bus
  DEPLOY_ROOT=/opt/quant-engine ./scripts/ops/start_self_hosted_chop.sh multileg-testnet [--no-orders]
  DEPLOY_ROOT=/opt/quant-engine ./scripts/ops/start_self_hosted_chop.sh multileg-mainnet

Prereqs:
  docker build -f docker/Dockerfile.live -t quant-engine:latest .
  data/parquet_data on host at $DEPLOY_ROOT/data/parquet_data
  For multileg-*: $ENV_FILE with MULTI_LEG_BINANCE_FUTURES_* keys
EOF
}

vols() {
  echo \
    -v "$DEPLOY_ROOT/data:/app/data" \
    -v "$DEPLOY_ROOT/live/shared_feature_bus:/app/live/shared_feature_bus" \
    -v "$DEPLOY_ROOT/live/highcap/data:/app/live/highcap/data"
}

cmd="${1:-}"
shift || true

case "$cmd" in
  shadow)
    exec docker run --rm -it \
      $(vols) \
      "$IMAGE" \
      python scripts/run_multi_leg_live.py \
        --mode shadow \
        --bar-source parquet \
        --strategies chop_grid \
        --symbols BTCUSDT \
        --data-dir data/parquet_data \
        --once
    ;;
  feature-bus)
    exec docker run --rm --name quant-feature-bus \
      -p 9192:9090 \
      -e MLBOT_LIVE_STORAGE_BASE=/app/live/highcap/data \
      -e MLBOT_FEATURE_BUS_ROOT=/app/live/shared_feature_bus \
      $(vols) \
      "$IMAGE" \
      python scripts/run_market_feature_publisher.py \
        --universe highcap \
        --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
        --feature-bus-root live/shared_feature_bus \
        --live-storage-base live/highcap/data \
        --strategies-root live/highcap/config/strategies \
        --warmup-days 7 \
        --max-rows 10080
    ;;
  multileg-testnet|multileg-mainnet)
  MODE="testnet"
  [[ "$cmd" == "multileg-mainnet" ]] && MODE="mainnet"
  EXTRA=()
  for arg in "$@"; do EXTRA+=("$arg"); done
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing $ENV_FILE" >&2
    exit 1
  fi
  exec docker run --rm --name quant-hedge-multileg \
    -p 9191:9090 \
    --env-file "$ENV_FILE" \
    -e MLBOT_ACCOUNT_SCOPE=multi_leg \
    $(vols) \
    "$IMAGE" \
    python scripts/run_multi_leg_live.py \
      --mode "$MODE" \
      --strategies chop_grid \
      --universe highcap \
      --bar-source feature-store \
      --feature-bus-root live/shared_feature_bus \
      --feature-store-timeframe 120T \
      --feature-store-execution-timeframe 1min \
      --poll-seconds 60 \
      --state-dir data/multi_leg_live/state \
      "${EXTRA[@]}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
