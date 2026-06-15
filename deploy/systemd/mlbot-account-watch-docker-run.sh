#!/usr/bin/env bash
# Poll multi-leg account equity / new positions → Telegram (every ~60s via timer).
set -euo pipefail

IMAGE="${MLBOT_MONITOR_IMAGE:-quant-engine:latest}"
ENV_FILE="${MLBOT_ACCOUNT_WATCH_ENV_FILE:-/opt/quant-engine/live/binance_mainnet.env}"
MONITOR_ENV="${MLBOT_MONITORING_ENV:-/opt/quant-engine/monitoring/.env}"
STATE_DIR="${MLBOT_ACCOUNT_TG_STATE_DIR:-/opt/quant-engine/data/monitoring}"
MAINNET_FLAG="${MLBOT_ACCOUNT_WATCH_MAINNET:-1}"

ARGS=(python scripts/monitoring/multi_leg_account_telegram_watch.py --once)
if [[ "${MAINNET_FLAG}" == "1" ]]; then
  ARGS+=(--mainnet)
fi

mkdir -p "${STATE_DIR}"

exec docker run --rm \
  --name "mlbot-account-watch-$$" \
  --env-file "${ENV_FILE}" \
  -e MLBOT_MONITORING_ENV="${MONITOR_ENV}" \
  -e MLBOT_ACCOUNT_TG_STATE="${STATE_DIR}/multi_leg_account_tg_state.json" \
  -v "${STATE_DIR}:${STATE_DIR}" \
  -v "${MONITOR_ENV}:${MONITOR_ENV}:ro" \
  "${IMAGE}" \
  "${ARGS[@]}"
