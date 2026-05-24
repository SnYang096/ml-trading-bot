#!/usr/bin/env bash
# Telegram alert for quant systemd units (OnFailure / ExecStopPost).
# Rate-limited per unit to avoid crash-loop spam.
set -euo pipefail

UNIT="${1:-unknown}"
STATE="${2:-failed}"
HOST="$(hostname -s 2>/dev/null || hostname)"
COOLDOWN_SEC="${QUANT_TG_NOTIFY_COOLDOWN_SEC:-600}"

ENV_FILE="${QUANT_MONITORING_ENV:-/opt/quant-engine/monitoring/.env}"
STAMP_DIR="${QUANT_TG_NOTIFY_STAMP_DIR:-/tmp}"
STAMP="${STAMP_DIR}/quant_tg_notify_${UNIT//\//_}.stamp"

if [[ -f "$STAMP" ]]; then
  last="$(stat -c %Y "$STAMP" 2>/dev/null || echo 0)"
  now="$(date +%s)"
  if (( now - last < COOLDOWN_SEC )); then
    exit 0
  fi
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "quant_telegram_notify: missing $ENV_FILE" >&2
  exit 0
fi

# shellcheck disable=SC1090
set -a
source "$ENV_FILE"
set +a

TOKEN="${GRAFANA_ALERT_TELEGRAM_BOT_TOKEN:-}"
CHAT="${GRAFANA_ALERT_TELEGRAM_CHAT_ID:-}"
if [[ -z "$TOKEN" || -z "$CHAT" ]]; then
  echo "quant_telegram_notify: Telegram not configured in $ENV_FILE" >&2
  exit 0
fi

MSG="⚠️ Quant ${UNIT} ${STATE} on ${HOST} ($(date -u +'%Y-%m-%d %H:%M:%S UTC'))"
ENC_MSG="$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))' "$MSG")"

curl -fsS --max-time 15 \
  "https://api.telegram.org/bot${TOKEN}/sendMessage" \
  -d "chat_id=${CHAT}" \
  -d "text=${ENC_MSG}" \
  -d "disable_web_page_preview=true" \
  >/dev/null

touch "$STAMP"
echo "quant_telegram_notify: sent for ${UNIT}"
