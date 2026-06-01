#!/usr/bin/env bash
# Telegram alert for mlbot weekly monitor (systemd OnFailure).
set -euo pipefail

UNIT="${1:-mlbot-weekly-watchdog.service}"
STATE="${2:-failed}"
HOST="$(hostname -s 2>/dev/null || hostname)"
COOLDOWN_SEC="${MLBOT_TG_NOTIFY_COOLDOWN_SEC:-600}"

ENV_FILE="${MLBOT_MONITORING_ENV:-${QUANT_MONITORING_ENV:-/opt/quant-engine/monitoring/.env}}"
STAMP_DIR="${MLBOT_TG_NOTIFY_STAMP_DIR:-/tmp}"
STAMP="${STAMP_DIR}/mlbot_monitor_tg_${UNIT//\//_}.stamp"

if [[ -f "$STAMP" ]]; then
  last="$(stat -c %Y "$STAMP" 2>/dev/null || echo 0)"
  now="$(date +%s)"
  if (( now - last < COOLDOWN_SEC )); then
    exit 0
  fi
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "monitor_telegram_notify: missing $ENV_FILE" >&2
  exit 0
fi

# shellcheck disable=SC1090
set -a
source "$ENV_FILE"
set +a

TOKEN="${GRAFANA_ALERT_TELEGRAM_BOT_TOKEN:-}"
CHAT="${GRAFANA_ALERT_TELEGRAM_CHAT_ID:-}"
if [[ -z "$TOKEN" || -z "$CHAT" ]]; then
  echo "monitor_telegram_notify: Telegram not configured in $ENV_FILE" >&2
  exit 0
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SUMMARY=""
latest_hb="$(find "$ROOT/results/monitoring" -name heartbeat.json -type f 2>/dev/null | sort -r | head -1 || true)"
if [[ -n "$latest_hb" && -f "$latest_hb" ]]; then
  SUMMARY="$(python3 - "$latest_hb" <<'PY'
import json, sys
path = sys.argv[1]
try:
    hb = json.load(open(path, encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit
parts = [
    f"task={hb.get('task','?')}",
    f"status={hb.get('status','?')}",
]
for k in ("watchdog_parquet", "drift_parquet", "output_dir"):
    if hb.get(k):
        parts.append(f"{k}={hb[k]}")
print(" | ".join(parts))
PY
)"
fi

MSG="⚠️ mlbot monitor ${UNIT} ${STATE} on ${HOST} ($(date -u +'%Y-%m-%d %H:%M:%S UTC'))"
if [[ -n "$SUMMARY" ]]; then
  MSG="${MSG}
${SUMMARY}"
fi

ENC_MSG="$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))' "$MSG")"
curl -fsS --max-time 15 \
  "https://api.telegram.org/bot${TOKEN}/sendMessage" \
  -d "chat_id=${CHAT}" \
  -d "text=${ENC_MSG}" \
  -d "disable_web_page_preview=true" \
  >/dev/null

touch "$STAMP"
echo "monitor_telegram_notify: sent for ${UNIT}"
