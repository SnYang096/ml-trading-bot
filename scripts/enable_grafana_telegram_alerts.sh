#!/usr/bin/env bash
# Enable Grafana Telegram contact point + notification policy on the monitoring host.
# Requires monitoring/.env with GRAFANA_ALERT_TELEGRAM_BOT_TOKEN set.
set -euo pipefail

ROOT="${1:-/opt/quant-engine/monitoring}"
ALERT_DIR="${ROOT}/grafana-provisioning/alerting"
ENV_FILE="${ROOT}/.env"

if [ ! -f "${ENV_FILE}" ]; then
  echo "Missing ${ENV_FILE}; cp .env.example .env and set GRAFANA_ALERT_TELEGRAM_BOT_TOKEN" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"
if [ -z "${GRAFANA_ALERT_TELEGRAM_BOT_TOKEN:-}" ]; then
  echo "GRAFANA_ALERT_TELEGRAM_BOT_TOKEN is empty in ${ENV_FILE}" >&2
  exit 1
fi

cp "${ALERT_DIR}/contact-points.telegram.yml.template" "${ALERT_DIR}/contact-points.telegram.yml"
cp "${ALERT_DIR}/notification-policies.telegram.yml.template" "${ALERT_DIR}/notification-policies.yml"
echo "Telegram provisioning files installed under ${ALERT_DIR}"

cd "${ROOT}"
docker compose -f docker-compose.monitoring.yml up -d grafana promtail
echo "Restarted grafana + promtail"
