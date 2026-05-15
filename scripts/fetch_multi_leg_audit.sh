#!/usr/bin/env bash
# Copy multi-leg audit logs from a remote host for local rg/tail (avoids huge SSH reads).
#
#   REMOTE=ubuntu@x.y.z REMOTE_LOG_DIR=/opt/quant-engine/data/multi_leg_live/state/logs \\
#     ./scripts/fetch_multi_leg_audit.sh
#
# Requires: scp, ssh; key via SSH_AUTH_SOCK or -i in ~/.ssh/config.
set -euo pipefail
REMOTE="${REMOTE:-}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/opt/quant-engine/data/multi_leg_live/state/logs}"
DEST="${DEST:-./multi_leg_audit_download}"
if [[ -z "${REMOTE}" ]]; then
  echo "Set REMOTE= user@host" >&2
  exit 1
fi
mkdir -p "${DEST}"
echo "Fetching audit logs from ${REMOTE}:${REMOTE_LOG_DIR} -> ${DEST}/" >&2
scp -o ConnectTimeout=15 -o ServerAliveInterval=10 \
  "${REMOTE}:${REMOTE_LOG_DIR}/multi_leg_audit.log" \
  "${REMOTE}:${REMOTE_LOG_DIR}/multi_leg_audit.log."* \
  "${DEST}/" 2>/dev/null || true
ls -la "${DEST}/" >&2
echo "Local quick search examples:" >&2
echo "  rg 'risk veto|multi-leg cancel|place result|reconcile' \"${DEST}\"" >&2
