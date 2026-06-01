#!/usr/bin/env bash
# Weekly monitoring: regime_watchdog + regime_drift_monitor + heartbeat.
# Prefer: mlbot monitor weekly  (see docs/strategy/漂移监控_mlbot_monitor_CN.md)
#
# Required env (no train_final fallback — see C1):
#   WATCHDOG_PARQUET  — short window for gate / PSI / bull_share (e.g. bus 7d export)
#   DRIFT_PARQUET     — long window for regime plateau (e.g. 6m segment); defaults to WATCHDOG_PARQUET
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
RUN_TS="$(date -u +%Y%m%d_%H%M)"
OUTDIR="results/monitoring/weekly_watchdog/${RUN_TS}"
mkdir -p "$OUTDIR"

WATCHDOG_PARQ="${WATCHDOG_PARQUET:-}"
DRIFT_PARQ="${DRIFT_PARQUET:-${WATCHDOG_PARQUET:-}}"

if [[ -z "$WATCHDOG_PARQ" || ! -f "$WATCHDOG_PARQ" ]]; then
  echo "ERROR: set WATCHDOG_PARQUET to an existing features parquet (see docs/strategy/漂移监控_mlbot_monitor_CN.md §7)" >&2
  exit 3
fi
if [[ -z "$DRIFT_PARQ" || ! -f "$DRIFT_PARQ" ]]; then
  echo "ERROR: set DRIFT_PARQUET (or WATCHDOG_PARQUET) to an existing features parquet" >&2
  exit 3
fi

STATUS="OK"
EXIT=0

PYTHONPATH=src:scripts python scripts/regime_watchdog.py \
  --strategies bpc,tpc,me,srb \
  --window-parquet "$WATCHDOG_PARQ" \
  --baseline-json config/monitoring/regime_watchdog_baseline.json \
  --out-dir "$OUTDIR/watchdog" \
  || { STATUS="ALERT"; EXIT=1; }

PYTHONPATH=src:scripts python scripts/regime_drift_monitor.py \
  --strategies bpc,tpc,me,srb \
  --window-parquet "$DRIFT_PARQ" \
  --out-dir "$OUTDIR/drift" \
  || { STATUS="ALERT"; EXIT=1; }

cat > "$OUTDIR/heartbeat.json" <<EOF
{"task": "weekly_watchdog", "ts": "$(date -u --iso-8601=seconds)", "status": "${STATUS}", "watchdog_parquet": "${WATCHDOG_PARQ}", "drift_parquet": "${DRIFT_PARQ}"}
EOF

echo "monitoring: $OUTDIR (status=${STATUS})"
exit "${EXIT}"
