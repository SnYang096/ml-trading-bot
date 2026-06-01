#!/usr/bin/env bash
# Weekly monitoring: regime_watchdog + regime_drift_monitor + heartbeat.
# Prefer: mlbot monitor run --config config/monitoring/weekly_rule_stack.yaml
#     or: mlbot monitor weekly
#
# Parquets (no train_final fallback — see C1):
#   WATCHDOG_PARQUET  — short window (7d bus export)
#   DRIFT_PARQUET     — long window (bus export, default all rows in bus); else WATCHDOG_PARQUET
#
# When MLBOT_MONITOR_AUTO_WINDOW=1 and parquets are unset, this script produces them first.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
RUN_TS="$(date -u +%Y%m%d_%H%M)"
OUTDIR="results/monitoring/weekly_watchdog/${RUN_TS}"
WINDOW_DIR="results/monitoring/window/${RUN_TS}"
mkdir -p "$OUTDIR" "$WINDOW_DIR"

WATCHDOG_PARQ="${WATCHDOG_PARQUET:-}"
DRIFT_PARQ="${DRIFT_PARQUET:-}"
AUTO="${MLBOT_MONITOR_AUTO_WINDOW:-0}"

_py() {
  PYTHONPATH=src:scripts python "$@"
}

_produce_short() {
  local out="${WINDOW_DIR}/features_current_7d.parquet"
  echo "monitoring: export-window → ${out}"
  _py scripts/monitoring/export_feature_bus_window.py \
    --lookback-days "${MLBOT_WATCHDOG_LOOKBACK_DAYS:-7}" \
    --output "${out}"
  WATCHDOG_PARQ="${out}"
}

_produce_long() {
  local out="${WINDOW_DIR}/features_current_long.parquet"
  echo "monitoring: export-window (long/bus) → ${out}"
  _py scripts/monitoring/export_feature_bus_window.py \
    --lookback-days "${MLBOT_DRIFT_LOOKBACK_DAYS:-0}" \
    --output "${out}"
  DRIFT_PARQ="${out}"
}

if [[ -z "$WATCHDOG_PARQ" ]]; then
  if [[ "$AUTO" == "1" ]]; then
    _produce_short
  else
    echo "ERROR: set WATCHDOG_PARQUET or MLBOT_MONITOR_AUTO_WINDOW=1 (see docs/strategy/漂移监控_mlbot_monitor_CN.md §7)" >&2
    exit 3
  fi
fi
if [[ ! -f "$WATCHDOG_PARQ" ]]; then
  echo "ERROR: WATCHDOG_PARQUET not found: ${WATCHDOG_PARQ}" >&2
  exit 3
fi

if [[ -z "$DRIFT_PARQ" ]]; then
  if [[ "$AUTO" == "1" ]]; then
    _produce_long
  else
    DRIFT_PARQ="$WATCHDOG_PARQ"
  fi
fi
if [[ ! -f "$DRIFT_PARQ" ]]; then
  echo "ERROR: DRIFT_PARQUET not found: ${DRIFT_PARQ}" >&2
  exit 3
fi

export WATCHDOG_PARQUET="$WATCHDOG_PARQ"
export DRIFT_PARQUET="$DRIFT_PARQ"

STATUS="OK"
EXIT=0

_py scripts/regime_watchdog.py \
  --strategies bpc,tpc,me,srb \
  --window-parquet "$WATCHDOG_PARQ" \
  --baseline-json config/monitoring/regime_watchdog_baseline.json \
  --out-dir "$OUTDIR/watchdog" \
  || { STATUS="ALERT"; EXIT=1; }

_py scripts/regime_drift_monitor.py \
  --strategies bpc,tpc,me,srb \
  --window-parquet "$DRIFT_PARQ" \
  --out-dir "$OUTDIR/drift" \
  || { STATUS="ALERT"; EXIT=1; }

cat > "$OUTDIR/heartbeat.json" <<EOF
{"task": "weekly_watchdog", "ts": "$(date -u --iso-8601=seconds)", "status": "${STATUS}", "watchdog_parquet": "${WATCHDOG_PARQ}", "drift_parquet": "${DRIFT_PARQ}", "output_dir": "${OUTDIR}"}
EOF

echo "monitoring: $OUTDIR (status=${STATUS})"
exit "${EXIT}"
