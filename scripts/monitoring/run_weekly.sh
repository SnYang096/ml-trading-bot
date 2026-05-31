#!/usr/bin/env bash
# Weekly monitoring: regime watchdog + heartbeat for CMS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
RUN_TS="$(date -u +%Y%m%d_%H%M)"
OUTDIR="results/monitoring/weekly_watchdog/${RUN_TS}"
mkdir -p "$OUTDIR"

PARQ="${WATCHDOG_PARQUET:-}"
if [[ -z "$PARQ" ]]; then
  PARQ="$(ls -t results/train_final/bpc/train_final_*/bpc/features_labeled.parquet 2>/dev/null | head -1 || true)"
fi
if [[ -z "$PARQ" || ! -f "$PARQ" ]]; then
  echo "ERROR: set WATCHDOG_PARQUET or run prepare-only first" >&2
  exit 1
fi

STATUS="OK"
PYTHONPATH=src:scripts python scripts/regime_watchdog.py \
  --strategies bpc,tpc,me,srb \
  --window-parquet "$PARQ" \
  --baseline-json config/monitoring/regime_watchdog_baseline.json \
  --output "$OUTDIR/result.json" \
  || STATUS="ALERT"

cat > "$OUTDIR/heartbeat.json" <<EOF
{"task": "weekly_watchdog", "ts": "$(date -u --iso-8601=seconds)", "status": "${STATUS}"}
EOF

echo "monitoring: $OUTDIR (status=${STATUS})"
exit 0
