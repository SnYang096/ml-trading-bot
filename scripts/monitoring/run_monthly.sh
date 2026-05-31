#!/usr/bin/env bash
# Monthly drift: placeholder calls calibrate_roll or fixed event_backtest (configure locally).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
RUN_TS="$(date -u +%Y%m%d_%H%M)"
OUTDIR="results/monitoring/monthly_drift/${RUN_TS}"
mkdir -p "$OUTDIR"

STATUS="OK"
if [[ "${MONTHLY_SKIP_PIPELINE:-0}" != "1" ]]; then
  mlbot pipeline run --all \
    --config config/strategies/bpc/research/calibrate_roll.default.yaml \
    --stage rolling_sim --skip-shap \
    || STATUS="ALERT"
fi

cat > "$OUTDIR/heartbeat.json" <<EOF
{"task": "monthly_drift", "ts": "$(date -u --iso-8601=seconds)", "status": "${STATUS}"}
EOF

echo "monthly monitoring: $OUTDIR (status=${STATUS})"
