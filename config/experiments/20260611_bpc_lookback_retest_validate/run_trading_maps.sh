#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src:scripts

SYMS="BTCUSDT,SOLUSDT"
START="2022-01-01"
END="2026-04-01"
DATA="data/parquet_data"
BASE="results/bpc/maps/lookback_retest_20260611"
LOG_DIR="${BASE}/logs"
mkdir -p "$LOG_DIR"

VARIANTS=(
  "B0_prod|config/strategies"
  "B_L120_retest|config_experiments/bpc_lb120_retest_strategies"
  "B_L120|config_experiments/bpc_lb120_strategies"
)

run_one() {
  local tag="$1"
  local sroot="$2"
  local out="${BASE}/${tag}"
  mkdir -p "$out"
  echo "=== ${tag} $(date -Iseconds) ==="
  python -m scripts.event_backtest \
    --strategy bpc \
    --symbols "$SYMS" \
    --start-date "$START" \
    --end-date "$END" \
    --strategies-root "$sroot" \
    --data-path "$DATA" \
    --trades-csv "${out}/event_trades_bpc.csv" \
    --capital-report "$out" \
    --trading-map "${out}/trading_map_bpc_event.html" \
    --no-kill-switch \
    --quiet-signal-logs
}

exec >>"${LOG_DIR}/maps.log" 2>&1
echo "trading maps start $(date -Iseconds)"
for row in "${VARIANTS[@]}"; do
  IFS='|' read -r tag sroot <<< "$row"
  run_one "$tag" "$sroot"
done
echo "trading maps done $(date -Iseconds)"
