#!/usr/bin/env bash
# Full-window trading maps — edit VARIANTS after segment grid review.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src:scripts

SYMS="BTCUSDT,SOLUSDT"
START="2022-01-01"
END="2026-04-01"
DATA="data/parquet_data"
BASE="results/tpc/maps/macro_pullback_replace_20260610"
LOG_DIR="${BASE}/logs"
mkdir -p "$LOG_DIR"

VARIANTS=(
  "E0_prod|config/strategies"
  "M_replace_L15_S12|config_experiments/tpc_macro_replace_L15_S12_strategies"
  "M_replace_L20_S15|config_experiments/tpc_macro_replace_L20_S15_strategies"
)

run_one() {
  local tag="$1"
  local sroot="$2"
  local out="${BASE}/${tag}"
  mkdir -p "$out"
  echo "=== ${tag} $(date -Iseconds) ==="
  python -m scripts.event_backtest \
    --strategy tpc \
    --symbols "$SYMS" \
    --start-date "$START" \
    --end-date "$END" \
    --strategies-root "$sroot" \
    --data-path "$DATA" \
    --trades-csv "${out}/event_trades_tpc.csv" \
    --capital-report "$out" \
    --trading-map "${out}/trading_map_tpc_event.html" \
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
