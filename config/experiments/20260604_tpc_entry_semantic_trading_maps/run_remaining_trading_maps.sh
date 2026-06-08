#!/usr/bin/env bash
# E2a / E1e2 / S50 only (skip E0_prod if already done).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src:scripts

SYMS="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT"
START="2022-01-01"
END="2026-04-01"
DATA="data/parquet_data"
BASE="results/tpc/maps/compare_entry_semantic_20260604"
LOG_DIR="${BASE}/logs"
mkdir -p "$LOG_DIR"

python scripts/research/prepare_tpc_entry_semantic_snapshots.py

run_one() {
  local tag="$1"
  local root="$2"
  local out="${BASE}/${tag}"
  mkdir -p "$out"
  echo ""
  echo "=== ${tag} $(date -Iseconds) ==="
  python -m scripts.event_backtest \
    --strategy tpc \
    --symbols "$SYMS" \
    --start-date "$START" \
    --end-date "$END" \
    --strategies-root "$root" \
    --data-path "$DATA" \
    --trades-csv "${out}/event_trades_tpc.csv" \
    --capital-report "$out" \
    --trading-map "${out}/trading_map_tpc_event.html" \
    --no-kill-switch \
    --quiet-signal-logs
}

exec >>"${LOG_DIR}/remaining.log" 2>&1
echo "remaining runs start $(date -Iseconds)"

run_one "E2a_or_anti_chase" "config_experiments/tpc_entry_e2a_or_anti_chase_strategies"
run_one "E1e2_band_or_anti" "config_experiments/tpc_entry_e1e2_band_or_anti_strategies"
run_one "S50_depth_gt50" "config_experiments/tpc_semantic_depth_gt50_strategies"

echo "remaining runs done $(date -Iseconds)"
