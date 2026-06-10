#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src:scripts

SYMS="BTCUSDT,SOLUSDT"
START="2022-01-01"
END="2026-04-01"
DATA="data/parquet_data"
BASE="results/srb/maps/sr_tf_compare_20260609"
LOG_DIR="${BASE}/logs"
mkdir -p "$LOG_DIR"

# 变体列表（Phase 3 跑完后更新路径）
# 格式: TAG|strategies_root
VARIANTS=(
  "A_prod|config/strategies"
  # "B_l2_only|config_experiments/srb_l2_only_strategies"
  # "C_l3_relaxed|config_experiments/srb_l3_relaxed_strategies"
)

run_one() {
  local tag="$1"
  local sroot="$2"
  local out="${BASE}/${tag}"
  mkdir -p "$out"
  echo "=== ${tag} $(date -Iseconds) ==="
  python -m scripts.event_backtest \
    --strategy srb \
    --symbols "$SYMS" \
    --start-date "$START" \
    --end-date "$END" \
    --strategies-root "$sroot" \
    --data-path "$DATA" \
    --trades-csv "${out}/event_trades_srb.csv" \
    --capital-report "$out" \
    --trading-map "${out}/trading_map_srb_event.html" \
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