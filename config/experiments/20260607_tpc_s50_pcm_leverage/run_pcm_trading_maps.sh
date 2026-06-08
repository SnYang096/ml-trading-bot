#!/usr/bin/env bash
# PCM 联合交易地图（需完整回测 + --trading-map）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src:scripts

SYMS="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT"
START="2022-01-01"
END="2026-04-01"
DATA="data/parquet_data"
BASE="results/pcm/maps/s50_pcm_20260607"
LOG_DIR="${BASE}/logs"
mkdir -p "$LOG_DIR"

python scripts/research/prepare_tpc_s50_pcm_leverage_experiments.py

run_one() {
  local tag="$1"
  local root="$2"
  local const="$3"
  local out="${BASE}/${tag}"
  mkdir -p "$out"
  echo ""
  echo "=== ${tag} $(date -Iseconds) ==="
  python -m scripts.event_backtest \
    --strategy bpc,tpc \
    --symbols "$SYMS" \
    --start-date "$START" \
    --end-date "$END" \
    --strategies-root "$root" \
    --data-path "$DATA" \
    --constitution-yaml "$const" \
    --trades-csv "${out}/event_trades_bpc,tpc.csv" \
    --capital-report "$out" \
    --trading-map "${out}/trading_map_bpc_tpc_event.html" \
    --no-kill-switch \
    --quiet-signal-logs
}

exec >>"${LOG_DIR}/pcm_maps.log" 2>&1
echo "pcm trading maps start $(date -Iseconds)"

run_one "pcm_s50_tpc_heavy" \
  "config_experiments/tpc_s50_bpc_pcm_strategies" \
  "config/experiments/20260607_tpc_s50_pcm_leverage/constitution/pcm_tpc_heavy.yaml"

run_one "pcm_prod_baseline" \
  "config/strategies" \
  "config/experiments/20260607_tpc_s50_pcm_leverage/constitution/pcm_equal.yaml"

echo "pcm trading maps done $(date -Iseconds)"
