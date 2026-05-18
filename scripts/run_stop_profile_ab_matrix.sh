#!/usr/bin/env bash
# 生产档（bpc,tpc,me / constitution.yaml，当前 archetype 为 10R 宽止损）
# vs Swing 档（bpc_swing,tpc_swing,me_swing / constitution_abc_v1.yaml，4R 紧止损）
# 三窗：bear / bull / transition。产出独立 JSON 便于对比 per_archetype。
#
# Usage（仓库根目录）:
#   bash scripts/run_stop_profile_ab_matrix.sh [OUT_DIR]

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src

OUT="${1:-results/120T/stop_ab_matrix/$(date +%Y%m%d_%H%M)}"
mkdir -p "$OUT"
LOG="$OUT/run.log"
exec >>"$LOG" 2>&1
echo "OUT=$OUT started=$(date -Iseconds)"

run_prod() {
  local tag="$1" s="$2" e="$3"
  python scripts/event_backtest.py --strategy bpc,tpc,me \
    --start-date "$s" --end-date "$e" \
    --data-path data/parquet_data \
    --constitution-yaml config/constitution/constitution.yaml \
    --output "$OUT/prod_${tag}.json" \
    --trades-csv "$OUT/prod_${tag}_trades.csv" \
    --capital-report "$OUT" \
    --quiet-signal-logs
}

run_swing() {
  local tag="$1" s="$2" e="$3"
  python scripts/event_backtest.py --strategy bpc_swing,tpc_swing,me_swing \
    --start-date "$s" --end-date "$e" \
    --data-path data/parquet_data \
    --constitution-yaml config/constitution/constitution_abc_v1.yaml \
    --output "$OUT/swing_${tag}.json" \
    --trades-csv "$OUT/swing_${tag}_trades.csv" \
    --capital-report "$OUT" \
    --quiet-signal-logs
}

for strat in bear bull transition; do
  case "$strat" in
    bear) s=2022-01-01 e=2022-12-31 ;;
    bull) s=2023-01-01 e=2025-01-31 ;;
    transition) s=2025-02-01 e=2026-05-01 ;;
  esac
  tag="${strat}_${s}_${e}"
  echo ">>> PROD (10R archetypes) $tag"
  run_prod "$tag" "$s" "$e"
  echo ">>> SWING (4R archetypes) $tag"
  run_swing "$tag" "$s" "$e"
done

echo "DONE OUT=$OUT finished=$(date -Iseconds)"
