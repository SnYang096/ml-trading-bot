#!/usr/bin/env bash
# Prod（tpc,me / constitution.yaml，10R archetypes）vs Swing ABC（tpc,me_swing… / constitution_abc_v1，4R）
# Archived slugs bpc、bpc_swing、tpc_swing 仍可经 bad-candidates 由 CLI 加载，脚本默认不再使用。
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
  python scripts/event_backtest.py --strategy tpc,me \
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
  python scripts/event_backtest.py --strategy tpc,me_swing \
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
  echo ">>> SWING ABC (4R archetypes) $tag"
  run_swing "$tag" "$s" "$e"
done

echo "DONE OUT=$OUT finished=$(date -Iseconds)"
