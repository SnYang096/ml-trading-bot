#!/usr/bin/env bash
# ME vs me_swing — same bear/bull/transition/full windows as spot_a_split research matrix.
#
# Constitution: ``me`` uses production trend pool (constitution.yaml);
# ``me_swing`` uses ABC v1 resource slots (constitution_abc_v1.yaml).
#
# Usage (repo root):
#   bash scripts/run_me_vs_me_swing_matrix.sh [OUT_DIR]

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src

OUT="${1:-results/120T/me_matrix/$(date +%Y%m%d_%H%M)}"
mkdir -p "$OUT"
LOG="$OUT/matrix_run.log"
exec >>"$LOG" 2>&1
echo "OUT=$OUT  started=$(date -Iseconds)"

run_me() {
  local tag="$1" s="$2" e="$3"
  python scripts/event_backtest.py --strategy me --start-date "$s" --end-date "$e" \
    --data-path data/parquet_data --constitution-yaml config/constitution/constitution.yaml \
    --output "$OUT/me_${tag}.json" --trades-csv "$OUT/me_${tag}_trades.csv" \
    --capital-report "$OUT" --quiet-signal-logs
}

run_swing() {
  local tag="$1" s="$2" e="$3"
  python scripts/event_backtest.py --strategy me_swing --start-date "$s" --end-date "$e" \
    --data-path data/parquet_data --constitution-yaml config/constitution/constitution_abc_v1.yaml \
    --output "$OUT/me_swing_${tag}.json" --trades-csv "$OUT/me_swing_${tag}_trades.csv" \
    --capital-report "$OUT" --quiet-signal-logs
}

for strat in bear bull transition full; do
  case "$strat" in
    bear) s=2022-01-01 e=2022-12-31 ;;
    bull) s=2023-01-01 e=2025-01-31 ;;
    transition) s=2025-02-01 e=2026-05-01 ;;
    full) s=2022-01-01 e=2026-05-01 ;;
  esac
  tag="${strat}_${s}_${e}"
  echo ">>> me $tag"
  run_me "$tag" "$s" "$e"
  echo ">>> me_swing $tag"
  run_swing "$tag" "$s" "$e"
done

echo "DONE OUT=$OUT  finished=$(date -Iseconds)"
