#!/usr/bin/env bash
# A1/A1b/A1_combined: chop_grid Phase 2 variant backtests
#
# A1: box_stability_240 < 0.4 (IC@1=-0.04, low stability → better RR)
# A1b: box_pos_480 ∈ [0.3, 0.7] (IC@1=+0.027)
# A1_combined: both rules
#
# Usage:
#   bash config/experiments/20260617_chop_grid_prefilter_fix/run_a1_variants.sh [a1|a1b|a1_combined|all]
set -euo pipefail
cd "$(dirname "$0")/../../.."

VARIANT="${1:-all}"
EXP_DIR="config/experiments/20260617_chop_grid_prefilter_fix"

SEGMENTS=(
  "bear_2022:2022-01-01:2023-11-01"
  "bull_2023_2024:2023-06-01:2025-01-01"
  "recent_range_to_bear:2025-01-01:2026-05-31"
  "recent_6m_oos:2025-12-01:2026-05-31"
)

TREND_CFG="config/experiments/20260613_multileg_sizing_validate/variants/trend_prod/meta.yaml"
CONSTITUTION="live/highcap/config/constitution/constitution.yaml"
SYMBOLS="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"

run_variant() {
  local variant_name="$1"
  local chop_cfg="$EXP_DIR/variants/$variant_name/meta.yaml"
  local out_root="results/chop_grid/experiments/prefilter_fix_20260617/$variant_name"
  local preload="$out_root/preload.pkl"

  mkdir -p "$out_root"

  # Reuse A0 preload if exists
  local a0_preload="results/chop_grid/experiments/prefilter_fix_20260617/a0_baseline/preload.pkl"

  FIRST=1
  for seg in "${SEGMENTS[@]}"; do
    IFS=: read -r seg_id start end <<< "$seg"
    seg_out="$out_root/$seg_id"
    mkdir -p "$seg_out"
    echo ""
    echo "=== $variant_name: $seg_id ($start → $end) ==="

    PRELOAD_ARGS=""
    if [ "$FIRST" -eq 1 ] && [ -f "$a0_preload" ]; then
      PRELOAD_ARGS="--load-preload $a0_preload"
      FIRST=0
    elif [ -f "$preload" ]; then
      PRELOAD_ARGS="--load-preload $preload"
    fi

    # shellcheck disable=SC2086
    python scripts/backtest_multileg_timeline.py \
      --start "$start" \
      --end "$end" \
      --symbols "$SYMBOLS" \
      --equity 10000 \
      --chop-config "$chop_cfg" \
      --trend-config "$TREND_CFG" \
      --constitution-yaml "$CONSTITUTION" \
      --no-trend \
      --summary-json "$seg_out/summary.json" \
      $PRELOAD_ARGS
  done

  # Merge segment summaries
  python -c "
import json, glob, os
root = '$out_root'
segs = []
for d in sorted(glob.glob(os.path.join(root, '*', 'summary.json'))):
    s = json.loads(open(d).read())
    seg_id = os.path.basename(os.path.dirname(d))
    s['segment_id'] = seg_id
    segs.append(s)
merged = {'engine': 'backtest_multileg_timeline', 'experiment_id': '${variant_name}_20260617', 'segments': segs}
out = os.path.join(root, 'joint', 'summary.json')
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(merged, open(out, 'w'), indent=2)
print(f'Merged {len(segs)} segments → {out}')
"
  echo ""
  echo "=== $variant_name done. Results in $out_root ==="
}

case "$VARIANT" in
  a1)
    run_variant a1_low_stability
    ;;
  a1b)
    run_variant a1b_pos480_range
    ;;
  a1_combined|combined)
    run_variant a1_combined
    ;;
  all)
    run_variant a1_low_stability
    run_variant a1b_pos480_range
    run_variant a1_combined
    ;;
  *)
    echo "Usage: $0 [a1|a1b|a1_combined|all]"
    exit 1
    ;;
esac

echo ""
echo "=== All requested variants done ==="
echo "Compare results:"
echo "  A0 baseline:  results/chop_grid/experiments/prefilter_fix_20260617/a0_baseline/joint/summary.json"
echo "  A1 low stab:  results/chop_grid/experiments/prefilter_fix_20260617/a1_low_stability/joint/summary.json"
echo "  A1b pos480:   results/chop_grid/experiments/prefilter_fix_20260617/a1b_pos480_range/joint/summary.json"
echo "  A1 combined:  results/chop_grid/experiments/prefilter_fix_20260617/a1_combined/joint/summary.json"
