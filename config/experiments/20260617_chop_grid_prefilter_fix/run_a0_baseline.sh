#!/usr/bin/env bash
# A0: chop_grid 基线回测 — P0 prefilter fix 验证
#
# 使用 backtest_multileg_timeline.py --no-trend 测试 chop_grid 独立表现。
# P0 fix: _lookup 现在传递完整 box windowed 列（与 live 一致）。
set -euo pipefail
cd "$(dirname "$0")/../../.."

SEGMENTS=(
  "bear_2022:2022-01-01:2023-11-01"
  "bull_2023_2024:2023-06-01:2025-01-01"
  "recent_range_to_bear:2025-01-01:2026-05-31"
  "recent_6m_oos:2025-12-01:2026-05-31"
)

OUT_ROOT="results/chop_grid/experiments/prefilter_fix_20260617/a0_baseline"
PRELOAD="$OUT_ROOT/preload.pkl"
CHOP_CFG="config/experiments/20260617_chop_grid_prefilter_fix/variants/baseline/meta.yaml"
TREND_CFG="config/experiments/20260613_multileg_sizing_validate/variants/trend_prod/meta.yaml"
CONSTITUTION="live/highcap/config/constitution/constitution.yaml"
SYMBOLS="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"

mkdir -p "$OUT_ROOT"

FIRST=1
for seg in "${SEGMENTS[@]}"; do
  IFS=: read -r seg_id start end <<< "$seg"
  seg_out="$OUT_ROOT/$seg_id"
  mkdir -p "$seg_out"
  echo ""
  echo "=== A0 baseline: $seg_id ($start → $end) ==="

  PRELOAD_ARGS=""
  if [ "$FIRST" -eq 1 ] && [ ! -f "$PRELOAD" ]; then
    PRELOAD_ARGS="--save-preload $PRELOAD"
    FIRST=0
  elif [ -f "$PRELOAD" ]; then
    PRELOAD_ARGS="--load-preload $PRELOAD"
  fi

  # shellcheck disable=SC2086
  python scripts/backtest_multileg_timeline.py \
    --start "$start" \
    --end "$end" \
    --symbols "$SYMBOLS" \
    --equity 10000 \
    --chop-config "$CHOP_CFG" \
    --trend-config "$TREND_CFG" \
    --constitution-yaml "$CONSTITUTION" \
    --no-trend \
    --summary-json "$seg_out/summary.json" \
    $PRELOAD_ARGS
done

# Merge segment summaries
python -c "
import json, glob, os
root = '$OUT_ROOT'
segs = []
for d in sorted(glob.glob(os.path.join(root, '*', 'summary.json'))):
    s = json.loads(open(d).read())
    seg_id = os.path.basename(os.path.dirname(d))
    s['segment_id'] = seg_id
    segs.append(s)
merged = {'engine': 'backtest_multileg_timeline', 'experiment_id': 'chop_grid_prefilter_fix_a0_baseline_20260617', 'segments': segs}
out = os.path.join(root, 'joint', 'summary.json')
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(merged, open(out, 'w'), indent=2)
print(f'Merged {len(segs)} segments → {out}')
"

echo ""
echo "=== Done. Results in $OUT_ROOT ==="
