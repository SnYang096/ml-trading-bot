#!/usr/bin/env bash
# 多策略研究回测 + 交易地图（K 线 + 成交标记）
#
# 用法:
#   bash scripts/run_multileg_backtest_with_maps.sh              # 默认 both
#   bash scripts/run_multileg_backtest_with_maps.sh chop
#   bash scripts/run_multileg_backtest_with_maps.sh trend
#   bash scripts/run_multileg_backtest_with_maps.sh both
#   bash scripts/run_multileg_backtest_with_maps.sh srb
#
# 修改下方「可配置区」即可换时间区间 / 币种 / 是否打开浏览器。
#
# 查看说明:
#   - backtest_multileg_timeline → 只有 summary.json（数字），无 K 线图
#   - diagnose_* / chop_grid_backtest → K 线 + 成交标记 + capital_report.html
#   - trend_scalp 成交极密（上万笔），全窗口单币图会糊成一团 → 用 capital_report
#     看多币对比，用 trading_map_continuous.html 分 panel，或框选 zoom 看局部

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# ── 可配置区（改这里）────────────────────────────────────────
START_DATE="${START_DATE:-2024-01-01}"
END_DATE="${END_DATE:-2026-05-31}"
SYMBOLS="${SYMBOLS:-BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT}"
# 交易地图只画这些币（默认与 SYMBOLS 相同；可设为 BTCUSDT 加快出图）
MAP_SYMBOLS="${MAP_SYMBOLS:-$SYMBOLS}"
EQUITY="${EQUITY:-10000}"
DATA_DIR="${DATA_DIR:-data/parquet_data}"
# chop / trend / both / srb
STRATEGY="${1:-${STRATEGY:-both}}"
# 是否额外跑 live 引擎 timeline（summary.json，无 K 线图）
RUN_TIMELINE="${RUN_TIMELINE:-1}"
# 回测结束后 macOS 自动 open 交易地图
OPEN_RESULTS="${OPEN_RESULTS:-1}"
# ───────────────────────────────────────────────────────────

TAG="${START_DATE}_${END_DATE}"
CHOP_PROD="config/experiments/20260613_multileg_sizing_validate/variants/chop_prod/meta.yaml"
TREND_PROD="config/experiments/20260613_multileg_sizing_validate/variants/trend_prod/meta.yaml"
CONSTITUTION="live/highcap/config/constitution/constitution.yaml"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
if [[ -f "$ROOT/scripts/env_macos_blas.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/env_macos_blas.sh"
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

open_if_exists() {
  local path="$1"
  if [[ "$OPEN_RESULTS" == "1" && -f "$path" ]]; then
    echo "  → open $path"
    open "$path"
  fi
}

open_dir_if_exists() {
  local path="$1"
  if [[ "$OPEN_RESULTS" == "1" && -d "$path" ]]; then
    echo "  → open $path"
    open "$path"
  fi
}

# 多币对比表 + per_symbol_summary.csv
print_per_symbol_summary() {
  local out="$1"
  local trades_csv="$2"
  local label="$3"
  if [[ ! -f "$trades_csv" ]]; then
    return 0
  fi
  python - "$trades_csv" "$out/per_symbol_summary.csv" "$label" <<'PY'
import sys
from pathlib import Path

import pandas as pd

trades_path, out_csv, label = sys.argv[1:4]
trades = pd.read_csv(trades_path)
if trades.empty or "symbol" not in trades.columns:
    print(f"\n=== {label} per-symbol: (no trades) ===")
    raise SystemExit(0)

pc = pd.to_numeric(trades.get("pnl_per_capital"), errors="coerce").fillna(0.0)
trades = trades.assign(_pc=pc)
g = trades.groupby("symbol", sort=True)
rows = []
for sym, df in g:
    seg_col = df.get("segment_id")
    n_seg = int(seg_col.nunique()) if seg_col is not None else 0
    rows.append(
        {
            "symbol": sym,
            "trades": len(df),
            "segments": n_seg,
            "sum_pnl_per_capital": float(df["_pc"].sum()),
            "return_pct": float(df["_pc"].sum() * 100.0),
            "win_rate": float((df["_pc"] > 0).mean()) if len(df) else 0.0,
        }
    )
summary = pd.DataFrame(rows).sort_values("return_pct", ascending=False)
summary.to_csv(out_csv, index=False)
print(f"\n=== {label} 多币对比 (return_pct = 该币 capital bucket 累计) ===")
print(summary.to_string(index=False, formatters={"return_pct": "{:.2f}%".format, "win_rate": "{:.1%}".format}))
print(f"\n  已写入: {out_csv}")
PY
}

open_multileg_results() {
  local out="$1"
  local trades_csv="$2"
  local label="$3"
  local single_map="$4"

  print_per_symbol_summary "$out" "$trades_csv" "$label"

  # 优先：一体化 continuous map（顶部权益曲线 + 分币对比表 + 分 panel K 线）
  open_if_exists "$out/trading_map_continuous.html"
  # 详细资金报告（表格 + SVG 曲线）
  open_if_exists "$out/capital_report.html"
  open_if_exists "$out/report.html"
  # 仅单币时才自动打开「满屏三角」单图，避免 5 币全窗口糊图
  if [[ "$MAP_SYMBOLS" != *","* ]]; then
    open_if_exists "$single_map"
  else
    echo "  (跳过单币 trading_map：多币请用 capital_report + continuous map；"
    echo "   要看单币可设 MAP_SYMBOLS=BTCUSDT 或框选 zoom)"
  fi
  open_dir_if_exists "$out"
}

run_chop_timeline() {
  local out="$ROOT/results/chop_grid/backtest_${TAG}"
  mkdir -p "$out"
  echo ""
  echo "=== chop_grid timeline (live engine, summary only) ==="
  local preload_args=()
  if [[ -f "$out/preload.pkl" ]]; then
    preload_args=(--load-preload "$out/preload.pkl")
  else
    preload_args=(--save-preload "$out/preload.pkl")
  fi
  python scripts/backtest_multileg_timeline.py \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --symbols "$SYMBOLS" \
    --data-dir "$DATA_DIR" \
    --equity "$EQUITY" \
    --chop-config "$CHOP_PROD" \
    --trend-config "$TREND_PROD" \
    --constitution-yaml "$CONSTITUTION" \
    --no-trend \
    --summary-json "$out/summary.json" \
    "${preload_args[@]}"
}

run_chop_maps() {
  local out="$ROOT/results/chop_grid/diagnose_${TAG}"
  mkdir -p "$out"
  echo ""
  echo "=== chop_grid diagnose (K 线 + trading map) ==="
  python scripts/chop_grid_backtest.py \
    --config config/strategies/chop_grid/meta.yaml \
    --data-dir "$DATA_DIR" \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --symbols "$SYMBOLS" \
    --map-symbols "$MAP_SYMBOLS" \
    --continuous-map-symbols "$MAP_SYMBOLS" \
    --out-dir "$out"
  echo ""
  echo "chop_grid 产出: $out"
  local first_sym="${MAP_SYMBOLS%%,*}"
  open_multileg_results "$out" "$out/grid_trades.csv" "chop_grid" \
    "$out/trading_map_grid_${first_sym}.html"
}

run_trend_timeline() {
  local out="$ROOT/results/trend_scalp/backtest_${TAG}"
  mkdir -p "$out"
  echo ""
  echo "=== trend_scalp timeline (live engine, summary only) ==="
  local preload_args=()
  if [[ -f "$out/preload.pkl" ]]; then
    preload_args=(--load-preload "$out/preload.pkl")
  else
    preload_args=(--save-preload "$out/preload.pkl")
  fi
  python scripts/backtest_multileg_timeline.py \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --symbols "$SYMBOLS" \
    --data-dir "$DATA_DIR" \
    --equity "$EQUITY" \
    --chop-config "$CHOP_PROD" \
    --trend-config "$TREND_PROD" \
    --constitution-yaml "$CONSTITUTION" \
    --no-chop \
    --summary-json "$out/summary.json" \
    "${preload_args[@]}"
}

run_trend_maps() {
  local out="$ROOT/results/trend_scalp/diagnose_${TAG}"
  mkdir -p "$out"
  echo ""
  echo "=== trend_scalp diagnose (K 线 + trading map) ==="
  python scripts/diagnose_dual_add_trend.py \
    --config config/strategies/trend_scalp \
    --data-dir "$DATA_DIR" \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --symbols "$SYMBOLS" \
    --map-symbols "$MAP_SYMBOLS" \
    --continuous-map-symbols "$MAP_SYMBOLS" \
    --timeframe 2h \
    --execution-timeframe 1min \
    --no-initial-hedge \
    --out-dir "$out"
  echo ""
  echo "trend_scalp 产出: $out"
  local first_sym="${MAP_SYMBOLS%%,*}"
  open_multileg_results "$out" "$out/dual_add_trades.csv" "trend_scalp" \
    "$out/trading_map_dual_add_${first_sym}.html"
}

run_srb_event_backtest() {
  local out="$ROOT/results/event_backtest/srb_${TAG}"
  mkdir -p "$out"
  echo ""
  echo "=== SRB event_backtest (K 线 + trading map) ==="
  # SRB 常用单币；多币请改 SYMBOLS 并注意 runtime
  local srb_symbols="${SRB_SYMBOLS:-BTCUSDT}"
  python scripts/event_backtest.py \
    --strategy srb \
    --symbols "$srb_symbols" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --data-path "$DATA_DIR" \
    --strategies-root config/strategies \
    --fast \
    --output "$out/event_backtest_srb.json" \
    --trading-map "$out/trading_map_srb.html" \
    --trades-csv "$out/event_trades_srb.csv"
  echo ""
  echo "SRB 产出: $out"
  open_if_exists "$out/trading_map_srb.html"
  open_dir_if_exists "$out"
}

echo "================================================================"
echo " 研究回测 + 交易地图"
echo " 策略:     $STRATEGY"
echo " 区间:     $START_DATE → $END_DATE"
echo " 币种:     $SYMBOLS"
echo " 地图币种: $MAP_SYMBOLS"
echo "================================================================"

case "$STRATEGY" in
  chop)
    [[ "$RUN_TIMELINE" == "1" ]] && run_chop_timeline
    run_chop_maps
    ;;
  trend)
    [[ "$RUN_TIMELINE" == "1" ]] && run_trend_timeline
    run_trend_maps
    ;;
  both)
    [[ "$RUN_TIMELINE" == "1" ]] && run_chop_timeline
    run_chop_maps
    [[ "$RUN_TIMELINE" == "1" ]] && run_trend_timeline
    run_trend_maps
    ;;
  srb)
    run_srb_event_backtest
    ;;
  *)
    echo "未知策略: $STRATEGY （可选: chop | trend | both | srb）" >&2
    exit 1
    ;;
esac

echo ""
echo "✅ 全部完成"
