#!/usr/bin/env bash
# 快管线 AB：SRB E1 (time_stop 分层) + E2 (L3 结构化退出) + E4 (加仓趋势健康度门)
# vs baseline (2026-04-22 rolling_sim: 20260422_212338)
#
# 用法：bash scripts/fast_ab_srb_e1_e2_e4.sh [TAG]
#   TAG 默认 "e1_e2_e4"
#
# 做什么：
#   1) 复用 20260422_212338/fast_month_$M/strategies_calibrated 作 baseline（threshold 已校准）
#   2) 再把当前 config/strategies/srb 覆盖到一份新的 calibrated snapshot，作为 treatment
#   3) 对两组都跑 event_backtest.py --fast，输出到 reports/srb_fast_ab_$TAG/{baseline,treatment}/$M/
#   4) 汇总 trades.csv → total_r / n_trades / add_n / add_r 摘要
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:-e1_e2_e4}"
BASELINE_RUN="results/srb/research_roll.features_on/_rolling_sim/20260422_212338"
OUT_ROOT="reports/srb_fast_ab_${TAG}"
mkdir -p "${OUT_ROOT}"

# 16 个月全窗口（2023-09 ~ 2024-12）— 与 slow rolling_sim 覆盖一致
MONTHS=(
  2023-09 2023-10 2023-11 2023-12
  2024-01 2024-02 2024-03 2024-04
  2024-05 2024-06 2024-07 2024-08
  2024-09 2024-10 2024-11 2024-12
)

echo "=== SRB fast-AB TAG=${TAG} ==="
echo "Months: ${MONTHS[*]}"
echo "Output: ${OUT_ROOT}"
echo

run_month () {
  local arm="$1"   # baseline | treatment
  local m="$2"
  local strat_root_src="${BASELINE_RUN}/fast_month_${m}/strategies_calibrated"
  local out_dir="${OUT_ROOT}/${arm}/${m}"

  if [[ ! -d "${strat_root_src}" ]]; then
    echo "[WARN] ${strat_root_src} 不存在，跳过 ${m}"
    return
  fi
  mkdir -p "${out_dir}"
  local strat_root_run="${out_dir}/strategies_calibrated"
  # 拷贝一份独立 snapshot
  rm -rf "${strat_root_run}"
  cp -r "${strat_root_src}" "${strat_root_run}"

  # 两臂都用 HEAD yaml 作底本，避免历史 activation_r/trail_r 差异污染 AB。
  # baseline 臂：关闭 E1/E2/E4 flags；treatment 臂：保持 HEAD 配置。
  cp config/strategies/srb/archetypes/execution.yaml \
     "${strat_root_run}/srb/archetypes/execution.yaml"

  if [[ "${arm}" == "baseline" ]]; then
    python - <<PYEOF
import yaml, pathlib
p = pathlib.Path("${strat_root_run}/srb/archetypes/execution.yaml")
d = yaml.safe_load(p.read_text())
# 关 E1：恢复无 time_stop 模式
d.setdefault("holding", {})
d["holding"]["max_holding_bars"] = None
d["holding"].pop("time_stop_uncap_mfe_r", None)
d["holding"]["time_stop_bars"] = 0
# 关 E2：L3 structural exit
d.setdefault("l3_structural_exit", {})["enabled"] = False
# 关 E4：trend_health_gate
pol = d.setdefault("srb_add_position_policy", {})
gate = pol.setdefault("post_hoc_shape_gate", {}).setdefault("trend_health_gate", {})
gate["enabled"] = False
p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
PYEOF
  fi

  local last_day
  last_day=$(python -c "import calendar,datetime; y,m=map(int,'${m}'.split('-'));print(calendar.monthrange(y,m)[1])")

  python scripts/event_backtest.py \
    --strategy srb \
    --start-date "${m}-01" \
    --end-date "${m}-${last_day}" \
    --strategies-root "${strat_root_run}" \
    --data-path data/parquet_data \
    --trades-csv "${out_dir}/trades.csv" \
    --output "${out_dir}/summary.json" \
    --fast \
    >"${out_dir}/run.log" 2>&1 &
}

# 并行两个 arm × N month。控制并发到 6 避免 OOM。
MAX_JOBS=6
JOBS=0
for arm in baseline treatment; do
  for m in "${MONTHS[@]}"; do
    run_month "${arm}" "${m}"
    JOBS=$((JOBS+1))
    if (( JOBS >= MAX_JOBS )); then
      wait -n
      JOBS=$((JOBS-1))
    fi
  done
done
wait

echo
echo "=== 汇总 ==="
python scripts/summarize_fast_ab_srb.py "${OUT_ROOT}"
