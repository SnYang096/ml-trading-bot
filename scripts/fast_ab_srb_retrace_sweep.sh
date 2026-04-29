#!/usr/bin/env bash
# 快管线 AB：SRB retrace_guard 阈值扫描（F 实验）。
# 三臂：baseline (retrace off) vs rg_050 (0.5) vs rg_070 (0.7) vs rg_085 (0.85)
# baseline 以 HEAD yaml 为底本并关掉 retrace_guard / E1 / E2；
# treatment 臂以 HEAD yaml 为底本，开启 retrace_guard（保持 E1+E2 状态）。
# 目的：判断 "current_r < pct × mfe_r 时拒绝加仓" 对 16 个月净 R 的影响，
# 并与诊断脚本 diag_srb_losing_adds.py 的 "add_est_current_r<0.2 +7.56R" 指引交叉验证。

set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:-retrace_sweep}"
BASELINE_RUN="results/srb/slow-rolling-sim/_rolling_sim/20260422_212338"
OUT_ROOT="results/reports/srb_fast_ab_${TAG}"
mkdir -p "${OUT_ROOT}"

MONTHS=(
  2023-09 2023-10 2023-11 2023-12
  2024-01 2024-02 2024-03 2024-04
  2024-05 2024-06 2024-07 2024-08
  2024-09 2024-10 2024-11 2024-12
)

ARMS=(baseline rg_050 rg_070 rg_085)

echo "=== SRB retrace sweep TAG=${TAG} ==="
echo "Months: ${MONTHS[*]}"
echo "Arms:   ${ARMS[*]}"
echo "Output: ${OUT_ROOT}"
echo

run_month () {
  local arm="$1"
  local m="$2"
  local strat_root_src="${BASELINE_RUN}/fast_month_${m}/strategies_calibrated"
  local out_dir="${OUT_ROOT}/${arm}/${m}"

  if [[ ! -d "${strat_root_src}" ]]; then
    echo "[WARN] ${strat_root_src} 不存在，跳过 ${m}"
    return
  fi
  mkdir -p "${out_dir}"
  local strat_root_run="${out_dir}/strategies_calibrated"
  rm -rf "${strat_root_run}"
  cp -r "${strat_root_src}" "${strat_root_run}"

  cp config/strategies/bad-candidates/srb/archetypes/execution.yaml \
     "${strat_root_run}/srb/archetypes/execution.yaml"

  python - <<PYEOF
import yaml, pathlib
p = pathlib.Path("${strat_root_run}/srb/archetypes/execution.yaml")
d = yaml.safe_load(p.read_text())
arm = "${arm}"
rg = d.setdefault("srb_add_position_policy", {}) \
      .setdefault("post_hoc_shape_gate", {}) \
      .setdefault("retrace_guard", {})
if arm == "baseline":
    rg["enabled"] = False
elif arm == "rg_050":
    rg["enabled"] = True
    rg["min_captured_pct"] = 0.50
elif arm == "rg_070":
    rg["enabled"] = True
    rg["min_captured_pct"] = 0.70
elif arm == "rg_085":
    rg["enabled"] = True
    rg["min_captured_pct"] = 0.85
p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
PYEOF

  local last_day
  last_day=$(python -c "import calendar,datetime; y,m=map(int,'${m}'.split('-'));print(calendar.monthrange(y,m)[1])")

  python scripts/event_backtest.py \
    --strategy srb \
    --start-date "${m}-01" \
    --end-date "${m}-${last_day}" \
    --strategies-root "${strat_root_run}" \
    --data-path data/parquet_data \
    --export "${out_dir}/trades.csv" \
    --output "${out_dir}/summary.json" \
    --fast \
    >"${out_dir}/run.log" 2>&1 &
}

MAX_JOBS=6
JOBS=0
for arm in "${ARMS[@]}"; do
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
python scripts/summarize_fast_ab_srb_multi.py "${OUT_ROOT}"
