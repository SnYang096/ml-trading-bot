#!/usr/bin/env bash
# 快 AB：SRB 两段式首仓（cross 2a + EMA1200 2b）arm 门控 vs 基线（默认关）。
#
# baseline   : HEAD execution.yaml（srb_staged_entry_2b.enabled: false）
# treatment  : 同上但 enabled: true
#
# 依赖与 scripts/fast_ab_srb_prefilter_ols.sh 相同的 rolling_sim 快照目录。
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:-staged_2b}"
BASELINE_RUN="${BASELINE_RUN:-results/srb/slow-rolling-sim/_rolling_sim/20260422_212338}"
OUT_ROOT="results/reports/srb_fast_ab_${TAG}"
mkdir -p "${OUT_ROOT}"

MONTHS=(
  2023-09 2023-10 2023-11 2023-12
  2024-01 2024-02 2024-03 2024-04
  2024-05 2024-06 2024-07 2024-08
  2024-09 2024-10 2024-11 2024-12
)

ARMS=(baseline treatment)

echo "=== SRB staged 2b arm gate AB  TAG=${TAG} ==="
echo "BASELINE_RUN=${BASELINE_RUN}"
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

  if [[ "${arm}" == "treatment" ]]; then
    python - <<PYEOF
import pathlib
import yaml

ex_path = pathlib.Path("${strat_root_run}/srb/archetypes/execution.yaml")
ex = yaml.safe_load(ex_path.read_text())
st = ex.setdefault("srb_staged_entry_2b", {})
st["enabled"] = True
ex_path.write_text(yaml.safe_dump(ex, sort_keys=False, allow_unicode=True))
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
