#!/usr/bin/env bash
# 快管线 AB：SRB prefilter 加 fer_ols_pos 极端区间硬门（实验 G）。
#
# 假设：SRB 与 FBF strict 在同一 OLS(96) 通道上形成"极性相反"对偶。
#   FBF strict V2 prefilter: fer_ols_pos ≥ 0.75 / ≤ 0.25   （失败突破，反向）
#   SRB    V2 prefilter:     fer_ols_pos ≥ 0.90 / ≤ 0.10   （成功突破，顺向）
#
# diag_srb_l3_breach_vs_proximity.py 16 月首仓数据支持：
#   >0.90 ∪ <0.10 : n=110, sumR=+48.43R   (全集 n=143 sumR=+36.43R)
#   0.10-0.75     : n=25,  sumR=-12.25R   (主要负贡献源)
#
# 两臂：
#   baseline  : HEAD prefilter（3 条 soft prior，无 OLS 硬门）
#   treatment : HEAD prefilter + 新加 any_of(fer_ols_pos ≥0.90 OR ≤0.10)
#
# 两臂都用 HEAD execution.yaml（保留 E1+E2 的状态）。
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:-prefilter_ols}"
BASELINE_RUN="results/srb/slow-rolling-sim/_rolling_sim/20260422_212338"
OUT_ROOT="results/reports/srb_fast_ab_${TAG}"
mkdir -p "${OUT_ROOT}"

MONTHS=(
  2023-09 2023-10 2023-11 2023-12
  2024-01 2024-02 2024-03 2024-04
  2024-05 2024-06 2024-07 2024-08
  2024-09 2024-10 2024-11 2024-12
)

ARMS=(baseline treatment)

echo "=== SRB prefilter OLS hard-gate AB  TAG=${TAG} ==="
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

  # 两臂都覆盖当前 HEAD 的 execution.yaml / prefilter.yaml / features_prefilter.yaml
  # (snapshot 里的 yaml 是校准时的老版，我们要让 AB 基于 HEAD)
  cp config/strategies/srb/archetypes/execution.yaml \
     "${strat_root_run}/srb/archetypes/execution.yaml"
  cp config/strategies/srb/archetypes/prefilter.yaml \
     "${strat_root_run}/srb/archetypes/prefilter.yaml"
  cp config/strategies/srb/features_prefilter.yaml \
     "${strat_root_run}/srb/features_prefilter.yaml"

  if [[ "${arm}" == "treatment" ]]; then
    # 注入：prefilter 加 any_of(fer_ols_pos ≥0.90 OR ≤0.10)
    #      features_prefilter 加 fer_failure_signals_f（保证 fer_ols_pos 列可用）
    python - <<PYEOF
import yaml, pathlib

pf_path = pathlib.Path("${strat_root_run}/srb/archetypes/prefilter.yaml")
pf = yaml.safe_load(pf_path.read_text())
rules = pf.setdefault("rules", [])

new_rule = {
    "any_of": [
        {"feature": "fer_ols_pos", "operator": ">=", "value": 0.90},
        {"feature": "fer_ols_pos", "operator": "<=", "value": 0.10},
    ],
    "locked": True,
    "lock_reason": (
        "SRB 对偶 FBF strict（实验 G 2026-04-23）："
        "close 必须落在 OLS(96) 通道外极值，抓成功突破延续"
    ),
}
# 幂等：如果已经有这条规则就不重复添加
already = any(
    isinstance(r.get("any_of"), list)
    and any(
        (isinstance(x, dict) and x.get("feature") == "fer_ols_pos" and x.get("value") in (0.90, 0.10))
        for x in r["any_of"]
    )
    for r in rules
    if isinstance(r, dict)
)
if not already:
    rules.insert(0, new_rule)
pf_path.write_text(yaml.safe_dump(pf, sort_keys=False, allow_unicode=True))

# features_prefilter: 确保 fer_failure_signals_f 在列表里
fp_path = pathlib.Path("${strat_root_run}/srb/features_prefilter.yaml")
fp = yaml.safe_load(fp_path.read_text())
fpl = fp.setdefault("feature_pipeline", {})
req = fpl.setdefault("requested_features", [])
if "fer_failure_signals_f" not in req:
    req.append("fer_failure_signals_f")
fp_path.write_text(yaml.safe_dump(fp, sort_keys=False, allow_unicode=True))
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
