#!/usr/bin/env bash
# BPC / ME / TPC × turbo / slow / non_rolling，分阶段并行：
#   turbo / slow → --stage rolling_sim；non_rolling → 默认 full
#   阶段 1 — 三个策略同时跑 turbo
#   阶段 2 — 三个策略同时跑 slow
#   阶段 3 — 三个策略同时跑 non_rolling
# 每条独立日志；同一阶段内最多 3 个 python 并行（约等于 3×CPU/IO）。
# 用法：在项目根目录 ./scripts/run_nine_strategy_research.sh
# 环境：需已配置 mlbot / 数据；耗时极长。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/src"
export PYTHONUNBUFFERED=1
LOGDIR="${ROOT}/results/logs/batch_full_research_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
SUMMARY="${LOGDIR}/summary.txt"
{
  echo "log_dir=${LOGDIR}"
  echo "started=$(date -Is)"
  echo "mode=parallel_by_profile (turbo -> slow -> non_rolling; 3 strategies per phase)"
} | tee "$SUMMARY"

run_parallel_phase() {
  local profile=$1
  local -a strat=(bpc me tpc)
  local -a pids=()
  local -a stage_args=()
  if [[ "$profile" == "turbo" || "$profile" == "slow" ]]; then
    stage_args=(--stage rolling_sim)
  fi
  echo "===== PHASE START $(date -Is) profile=${profile} strategies=${strat[*]} =====" | tee -a "$SUMMARY"
  for s in "${strat[@]}"; do
    local tag="${s}_${profile}"
    local log="${LOGDIR}/${tag}.log"
    {
      echo "===== START $(date -Is) ${tag} ====="
      echo "cmd: python scripts/auto_research_pipeline.py --strategy ${s} --config config/strategies/${s}/research/${profile}.yaml ${stage_args[*]} --no-adopt"
    } | tee -a "$SUMMARY"
    rm -f "$log"
    (
      echo "===== START $(date -Is) ${tag} (worker) =====" >"$log"
      python scripts/auto_research_pipeline.py \
        --strategy "$s" \
        --config "config/strategies/${s}/research/${profile}.yaml" \
        "${stage_args[@]}" \
        --no-adopt 2>&1 | tee -a "$log"
      exit "${PIPESTATUS[0]}"
    ) &
    pids+=($!)
  done
  local i=0
  for s in "${strat[@]}"; do
    local tag="${s}_${profile}"
    set +e
    wait "${pids[$i]}"
    local ec=$?
    set -u
    if [[ "$ec" -eq 0 ]]; then
      echo "===== OK $(date -Is) ${tag} =====" | tee -a "$SUMMARY"
    else
      echo "===== FAIL $(date -Is) ${tag} exit=${ec} =====" | tee -a "$SUMMARY"
    fi
    i=$((i + 1))
  done
  echo "===== PHASE END $(date -Is) profile=${profile} =====" | tee -a "$SUMMARY"
}

for profile in turbo slow non_rolling; do
  run_parallel_phase "$profile"
done

echo "finished=$(date -Is)" | tee -a "$SUMMARY"
