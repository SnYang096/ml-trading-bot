#!/usr/bin/env bash
# 顺序跑 BPC / ME / TPC × turbo / slow / non_rolling（默认 full 管线），每条写独立日志 + summary。
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
} | tee "$SUMMARY"
for s in bpc me tpc; do
  for p in turbo slow non_rolling; do
    tag="${s}_${p}"
    log="${LOGDIR}/${tag}.log"
    {
      echo "===== START $(date -Is) ${tag} ====="
      echo "cmd: python scripts/auto_research_pipeline.py --strategy ${s} --config config/strategies/${s}/research/${p}.yaml --no-adopt"
    } | tee -a "$SUMMARY"
    set +e
    python scripts/auto_research_pipeline.py \
      --strategy "$s" \
      --config "config/strategies/${s}/research/${p}.yaml" \
      --no-adopt 2>&1 | tee "$log"
    ec=${PIPESTATUS[0]}
    set -u
    if [[ "$ec" -eq 0 ]]; then
      echo "===== OK $(date -Is) ${tag} =====" | tee -a "$SUMMARY"
    else
      echo "===== FAIL $(date -Is) ${tag} exit=${ec} =====" | tee -a "$SUMMARY"
    fi
  done
done
echo "finished=$(date -Is)" | tee -a "$SUMMARY"
