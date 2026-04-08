#!/usr/bin/env bash
# 顺序执行多条 turbo pipeline。
# 默认：BPC 极致回撤不破 lab（见 bpc如何回撤刚结束开仓.md §5）。
# 用法:
#   ./scripts/run_turbo_rolling_sim_batch.sh
#   STAGE=fast_month MONTH=2024-09 ./scripts/run_turbo_rolling_sim_batch.sh
#   CONFIGS="config/a.yaml config/b.yaml" ./scripts/run_turbo_rolling_sim_batch.sh
# BPC 主线 + ME 对比示例:
#   CONFIGS="config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_bpc_only.yaml config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_me_only.yaml" ./scripts/run_turbo_rolling_sim_batch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGE="${STAGE:-rolling_sim}"
SKIP_SHAP="${SKIP_SHAP:-1}"
MONTH="${MONTH:-}"

# 空格分隔的多个 --config 路径（相对仓库根）
if [[ -n "${CONFIGS:-}" ]]; then
  read -r -a CFG_ARR <<<"$CONFIGS"
else
  CFG_ARR=(
    "config/prod_train_pipeline_2h_turbo_2024bull_bpc_pullback_lab_extreme_pullback.yaml"
  )
fi

echo "=============================================="
echo "Turbo pipeline batch"
echo "  repo:    $ROOT"
echo "  stage:   $STAGE"
echo "  month:   ${MONTH:-"(rolling 全窗，未设 MONTH)"}"
echo "  configs: ${#CFG_ARR[@]} 个"
echo "=============================================="

i=0
for cfg in "${CFG_ARR[@]}"; do
  i=$((i + 1))
  if [[ ! -f "$cfg" ]]; then
    echo "[$i/${#CFG_ARR[@]}] SKIP: 找不到配置 $cfg"
    continue
  fi
  echo ""
  echo "----------------------------------------------"
  echo "[$i/${#CFG_ARR[@]}] mlbot pipeline run --all --config $cfg --stage $STAGE ..."
  echo "----------------------------------------------"
  cmd=(mlbot pipeline run --all --config "$cfg" --stage "$STAGE")
  [[ -n "$MONTH" ]] && cmd+=(--month "$MONTH")
  [[ "$SKIP_SHAP" == "1" || "$SKIP_SHAP" == "true" ]] && cmd+=(--skip-shap)
  "${cmd[@]}"
done

echo ""
echo "✅ 全部完成（共 ${#CFG_ARR[@]} 项）。"
