#!/usr/bin/env bash
# 可选：串联执行 2024 四季 slow_snapshot（expanding window，每季带 --end-date）。
# 运营向：每季结构只用到该季末日及之前数据，不偷看未来；配置里建议 start_date 足够早（如 2023-01-01）以满足样本与 holdout 切分。
# 注意：早季 + 严 prefilter 仍可能导致 Train 不足管线门槛而 gate 失败，需放宽规则或延后 end-date。
# 用法：
#   ./scripts/run_quarterly_slow_snapshot_2024.sh
#   ./scripts/run_quarterly_slow_snapshot_2024.sh config/prod_train_pipeline_2h_2024bull_quarterly.yaml
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-config/prod_train_pipeline_2h_2024bull_quarterly.yaml}"
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

# 与 prod_train_pipeline_2h.yaml 默认一致；若你改了 output.history_dir，可设环境变量 HISTORY_DIR
HISTORY_DIR="${HISTORY_DIR:-results/120T/prod_train_history}"
ROLLING_ROOT="$ROOT/$HISTORY_DIR/_rolling_sim"
REGISTRY="$ROLLING_ROOT/quarterly_slow_run_registry.txt"

echo "Using config: $CONFIG"
echo "History dir: $HISTORY_DIR (override with HISTORY_DIR=...)"
echo ""

mkdir -p "$ROLLING_ROOT"
: >"$REGISTRY.new"
for END in 2024-03-31 2024-06-30 2024-09-30 2024-12-31; do
  echo "================================================================================"
  echo "slow_snapshot  end-date=$END  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "================================================================================"
  MARKER="$(mktemp)"
  touch "$MARKER"
  mlbot pipeline run --all \
    --config "$CONFIG" \
    --end-date "$END" \
    --stage slow_snapshot \
    --skip-shap
  # 取本轮刚写入的 manifest（晚于 marker；若多个取最新 mtime）
  NEW_MAN=""
  if [[ -d "$ROLLING_ROOT" ]]; then
    NEW_MAN="$(find "$ROLLING_ROOT" -maxdepth 2 -name slow_snapshot_manifest.json -newer "$MARKER" -printf '%T@\t%p\n' 2>/dev/null | sort -n | tail -1 | cut -f2-)"
  fi
  rm -f "$MARKER"
  if [[ -n "$NEW_MAN" ]]; then
    RUN_ID="$(basename "$(dirname "$NEW_MAN")")"
    echo ""
    echo ">>> 本季 run_id=$RUN_ID"
    echo ">>> manifest=$NEW_MAN"
    echo "$END	$RUN_ID	$NEW_MAN" >>"$REGISTRY.new"
  else
    echo "" >&2
    echo ">>> 警告: 未检测到新的 slow_snapshot_manifest.json（请从终端日志里找 Snapshot Manifest 行）" >&2
  fi
  echo ""
done
mv -f "$REGISTRY.new" "$REGISTRY"

echo "Done."
echo "四季 run_id 已写入: $REGISTRY"
echo "每季一次运行会生成新的 run_id（目录名时间戳），互不覆盖："
echo "  - $HISTORY_DIR/<strategy>/<run_id>/"
echo "  - $HISTORY_DIR/_rolling_sim/<run_id>/slow_snapshot_manifest.json"
echo ""
echo "说明:"
echo "  - rolling.mode=slow_realistic 时，rolling_sim 会按 cadence 自动做季度慢变量快照并在月度循环中使用。"
echo "  - 本脚本仍可用于离线预生成季度 slow_snapshot（审计/回放/手工比对）。"
