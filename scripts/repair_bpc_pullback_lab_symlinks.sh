#!/usr/bin/env bash
# 重建 BPC「极致回撤不破」实验目录下的 symlink（根目录误用 ../../../ 时会断）。
# 策略根：config/strategies_pullback_lab_extreme_pullback/bpc
# 在仓库根执行: ./scripts/repair_bpc_pullback_lab_symlinks.sh
#
# 勿把 gate_draft.yaml symlink 到主线 strategies/bpc：turbo + disable_model_training 下
# optimize 读实验目录 gate_draft；若指向树模型草稿，hard_gates 会与 archetypes 脱钩。
# 本脚本从 archetypes/gate.yaml（解引用）复制为实体 gate_draft.yaml。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VARIANT="extreme_pullback"
BPC="config/strategies_pullback_lab_${VARIANT}/bpc"
mkdir -p "$BPC/archetypes"
[[ -d "$BPC" ]] || { echo "missing $BPC"; exit 1; }

# 本 lab 使用实体 direction.yaml / execution.yaml；若曾被链到主线，先拆掉链以免误写主线
for _lab_real in direction.yaml execution.yaml; do
  _p="$BPC/archetypes/$_lab_real"
  [[ -L "$_p" ]] && rm -f "$_p"
done

for f in config/strategies/bpc/*; do
  [ -f "$f" ] || continue
  name=$(basename "$f")
  [[ "$name" == gate_draft.yaml ]] && continue
  [[ "$name" == features_entry_filter.yaml ]] && continue
  ln -sfn "../../strategies/bpc/$name" "$BPC/$name"
done

for f in config/strategies/bpc/archetypes/*; do
  [ -f "$f" ] || continue
  name=$(basename "$f")
  [[ "$name" == entry_filters.yaml || "$name" == gate.yaml || "$name" == prefilter.yaml || "$name" == direction.yaml || "$name" == execution.yaml ]] && continue
  ln -sfn "../../../strategies/bpc/archetypes/$name" "$BPC/archetypes/$name"
done

rm -f "$BPC/gate_draft.yaml"
cp -L "$BPC/archetypes/gate.yaml" "$BPC/gate_draft.yaml"

echo "OK: repaired symlinks under $BPC"
