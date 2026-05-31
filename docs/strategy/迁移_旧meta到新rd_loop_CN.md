# 迁移手册：旧 meta R&D → 新 explicit rd_loop + variant-grid

> 配套：[`方法论_R_and_D流程_CN.md`](方法论_R_and_D流程_CN.md) · [`R&D工具矩阵_CN.md`](R&D工具矩阵_CN.md) · [`研究工具重构计划_CN.md`](研究工具重构计划_CN.md) §14

## ME Entry（before → after）

**Before（pipeline / optimize_entry_filter_plateau）**
```bash
python scripts/optimize_entry_filter_plateau.py --strategy me ...
mlbot pipeline run -c config/pipelines/research_roll.features_on.yaml ...
```

**After（显式假设 + snotio plateau）**
```bash
mlbot research plateau --strategy me --layer entry --kpi snotio --snotio-mode entry_rr \
  --features-parquet results/.../features_labeled.parquet \
  --feature vpin_ignition --operator "<=" --grid "0.3,0.4,0.5,0.6"

python scripts/rd_loop.py --hypothesis-yaml config/experiments/me/rd_loop_me_entry_filter.yaml

mlbot research calibrate --from-plateau results/rd_loop/.../plateau.json \
  --output config/strategies/me/archetypes/entry_filters_draft.yaml

mlbot research promote --from .../entry_filters_draft.yaml \
  --to config/strategies/me/archetypes/entry_filters.yaml --layer entry --yes
```

## TPC Gate（before → after）

**Before**
```bash
python scripts/optimize_gate_unified.py --strategy tpc --logs results/.../features_labeled.parquet ...
```

**After（零调用 optimize_gate_unified）**
```bash
python scripts/rd_loop.py --hypothesis-yaml config/experiments/tpc/rd_loop_tpc_gate_plateau.yaml

mlbot research calibrate --from-plateau results/rd_loop/tpc_gate_plateau/quick_scan/gate_plateau/gate_plateau_batch.json \
  --strategy tpc --output config/strategies/tpc/archetypes/gate_draft.yaml

python -m scripts.event_backtest --variant-grid config/experiments/tpc/tpc_gate_refinement_grid.yaml

mlbot research promote --from .../gate_draft.yaml --to config/strategies/tpc/archetypes/gate.yaml --yes
```

## 阈值约定提醒

- rd_loop yaml 里 **q50/q90 分位** 只用于 condition-set **探测**；
- 生产阈值必须来自 **plateau/lift 平坦高原** → `calibrate` → 人审 → `promote`。

## 验收硬指标

ME + TPC 各完成一次 Gate/Entry refine 闭环，且 **未调用** `optimize_gate_unified.py` / unattended meta pipeline。
