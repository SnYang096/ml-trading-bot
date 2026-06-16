# T5α wall entry v2 — OR tier + regime 对齐

**背景**：v1（`20260615_t5_wall_entry_validate`）把 `combination_mode` 改成 **AND**，笔数暴跌、Σ R 全面低于 prod。**v2 修正实验设计**，不重复 v1 错误。

## v1 失败根因

| 问题 | v1 做法 | v2 修正 |
|------|---------|---------|
| 组合语义 | `and`：vol **且** wall | 保持 prod **`or`**，wall 为新增可选 tier |
| 与 scan 不对齐 | 全样本 AND wall | regime 条件写进 **单个 filter 的 conditions**（S5 子集） |
| 过严 | W2/W4 bear 段 0 笔 | 不全局 AND；熊/牛用 `direction` + ema 门槛 |

## 变体（Phase 3）

| ID | entry 语义 | 假设 |
|----|------------|------|
| **E0_prod** | prod baseline | 对照 |
| **W5_or_long2** | OR + 多单 `wall≤2`（无 regime） | 墙本身是否提供 **额外** 入场路径 |
| **W6_or_bull_pullback** | OR + 牛 S5（ema≥0.10 ∧ chop ∧ depth ∧ wall≤2）long only | 对齐 Phase 1c **最强** label lift 子集 |
| **W7_or_regime_asym** | W6 + 熊 S5 short（ema≤-0.10 ∧ wall≤2.5） | 熊/牛 asym，仍 OR 并联 |

prod 原有 `tpc_deep_pullback_vol_confirm` **不变**；新 tier 与之 **OR**，不会把 vol 路径关掉。

## 执行顺序（勿跳步）

### Track A — 直接 Phase 3（可立刻跑）

不依赖 ema parquet 修复；event_backtest 全序列算特征，`ema_1200_position` 在回测链路可用。

```bash
PYTHONPATH=src:scripts python scripts/research/prepare_t5_wall_entry_v2_snapshots.py

PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260616_t5_wall_entry_v2_validate/t5_wall_entry_v2_grid.yaml \
  --quiet-signal-logs
```

### Track B — Phase 1 复扫（可选，与 A 并行）

`ema_1200_position` 在 **prepare-only parquet** 曾因月度分块 warmup 不足全零；scan 子集结论需 v3 parquet 复核。

1. **Phase 0**：`ema_1200_value_f` 设 `pass_full_df: true`（或 `monthly_warmup_months≥6`）后重跑 prepare-only → `t5_btc_eth_v3`
2. **Phase 1d**：`rd_loop_t5_phase1d_wall_or_align.yaml`（S5 子集 + wall plateau）
3. 若 plateau 与 v1c 一致 → Track A 结果更可信；若漂移 → 更新 τ 再跑 grid

## 晋升标准

同 [`LAYER_PROMOTION_CRITERIA.md`](../LAYER_PROMOTION_CRITERIA.md)：canonical 三阶段 **Σ R 提升**、maxDD 不恶化、笔数不可塌缩、trading map 语义对齐。

## 产物

`results/tpc/experiments/t5_wall_entry_v2_20260616/<variant>/<segment>/capital_report.json`
