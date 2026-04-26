# 实验 011：Chop Grid / Dual Add Trend 管线决策

## 结论

`chop_grid` 和 `dual_add_trend` 先保持 **multi-leg 专用管线**，不强行对齐到 BPC/TPC/ME 这类完整 ML 慢管线。

这里的判断不是“特征不重要”，而是这两类策略的核心风险和自由度主要不在 `features.yaml -> SHAP -> gate model -> entry_filter` 这条 supervised ML 路径上，而在：

- 场景过滤是否足够稳定；
- 库存是否会堆积；
- regime 变化后能否及时退出；
- forced exit / trend flip 是否吞掉大量小 TP；
- 手续费、滑点、funding 是否吃掉格子利润；
- 多 symbol 同时触发时，组合层面 gross exposure 是否过高。

因此当前更合理的方向是：**特征定义相对稳定，慢节奏更新结构快照，月度用最近窗口校准 multi-leg 执行/库存 profile，并用库存风险 KPI 做部署约束。**

## 为什么不补完整 ML 对齐

BPC 慢管线的主要职责是：特征构建、标签生成、prefilter/gate/entry 规则搜索、方向层调优、event backtest 与执行参数优化。它依赖 `labels_gate`、`has_prefilter`、`has_direction` 以及一套 supervised / quasi-supervised 的策略契约。

`chop_grid` 和 `dual_add_trend` 当前不是这种形态：

- `chop_grid` 是网格库存策略，核心是 chop/box 场景、网格 spacing、最大层数和退出规则。
- `dual_add_trend` 是趋势双腿加仓策略，核心是趋势场景、加仓步长、TP 距离、gross/net exposure、flip 行为和 loser timeout。
- 两者都走 multi-leg 回测脚本，不走单 `TradeIntent` 的 event backtest 主路径。
- 两者没有 `labels_gate`，也没有真正需要训练的 gate model。

如果强行补 SHAP / gate / entry_filter，会把问题变成“ML 选场景 + multi-leg 执行”的混合策略。这个方向可以作为后续增强，但当前会显著增加工程复杂度和过拟合风险，不适合作为上线前的默认对齐要求。

## 当前 slow / turbo 的含义

对 multi-leg 策略，`rolling.mode: slow_realistic` 和 `turbo_fixed_features` 的语义与 BPC 不完全相同。

- **Slow realistic**：周期性构建结构快照，并在每个月的校准窗口里选择 multi-leg profile。
- **Turbo fixed features**：固定策略根目录，但仍可在月度 replay 中做 multi-leg profile 校准。
- **threshold_calibration**：对 multi-leg 不是 ML 阈值标定，不是 prefilter/gate/entry 的统计规则搜索，而是若干执行/场景 profile 的候选择优。

也就是说，multi-leg 里的 `threshold_calibration` 应理解为：

- `chop_grid`：在 `box_window`、`entry_chop_min`、`exit_chop_below`、`atr_mult`、`min_pct`、`exclude_box_prefilter` 等候选之间择优。
- `dual_add_trend`：在 `box_window`、`entry_min`、`exit_below`、`max_semantic_chop_entry`、`max_semantic_chop_hold`、`step_atr_mult` 等候选之间择优。

这不是“特征筛选”，而是 **规则型 multi-leg 策略的 profile calibration**。

## 需要补齐的三个治理点

### 1. 显式记录 multi-leg calibration 语义

需要在策略 README / 实验文档中明确说明：

- `threshold_calibration` 对 multi-leg 不代表 ML 阈值标定；
- 它代表执行/场景 profile 选择；
- 选择依据是校准窗口上的 multi-leg 回测指标；
- 主要防的是库存尾部风险，而不是特征漂移。

这样可以避免后续把 `chop_grid` / `dual_add_trend` 和 BPC 慢管线混为一谈。

### 2. 把 candidate grid 从代码搬到 YAML

当前 multi-leg 候选参数写在 `scripts/auto_research_pipeline.py` 的 `_multileg_calibration_candidates()` 里。长期看这不利于追踪实验，也不利于比较 slow/turbo 配置差异。

建议后续迁移到：

- `config/strategies/chop_grid/research.yaml`
- `config/strategies/dual_add_trend/research.yaml`

目标是让候选 profile 成为策略配置的一部分，例如：

```yaml
calibration_profiles:
  - name: balanced
    box_window: 120
    entry_chop_min: 0.40
    exit_chop_below: 0.25
    atr_mult: 0.50
    min_pct: 0.004
```

这样 slow/turbo 的每次选择都能追溯到配置版本，而不是隐藏在 pipeline 代码里。

### 3. 补统一 KPI gate / deploy gate

multi-leg 最大风险不是“选错一个特征”，而是库存和退出尾部风险。因此比补 ML 特征筛选更重要的是补统一的部署门槛。

建议至少覆盖：

- `min_trades`
- `min_total_pnl_per_capital`
- `max_drawdown`
- `max_forced_rate`
- `max_risk_stop_rate`
- `max_trend_flip_loss`（适用于 `dual_add_trend`）
- `max_gross_exposure` / portfolio-level exposure cap
- fee / slippage / funding 压力测试后的最低收益要求

这些 gate 应该用于 slow/turbo rolling 的 stitched summary 和 capital report，而不是只看单月 TP 率或单笔 `pnl_pct`。

## 2h 信号 + segment 内细粒度执行（已实现 MVP）

与 `event_backtest` 里「决策时钟 vs 1m 更新」的思路对齐：`chop_grid_backtest.py` 与 `diagnose_dual_add_trend.py` 支持：

- **`--timeframe`**：信号与 segment 边界（例如 `2h`）；
- **`--execution-timeframe`**：段内成交路径（例如 `1min`），与信号不同时，在 segment 墙钟范围内把信号列 **backward `merge_asof`** 到执行 K 线上，再用细 K 的 high/low 跑库存。

实现见 `src/time_series_model/grid/subbar_replay.py`；`ChopGridEngine.simulate_segment` 支持 **`anchor_close` / `anchor_atr`**，用信号段起点冻结网格间距，避免用 1m 第一根 close 当 center。

注意：`dual_add` 在子周期模式下 **`max_loser_hold_bars` 计的是执行 bar 数**（例如 1m），与纯 2h 回测不可直接比数字；需自行换算或调参。

## 后续增强顺序

优先级建议：

1. 将 profile candidates 配置化；
2. 在策略 README 中补 multi-leg calibration 语义；
3. 在 rolling summary / deploy gate 中增加库存风险 KPI；
4. 增加 fee、slippage、funding 压力测试；
5. 再评估是否需要做“ML 选场景 + multi-leg 执行”的混合策略。

在当前阶段，`chop_grid` 和 `dual_add_trend` 可以继续按 multi-leg 专用管线推进 dry-run / live-test，不需要先补完整 ML 慢管线。
