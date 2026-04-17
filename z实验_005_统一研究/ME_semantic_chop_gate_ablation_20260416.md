# ME：`gate_me_semantic_chop_high` 消融与 rolling 结论（2026-04-16）

## 背景

- ME 曾对齐 BPC/TPC，在 `archetypes/gate.yaml` 中配置 `gate_me_semantic_chop_high`（`bpc_semantic_chop > 0.62`，deny）。
- 月度 rolling 中发现：locked 语义门有时**未进入** `gate_draft` / 最终 `strategies_calibrated`（prefilter 候选对比小循环里 Gate Opt `--promote` 覆盖 `archetypes/gate.yaml` 后，locked 注入源丢失）。已在 `scripts/auto_research_pipeline.py` 用**月初 locked gate 快照**修复。
- 修复后需判断：**这条门对 ME 是否仍有边际价值**。

## 信号层统计（修复后 run 的 bar 级 funnel）

在 `20260416_090720` 上，将 `funnel_per_bar` 与 feature store 的 `bpc_semantic_chop` 对齐后：

- `AUC(chop → gate_deny)` ≈ **0.504**（接近随机）。
- 阈值 0.62 两侧 `gate_deny` 比例几乎相同。
- 对 `pcm_n_accepted>0` 的判别 AUC 仅约 **0.53**。
- 基础开仓单 `pnl_r` 与 `bpc_semantic_chop` 相关接近 0。

结论：**在 ME 当前 prefilter + gate 体系下，`bpc_semantic_chop` 单独作 hard veto 的边际很弱。**

## Rolling 对照（同窗口 27 个月）

| Run ID | 说明 | stitched_total_r | stitched_total_trades |
|--------|------|------------------|------------------------|
| `20260416_152302` | 修复 locked 注入后，**模板仍含** `gate_me_semantic_chop_high` | 660.4274 | 410 |
| `20260416_222324` | 同上修复，**模板已去掉** `gate_me_semantic_chop_high`（仅保留两条 EMA1200 语义门） | **697.8116** | 412 |

- **差值**：约 **+37.38R**，交易数 **+2**。
- 月度上：`2025-04`、`2026-02`、`2024-12` 等改善明显；`2025-08`、`2026-03`、`2025-07` 等有所回撤，净效果仍为正。

在 `20260416_222324` 的 `strategies_calibrated/me` 产物中 **`gate_me_semantic_chop_high` 零命中**（与从 archetype 删除一致）。优化器仍可能以其他 rule id 使用 `bpc_semantic_chop`，与「删除固定 ME 语义硬门」是两件事。

## 产品 / 配置结论

1. **ME 不再保留 `gate_me_semantic_chop_high` 作为 locked 硬门**；已从 `config/strategies/me/archetypes/gate.yaml` 移除。
2. **保留** `gate_me_late_long_above_ema1200` / `gate_me_late_short_below_ema1200`（`locked` + `promote_never_disable`），继续由 Gate Train 调阈值。
3. BPC/TPC 侧 `*_semantic_chop_high` **未改**；本结论仅针对 ME。

## 代码与标签

- 移除 ME semantic chop 硬门提交：`155fea0`（message: `feat(me): drop semantic chop hard gate`）。
- 本实验记录：本文件。
- 建议 tag：`me-semantic-chop-ablation-20260416`（与 rolling 对照 run 时间一致）。
