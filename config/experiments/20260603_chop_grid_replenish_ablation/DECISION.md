# chop_grid replenish ablation — 判决

**日期：** 2026-06-03  
**窗口（smoke）：** `recent_6m_oos`（2025-10 → 2026-03）  
**配置：** prod archetype + `calibrate_roll.default.yaml`（2h / 1min exec，fee 20bps + funding 20bps）

## recent_6m_oos 对比（timeline return_pct）

| 变体 | max_replenish | return_pct | return_pct_pooled | segment_win_rate | trades | replenish_trades |
|------|--------------:|----------:|------------------:|-----------------:|-------:|-----------------:|
| replenish_unlimited | null | -0.75% | -3.75% | 37.3% | 220 | 19 |
| replenish_off | 0 | -0.65% | -3.24% | 38.4% | 181 | 0 |
| **replenish_live** | **1** | **-0.67%** | -3.37% | 37.3% | 216 | 15 |

## 机制结论（采纳 default=1）

**全窗证据（20260526 sweep，2022-01→2026-05，6 sym，2h）：** `max_replenish=1` aggregate PnL **+17.9% vs N=0**，六币全改善；N≥2 边际饱和。

**OOS prod profile：** N=1 与 unlimited/off 同量级（约 -0.7%）；replenish **不是** OOS 亏损主因（主因见 [`../20260603_chop_grid_reconcile_grid/DECISION.md`](../20260603_chop_grid_reconcile_grid/)：1min exec + 20bps 全成本）。

**决策：**

- [x] **Research + live 统一默认 `max_replenish_per_level_per_segment: 1`**（`config/strategies/chop_grid/archetypes/execution.yaml`）
- [x] 机制保留：TP 后同档补挂 1 次，价位不重锚
- [ ] 四段 segment validate 用新默认重跑

## 历史对照

20260602 unlimited OOS -0.75% — 不 promote chop_grid on prod cost stack alone.
