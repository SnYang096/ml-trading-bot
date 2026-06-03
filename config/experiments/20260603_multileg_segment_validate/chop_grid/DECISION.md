# chop_grid — segment validate (timeline)

**状态：** 已完成（2026-06-03，`max_replenish_per_level_per_segment: 1`）  
**产物：** `results/chop_grid/experiments/segment_validate_20260603_timeline/`  
**配置：** prod archetype + `calibrate_roll.default.yaml`（2h signal / **1min exec（canonical）**，20bps fee+funding，5 币）

> **Exec 对齐（2026-06-03 后）：** 回测默认 `execution_timeframe: 1min`；2h exec 仅作 legacy sensitivity，且与 live 一样使用右边界确认窗口。见 [`../../20260603_chop_grid_exec_align/README.md`](../../20260603_chop_grid_exec_align/README.md)。

## 判决标准

| 检查项 | 通过条件 |
|--------|----------|
| 四段 timeline return | bear/bull/range 显著为正；OOS 不崩 |
| OOS `recent_6m_oos` | timeline `return_pct` > 0 且 `segment_win_rate` > 40% |
| vs trend_scalp | 同窗口不应显著弱于 trend_scalp 一个数量级 |

## 结果（timeline `return_pct`，replenish=1）

| Segment | 窗 | return_pct | return_pct_pooled | daily_sharpe | max_dd_portfolio | segment_win_rate | trades |
|---------|-----|----------:|------------------:|-------------:|-----------------:|-----------------:|-------:|
| bear_2022 | 2022-01→2023-01 | **+3.54%** | 17.7% | 2.04 | -0.87% | 41.9% | 327 |
| bull_2023_2024 | 2023-01→2025-01 | **+5.16%** | 25.8% | 1.71 | -1.30% | 39.5% | 782 |
| recent_range_to_bear | 2025-01→2026-04 | **+2.46%** | 12.3% | 1.13 | -1.78% | 39.2% | 542 |
| **recent_6m_oos** | 2025-10→2026-03 | **-0.67%** | -3.37% | -0.79 | -0.97% | 37.3% | 216 |

对比 unlimited 旧跑批：OOS -0.75% → **-0.67%**（216 vs 220 trades）；历史三段与 20260602 eq-weight 基本一致（timeline ≡ eq-mean 本窗）。

## 决策

- **不 promote** — OOS timeline 仍略负，`segment_win_rate` 37% < 40%
- 长窗（bear/bull/range）在 prod cost stack 下 **仍为正**，策略未全面失效
- OOS 弱主因：1min exec + 20bps 全成本（见 [`../../20260603_chop_grid_reconcile_grid/DECISION.md`](../../20260603_chop_grid_reconcile_grid/)）

## 历史对照

20260602 eq-weight：OOS -0.75% pooled -3.7% — 口径与 replenish 均已更新。
