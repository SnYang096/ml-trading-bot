# chop_grid — segment validate (timeline)

**状态：** 待跑批（manifest 2026-06-03）  
**产物：** `results/chop_grid/experiments/segment_validate_20260603_timeline/`

## 判决标准

| 检查项 | 通过条件 |
|--------|----------|
| 四段 timeline return | 全段为正或 OOS 至少 flat |
| OOS `recent_6m_oos` | timeline `return_pct` > 0 且 `segment_win_rate` > 40% |
| vs trend_scalp | 同窗口不应显著弱于 trend_scalp 一个数量级 |

## 结果

> 跑完 `segment_summary.csv` 后填入。

| Segment | return_pct | return_pct_eq_mean | daily_sharpe | max_drawdown_portfolio | segment_win_rate |
|---------|----------:|-------------------:|-------------:|-----------------------:|-----------------:|
| bear_2022 | — | — | — | — | — |
| bull_2023_2024 | — | — | — | — | — |
| recent_range_to_bear | — | — | — | — | — |
| recent_6m_oos | **-0.75%** | -0.75% | -0.88 | -1.05% | 37.3% |

**2026-06-03 跑批（timeline）：** OOS 仍略负，与 20260602 eq-weight -0.75% 一致；**不 promote**。

## 历史对照

20260602 eq-weight OOS **-0.75%**（pooled -3.7%）— 不 promote。
