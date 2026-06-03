# trend_scalp — segment validate (timeline)

**状态：** 待跑批（manifest 2026-06-03）  
**产物：** `results/trend_scalp/experiments/segment_validate_20260603_timeline/`

## 判决标准

| 检查项 | 通过条件 |
|--------|----------|
| 四段 `return_pct` (timeline) | bear/bull/range 显著为正；OOS 不崩 |
| `max_drawdown_portfolio` | 与 20260602 `portfolio_cum_dd` 量级可比 |
| 分币格 | segment × symbol 多数为正 |
| Backtrader | recent_6m OOS 与 diagnose 偏差 < 2% |

## 结果

> 跑完 `segment_summary.csv` 后填入下表。

| Segment | return_pct | return_pct_eq_mean | max_drawdown_portfolio | segment_win_rate | trades |
|---------|----------:|-------------------:|-----------------------:|-----------------:|-------:|
| bear_2022 | — | — | — | — | — |
| bull_2023_2024 | — | — | — | — | — |
| recent_range_to_bear | — | — | — | — | — |
| recent_6m_oos | **+20.54%** | +20.54% | -1.18% | 70.7% | 1533 |

**2026-06-03 跑批（timeline）：** 五币分格全正（BTC +10.7% … SOL +31.5%）；与 20260602 eq-weight +20.6% 一致（本窗 timeline ≡ eq-mean）。

## 历史对照

20260602 eq-weight：`recent_6m_oos` +20.6% eq / +102.7% pooled — 见旧 DECISION。
