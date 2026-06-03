# chop_grid replenish ablation — 判决

**日期：** 2026-06-03  
**窗口（smoke）：** `recent_6m_oos`（2025-10 → 2026-03）  
**配置：** prod archetype + `calibrate_roll.default.yaml`（2h / 1min exec，fee 20bps）

## recent_6m_oos 对比（timeline return_pct）

| 变体 | max_replenish | return_pct | return_pct_pooled | segment_win_rate | trades | replenish_trades* |
|------|--------------:|----------:|------------------:|-----------------:|-------:|------------------:|
| replenish_unlimited | null | **-0.75%** | -3.75% | 37.3% | 220 | 19 |
| replenish_off | 0 | **-0.65%** | -3.24% | 38.4% | 181 | 0 |
| replenish_live | 1 | _(pending)_ | _(pending)_ | _(pending)_ | _(pending)_ | _(pending)_ |

\* segment `replenish_trades` 列求和（unlimited 跑批：`segment_validate_20260603_timeline`）。

## 解读

- **关闭 replenish 略好但仍负：** OOS -0.65% vs unlimited -0.75%；trades 181 vs 220。**不能**恢复 20260526 proxy 的 +38% pooled（不同 universe/fee/exec 口径）。
- **20260526 全窗 sweep：** 长窗 aggregate 上 N=1 **优于** N=0（+17.9%）；与本 OOS 短窗结论不矛盾 — replenish 非 OOS 亏损主因。
- **20260526 semantic proxy +38.89% pooled：** 6 sym、2h-only exec、可能更低 fee；勿与 timeline 数字直接对比。

## 决策

- [x] `replenish_off` OOS 未显著转正 → **replenish 不是恢复「旧好数据」的开关**
- [ ] 继续查 fee(20bps) / 1min exec / 5 vs 6 symbols
- [ ] 可选跑 `replenish_live`（max=1）与四段 validate
- [ ] Promote live `max_replenish=1` 仅在 ablation + 四段 validate 通过后
