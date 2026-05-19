# Trend flip / reseed ablation (2026-05-19)

**Question:** After switching from hedge-open to `initial_legs: TREND`, should an intra-segment
`trend_direction` flip immediately re-open in the new direction (reverse entry), or only
close offside inventory and wait for the **next regime segment**?

**Script:** `scripts/experiment_dual_add_flip_reseed.py`  
**Diagnose:** `scripts/diagnose_dual_add_trend.py` (paths updated to `config/strategies/trend_scalp`)  
**Window:** 2022-01-01 → 2026-03-31 · BTC/ETH/SOL/BNB/XRP · signal 2h · execution 1min replay  
**Profile:** `--no-initial-hedge` · basket TP · `regime_only` · `fee-bps 8`

## Variants

| Variant | flip_action | reseed_on_flip | Meaning |
| ------- | ----------- | -------------- | ------- |
| `reseed_on_flip_close_offside` | close_offside_all | true | **Old default:** flip closes offside, flat book re-seeds in new direction same segment |
| `flat_until_next_regime` | close_offside_all | false | **Adopted:** flip closes offside, stay flat until next regime segment |
| `keep_offside_legacy` | keep | true | Hedge-era: keep losing side on flip |

## Results (capital-bucket `sum_pnl_per_capital`)

| Variant | return_pct | portfolio_cum_dd | worst_segment | trades | segment_win_rate |
| ------- | ----------: | ----------------: | ------------: | -----: | -----------------: |
| **flat_until_next_regime** | **1273.1** | **-3.88%** | **-2.50%** | 14213 | **82.5%** |
| keep_offside_legacy | 1271.8 | -9.02% | -6.52% | 15048 | 82.4% |
| reseed_on_flip_close_offside | 1234.5 | -9.26% | -6.76% | 15610 | 80.6% |

Artifacts: `results/dual_add_flip_reseed_2022_2026/ablation_summary.csv`

## Conclusion

**Adopt `reseed_on_flip: false`** in `archetypes/execution.yaml`.

- Higher net PnL (~+3.1% vs old default)
- Much smaller portfolio drawdown (-3.9% vs -9.3%)
- Shallower worst segment (-2.5% vs -6.8%)
- Fewer trades → less fee drag; aligns with «regime confirms before entry»

**Rename:** strategy slug `dual_add_trend` → **`trend_scalp`** (config dir + constitution); engine module name unchanged for now.
