# E7 — four-segment validate (baseline vs dense 3L @ live costs)

**Window source:** `config/market_segment.yaml` (all four segments)

| Variant | levels | min_pct | maker | taker | forced slippage |
|---------|--------|---------|-------|-------|-----------------|
| baseline_prod | 2 | 0.011 | 20 | 20 | 20 |
| baseline_live_cost | 2 | 0.011 | 2 | 5 | 5 |
| dense_3l_live | 3 | 0.0033 | 2 | 5 | 5 |

**Output:** `results/chop_grid/experiments/stack_ablation_20260604/E7_four_segment/<variant>/segment_summary.csv`
