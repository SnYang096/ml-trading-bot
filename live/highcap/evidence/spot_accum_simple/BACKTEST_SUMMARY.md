# spot_accum_simple baseline (3 symbols)

**Run date:** 2026-05-18  
**Window:** 2022-01-01 → 2026-05-01 (UTC)  
**Symbols:** BTCUSDT, BNBUSDT, SOLUSDT  
**Initial equity:** $10,000 USDT  

## Results

| Metric | Value |
|--------|-------|
| Final equity | $49,344.62 |
| Total return | +393.4% |
| CAGR (capital report) | 44.60% |
| Closed trades | 514 |
| BTC buy-hold (fee-free) | +65.0% → $16,495.84 |
| EW basket (BTC+BNB+SOL) | +11.4% → $11,135.32 |

## Budget (constitution at run time)

- BTC 50% ($5,000), BNB 25% ($2,500), SOL 25% ($2,500)
- `tranches_per_symbol`: 20

## Artifacts

- `event_backtest.json` — full backtest output (copy in this folder)
- `run.log` — console log
- Canonical: `results/120T/spot_accum_simple/event_backtest_spot_accum_simple.json`

## Notes

Strategy equity materially exceeded fee-free BTC buy-and-hold over this window; EW basket underperformed due to alt weighting and timing.
