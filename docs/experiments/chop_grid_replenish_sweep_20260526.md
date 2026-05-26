# Chop grid: `max_replenish_per_level_per_segment` sweep (2026-05-26)

## Purpose

Align live `ChopGridLiveEngine` with backtest `ChopGridEngine`: after a grid level take-profit (TP),
optionally place the same limit again at the original `center ± spacing × level` while the regime
segment remains active.

Config semantics (user-confirmed):

| Value | Meaning |
|-------|---------|
| `0` | One fill per level per regime segment (legacy live) |
| `N>0` | Up to **N** post-TP replenishes per level (total entries ≤ **1+N**) |
| `null` | Unlimited (legacy backtest default) |

## Sweep setup

- **Script:** `scripts/sweep_chop_grid_replenish.py`
- **Symbols:** BTC, ETH, SOL, BNB, XRP, ADA (starter_a highcap)
- **Range:** 2022-01-01 .. 2026-05-01
- **Timeframe:** 120T (2h)
- **Prod-like params:** `entry_chop_min=0.50`, `exit_chop_below=0.32`, `atr_mult=1.0`,
  `min_pct=0.01`, `max_levels_per_side=2`, `max_open_levels_total=4`
- **Grid:** `0,1,2,3,5,null`
- **Raw CSV (committed):** `docs/experiments/chop_grid_replenish_sweep_20260526.csv`
- **Local rerun output:** `results/chop_grid/sweep_replenish_full.csv` (gitignored)

## Aggregate results (6 symbols, full sample)

| N | Total trades | Replenish TP trades* | Sum PnL (capital units) | vs N=0 | Trade-weighted forced† | Worst segment MDD |
|---|-------------|----------------------|-------------------------|--------|------------------------|-------------------|
| 0 | 401 | 0 | 0.846 | — | 41.1% | -0.94% |
| **1** | **484** | **39** | **0.998** | **+17.9%** | 43.2% | -0.94% |
| 2 | 496 | 44 | 1.019 | +20.4% | 43.5% | -0.94% |
| 3 | 498 | 45 | 1.023 | +20.9% | 43.3% | -0.94% |
| 5 / null | 498 | 45 | 1.023 | +20.9% | 43.3% | -0.94% |

\* Sum of segment `replenish_trades` (non-first `grid_tp` per level in segment).  
† Approximate: Σ(forced_rate × trades) / Σ(trades) per N bucket.

## Per-symbol: N=1 vs N=0

| Symbol | PnL (N=0) | PnL (N=1) | Δ PnL | Δ trades |
|--------|-----------|-----------|-------|----------|
| ADAUSDT | 0.199 | 0.250 | +0.051 | +27 |
| ETHUSDT | 0.130 | 0.167 | +0.036 | +17 |
| SOLUSDT | 0.154 | 0.174 | +0.020 | +11 |
| XRPUSDT | 0.147 | 0.165 | +0.019 | +10 |
| BNBUSDT | 0.108 | 0.124 | +0.016 | +8 |
| BTCUSDT | 0.108 | 0.117 | +0.009 | +10 |

All six symbols improve with N=1.

## Plateau (N≥2 ≈ unlimited)

Example **BTCUSDT** trades / PnL by N: 52 → 62 → 65 → 66 → 66 → 66.  
Example **ETHUSDT**: 69 → 86 → 89 → 89 → 89 → 89.

Marginal gain from N=1 to N=2 is small; N≥3 matches unlimited in this sample.

## KPI gates (calibrate_roll)

Target: `forced_exit_rate ≤ 0.35`, `max_drawdown_r ≥ -0.08`.

- Worst segment MDD ≈ **-0.94%** passes comfortably for all N.
- Trade-weighted forced exit ≈ **41–43%** for all N (N=0 already ~41%); replenishment adds ~**+2.1pp**,
  not a regime change in tail risk.

## Decision

**Live production (`live/highcap/config/strategies/chop_grid/archetypes/execution.yaml`):**

```yaml
max_replenish_per_level_per_segment: 1
```

Research default remains `null` in `config/strategies/chop_grid/archetypes/execution.yaml`.

## Reproduce

```bash
python scripts/sweep_chop_grid_replenish.py \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --start 2022-01-01 --end 2026-05-01 \
  --sweep-replenish 0,1,2,3,5,null \
  --out-csv results/chop_grid/sweep_replenish_full.csv  # or any path; see committed CSV above
```
