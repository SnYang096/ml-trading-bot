# spot_accum_simple — live deployment evidence

**Production strategy name:** `spot_accum_simple` (`spot_accum` is archived under `bad-candidates`, not deployed.)

## What was deployed (2026-05-18)

| Item | Detail |
|------|--------|
| Command | `python scripts/deploy_config_to_live.py --deploy --strategy spot_accum_simple --yes` |
| Repo `HEAD` (when artifacts were written) | `9de20bf6ba8f2c96e246ddd6917aebd5980a0286` |
| Strategy package | `live/highcap/config/strategies/spot_accum_simple/` (from `config/strategies/spot_accum_simple/`) |
| Global sync | **`live/highcap/config/constitution/constitution.yaml` was overwritten from `config/constitution/constitution.yaml`** — review if the host previously had divergent limits. |

## Backtest justification (frozen copy)

Copies live here for audit traceability:

- `BACKTEST_SUMMARY.md` — 3-symbol baseline (BTC/BNB/SOL), window 2022-01-01 → 2026-05-01. Source: `results/120T/spot_accum_simple/archive/baseline_3sym_btc_bnb_sol_20220101_20260501/BACKTEST_SUMMARY.md`.
- `COMPARISON_3sym_vs_6sym.md` — side study (expanded symbols underperformed in that window). Source: `results/120T/spot_accum_simple/COMPARISON.md`.

Canonical large artifacts remain under `results/120T/spot_accum_simple/` (see BACKTEST_SUMMARY § Artifacts).
