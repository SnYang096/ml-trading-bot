# Dual add trend research pipeline entrypoints

This directory mirrors the BPC profile naming:

- `turbo.yaml`: fixed features/regime thresholds, monthly profile calibration.
- `slow.yaml`: slower structural health check plus monthly profile calibration.
- `non_rolling.yaml`: full-window static holdout validation before gate/review.

The profile YAMLs use the same pipeline language as BPC:

- `threshold_calibration` controls which layers calibrate.
- `strategies.dual_add_trend.kpi_gates` holds KPI constraints.
- `dual_add_backtest` holds execution/report parameters.
- Strategy-owned calibration candidates are selected by the multi-leg dispatcher in code.

## Default execution replay (multi-leg dual_add)

Research and orchestrator configs use **2h for signal and segment boundaries** (features, regime, hysteresis) and **1m resampled OHLC for inventory simulation** (adds, TP, risk exits), so backtests are less optimistic than filling only on 2h highs/lows.

Set under `dual_add_backtest`:

| Key | Default (research) | Meaning |
| --- | --- | --- |
| `timeframe` | `2h` | Signal bar length; `scripts/diagnose_dual_add_trend.py` builds features and segments on this grid. |
| `execution_timeframe` | `1min` | Finer bars for `simulate_dual_add_segment`; 1m data is loaded from parquet and resampled. |
| `scale_max_loser_hold_to_signal` | `true` | When execution is finer than signal, scale `max_loser_hold_bars` so wall-clock patience matches the signal grid (e.g. 24×2h → 2880×1m). |
| `costs.market_exit_slippage_bps` | `5.0` | Adverse slip on basket/forced style exits (not per-leg `tp` fills). |
| `costs.intrabar_touch_buffer_bps` | `5.0` | Extra penetration past a level before add/TP is considered hit. |

`auto_research_pipeline.py` passes `--execution-timeframe`, `--scale-max-loser-hold-to-signal`, and cost flags into `diagnose_dual_add_trend.py`. Run artifacts include `signal_timeframe`, `execution_timeframe`, `execution_replay_enabled`, and `resolved_max_loser_hold_bars` in `summary.csv`.

Pipeline-wide defaults for multi-leg orchestration live in `config/pipelines/multileg_orchestrate_2h.yaml` under `dual_add_backtest` (same 1m + scaling + cost fields).

Rolling exports adoptable bundles under `results/dual_add_trend/<history>/dual_add_trend/<timestamp>/strategies/dual_add_trend/` (see `auto_research_pipeline` multileg export).
