# SRB reverse bug fix validation (2026-04-17)

## Goal
- Validate fix for `AttributeError: 'ReverseIntent' object has no attribute 'execution_profile'`.
- Confirm reverse path can open positions and complete event backtest without crash.

## Baseline (broken run)
- Run: `results/srb/slow-rolling-sim/_rolling_sim/20260416_220110`
- Previously failed months included: `2023-09`, `2023-11`, `2023-12`, `2024-02`, `2024-03`, `2024-04`, `2024-06`, `2024-07`, `2024-08`, `2024-09`, `2024-10`, `2024-12`.
- Error signature in logs:
  - `AttributeError: 'ReverseIntent' object has no attribute 'execution_profile'`

## Validation reruns (after fix)
- Validation output root:
  - `results/srb/slow-rolling-sim/_rolling_sim/20260417_reverse_fix_check`

### Case A: 2023-09 (previously crashed)
- Command:
  - `python scripts/event_backtest.py --strategy srb --start-date 2023-09-01 --end-date 2023-09-30 --strategies-root .../20260416_220110/fast_month_2023-09/strategies_calibrated --data-path data/parquet_data --export .../event_trades_2023-09.csv --output .../event_backtest_2023-09.json --fast --map-extra-months 4 --dump-end-state .../end_state_2023-09.json --keep-open-positions`
- Result:
  - Exit code: `0`
  - Trades: `5`
  - Funnel: `srb_reverse_opened = 1`, `srb_reverse_expired = 2`
  - Artifacts:
    - `event_backtest_2023-09.json`
    - `event_trades_2023-09.csv`
    - `trading_map_2023-09.html`

### Case B: 2024-12 (previously crashed)
- Command:
  - `python scripts/event_backtest.py --strategy srb --start-date 2024-12-01 --end-date 2024-12-31 --strategies-root .../20260416_220110/fast_month_2024-12/strategies_calibrated --data-path data/parquet_data --export .../event_trades_2024-12.csv --output .../event_backtest_2024-12.json --fast --map-extra-months 4 --resume-state .../20260416_220110/fast_month_2024-11/srb/end_state.json --dump-end-state .../end_state_2024-12.json`
- Result:
  - Exit code: `0`
  - Trades: `14`
  - Funnel: `srb_reverse_opened = 2`
  - Artifacts:
    - `event_backtest_2024-12.json`
    - `event_trades_2024-12.csv`
    - `trading_map_2024-12.html`

## Conclusion
- The reverse-entry code path no longer crashes on missing `execution_profile`.
- Reverse entries are now actually opened (`srb_reverse_opened > 0`) on previously failing months.
- Fix is validated for both early-window and late-window months with and without `--resume-state`.
