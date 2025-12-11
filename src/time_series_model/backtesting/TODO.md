# Backtesting TODO (Nautilus integration)

The legacy per-strategy event backtests (`sr_reversal_backtest.py`, `sr_breakout_backtest.py`, `compression_breakout_backtest.py`, `trend_following_backtest.py`) were removed from `src/time_series_model/strategies/backtesting/` to simplify the training pipeline (now using a single VectorBT backtest).

Next steps (to be implemented here under `src/time_series_model/backtesting/`):
1) Rebuild per-strategy event-driven backtests with Nautilus Trader for:
   - SR Reversal
   - SR Breakout
   - Compression Breakout
   - Trend Following
2) Keep VectorBT for fast training-time evaluation; use Nautilus for rich execution logic (trailing stop, add/reduce, partial TP, slippage/latency).
3) Provide CLI runners and Makefile targets once Nautilus backtests are implemented.

Current state:
- Training/backtest default: `VectorBTBacktest` (simple thresholds, optional RR exit, optional ATR-based position sizing).
- No event-driven backtests are present; this directory will host the new Nautilus-based implementations.

