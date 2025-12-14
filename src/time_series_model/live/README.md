# Live strategies overview

This folder contains live-trading helpers built on Nautilus Trader.

## TL;DR which file to use?
- **Run real trading quickly** → `run_nautilus_strategy.py` (main entry).  
  Command:  
  `python -m time_series_model.live.run_nautilus_strategy --strategy sr_reversal --symbol BTCUSDT-PERP --timeframe 15T --testnet`
- **Understand or extend the live strategy logic** → read/modify `nautilus_strategy_with_features.py`.

## Files
- `run_nautilus_strategy.py` — module-friendly entry script. Loads configs, builds Nautilus TradingNode, wires feature-engineering strategy, then runs.
- `nautilus_strategy_with_features.py` — main strategy class. Handles feature loading, model inference, order routing, risk hooks.

## Requirements / Notes
- Nautilus Trader must be installed.
- Binance keys must be set in env vars for the entry script (see docstring in `run_nautilus_strategy.py`).

