# Live strategies overview

This folder contains live-trading helpers built on Nautilus Trader.

## TL;DR which file to use?
- **Run real trading quickly** → `run_nautilus_strategy.py` (main entry).  
  Command:  
  `python -m time_series_model.live.run_nautilus_strategy --strategy-id meta_router --symbol BTCUSDT-PERP --timeframe 15T --testnet --live-config config/nnmultihead/live/meta_router_live_config_v1.yaml`
- **Understand or extend the live strategy logic** → read/modify `meta_router_strategy.py` (preferred) and `nautilus_strategy_with_features.py` (legacy / reference).

## Files
- `run_nautilus_strategy.py` — entry script. Builds Nautilus TradingNode, then runs **`MetaRouterStrategy`** (one strategy orchestrating multiple execution archetypes).
- `meta_router_strategy.py` — the MetaRouterStrategy implementation (Router → archetype selection → enforcement → order submit).
- `meta_router_config.py` — YAML schema loader for live config (enabled archetypes, size multipliers, router thresholds).
- `nnmh_live_inferencer.py` — optional nnmultihead online inference wrapper (model.pt → preds_* for rule router).
- `nautilus_strategy_with_features.py` — legacy strategy class (kept for reference).

## Requirements / Notes
- Nautilus Trader must be installed.
- Binance keys must be set in env vars for the entry script (see docstring in `run_nautilus_strategy.py`).

