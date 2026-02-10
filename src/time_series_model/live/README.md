# Live strategies overview

This folder contains live-trading helpers.

## TL;DR which file to use?
- **Run real trading quickly** → `scripts/run_live.py` (main entry).  
  Command:  
  `MLBOT_LIVE_SYMBOLS=BTCUSDT python scripts/run_live.py`
- **Understand the live strategy logic** → see `bpc_live_strategy.py` (BPC decision engine).

## Files
- `bpc_live_strategy.py` — BPC pure-logic decision engine (Gate → Entry Filter → Evidence → Tier → TradeIntent).
- `live_feature_plan.py` — live feature plan resolver (base training plan + live overlay).
- `execution_intelligence.py` — execution profile builder (SL/TP, holding time, confidence).
- `execution_profile_apply.py` — utilities for applying execution profiles to orders.
- `enforcement.py` — Constitution enforcement layer.

## Architecture
The live trading flow is: **WebSocket → OrderFlowListener → BPCLiveStrategy → ConstitutionExecutor → OrderManager**

See `docs/live_stream/README.md` for details.

## Live Feature Plan
`IncrementalFeatureComputer` reads `MLBOT_LIVE_FEATURE_PLAN_YAML` (default:
`config/live/live_feature_plan.yaml`) to decide which features are
expected/kept in live.
