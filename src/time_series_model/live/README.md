# Live strategies overview

This folder contains live-trading helpers.

## TL;DR which file to use?
- **Run real trading quickly** → `scripts/run_live.py` (main entry).  
  Command:  
  `python scripts/run_live.py` (see `README_CN.md` for environment variables)
- **Understand the live strategy logic** → see `src/time_series_model/core/meta_router_core.py` (core decision logic).

## Files
- `meta_router_config.py` — YAML schema loader for live config (enabled archetypes, size multipliers, router thresholds).
- `nnmh_live_inferencer.py` — optional nnmultihead online inference wrapper (model.pt → preds_* for rule router).
- `live_feature_plan.py` — live feature plan resolver (base training plan + live overlay).
- `execution_intelligence.py` — execution profile builder (SL/TP, holding time, confidence).
- `execution_profile_apply.py` — utilities for applying execution profiles to orders.

## Architecture
The live trading flow is: **WebSocket → OrderFlowListener → MetaRouterCore → ConstitutionExecutor → OrderManager**

See `docs/live_stream/README.md` for details.

## Live Feature Plan
`IncrementalFeatureComputer` reads `MLBOT_LIVE_FEATURE_PLAN_YAML` (default:
`config/live/live_feature_plan.yaml`) to decide which features are
expected/kept in live.
