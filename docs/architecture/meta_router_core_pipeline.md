# MetaRouterCore Pipeline (Live)

## Purpose

Provide a framework-agnostic, pure-logic router (`MetaRouterCore`) that consumes
live features and emits executable intents, then route those intents through
constitution enforcement and order execution.

## Data Flow

```text
WebSocket
  → live_data_stream (OrderFlowListener)
      → MetaRouterCore (pure logic)
          → ConstitutionExecutor (safety + slot/add_position)
              → OrderManager (execution + persistence)
```

## Responsibilities

### MetaRouterCore (pure logic)

- Input: live feature dict + recent bars.
- Output: `TradeIntent` with:
  - `action` (LONG/SHORT)
  - `archetype`
  - `execution_strategy`
  - `execution_tags` / `execution_evidence`
  - `size_multiplier` / optional `pcm_budget`
  - `execution_profile` (RR/holding constraints)
- Uses:
  - Router thresholds (`pred_dir_prob`, `pred_mfe_atr`, `pred_mae_atr`, `pred_t_to_mfe`)
  - Archetype gate rules (`execution_archetypes.yaml`)
  - Direction resolver (structure-based)
  - Optional PCM budget for sizing

### ConstitutionExecutor (safety + slots)

- Enforces kill-switch limits and risk gates.
- Reserves slots, checks add-position constraints.
- Persists runtime state.

### OrderManager (execution)

- Converts intent into concrete order placement.
- Owns broker/exchange integration, storage, and order lifecycle.

## Live Entry Point

`OrderFlowListener.on_feature_callback` triggers `_handle_features`, which:

1) Calls `MetaRouterCore.decide(features, bars)`  
2) Applies constitution enforcement  
3) Places order via `OrderManager`  

## Tests

Smoke test: `tests/live_data_stream/test_meta_router_pipeline.py`

- Ensures a valid feature payload triggers a single order.
- Confirms slot usage is enforced on subsequent calls.
