This directory is **nnmultihead-first**.

It defines StrategyProfile YAMLs used by the live adapter / execution constitution:
- `router_mode` (NO_TRADE/MEAN/TREND)
- `execution_strategy_id` (archetype)
- `evidence_rules` (how to compute evidence flags from features / primitives)

This intentionally does **NOT** reuse `config/strategies/*` (tree-model legacy configs),
which may be deprecated.

Core nnmultihead strategy IDs (TC/TE/FR/ET):
- `trend_continuation_tc`
- `trend_expansion_te`
- `failure_reversion_fr`
- `exhaustion_turn_et`
