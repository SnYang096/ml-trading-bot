# New Strategy Feature Pipeline Playbook

This note is the contract for adding a new strategy, especially when an AI agent
is wiring feature selection, backtests, and live/runtime code.

## Core Rule

Every research path must use the same feature surface:

1. Declare feature nodes in `config/strategies/<strategy>/features.yaml`.
2. Build FeatureStore from that config.
3. Declare candidate feature nodes in `features_prefilter.yaml` or the relevant
   selector config.
4. Declare strategy-specific feature semantics in `semantic_polarity.yaml` when
   threshold or range scans are needed.
5. Make backtest and live paths read from FeatureStore when rules reference
   FeatureStore columns.

Do not let a strategy build FeatureStore and then run a backtest on a separate
OHLCV-only dataframe. That creates false negatives: selector rules can be valid,
but the runtime dataframe will not contain the columns, so every rule masks out
all trades.

## File Roles

`features.yaml` is the materialization contract. If a column may be used by a
selector, prefilter, gate, execution layer, report, or live daemon, its `_f` node
must be reachable from this file.

`features_prefilter.yaml` is the candidate pool for prefilter/eligibility search.
It should be a subset of the materialized feature universe unless there is a very
specific reason not to compare that feature.

`semantic_polarity.yaml` is strategy-local meaning. The same feature may have a
different direction in different strategies. For example, `bpc_semantic_chop` is
risk for BPC-style breakout continuation, but eligibility for `chop_grid`.

`research/calibrate_roll.default.yaml`, `research/research_roll.features_on.yaml`, and `research/validate_static.full_study.yaml`
describe how to run experiments. They should point back to the strategy config
directory and, for FeatureStore-backed strategies, set `feature_store_dir` and
`feature_store_timeframe` in the strategy-specific backtest section.

`archetypes/*.yaml` contains the adopted runtime knobs and rules. Research code
may write experiment copies of these files; adopt/deploy should only promote the
intended archetype deltas.

## Backtest Contract

A backtest dataframe must include all columns used by strategy rules.

For tree-style strategies, this usually happens through the existing prepare /
training pipeline after FeatureStore build.

For multi-leg strategies, standalone backtests must explicitly read FeatureStore
when configured. `chop_grid` now follows this rule: if `grid_backtest` has
`feature_store_dir` and a detected or explicit `feature_store_layer`, both `raw`
and `ts_quantile` chop modes load `features.yaml` outputs from FeatureStore.
`raw` still uses `bpc_semantic_chop` as the entry signal; additional WPT/Hurst/
Hilbert columns are available for prefilter rules.

## Adding A New Strategy

1. Create `config/strategies/<strategy>/features.yaml` with the full feature
   materialization pool.
2. Create strategy archetypes under `archetypes/` for runtime rules and execution
   knobs.
3. Add `research/calibrate_roll.default.yaml`; add `research_roll.features_on.yaml` and `validate_static.*.yaml` when the
   strategy needs rolling validation or static holdout validation.
4. Ensure the research YAML runs FeatureStore build before any selector/backtest
   that depends on feature columns.
5. Ensure the backtest command has a FeatureStore read path, or explicitly prove
   every rule column is computed locally.
6. Add a test that injects a fake or tiny FeatureStore result and asserts that a
   representative selector/rule column survives into the backtest dataframe.
7. Add a no-trade guardrail test when rules reference missing columns; missing
   columns should fail loudly in strict FeatureStore mode, not silently produce
   zero trades.

## Multi-Leg Selector Rule Generation

Multi-leg feature selection should be configuration-driven:

- candidate nodes come from `features_prefilter.yaml`;
- output columns come from `feature_dependencies.yaml`;
- monotonic directions come from `semantic_polarity.yaml`;
- non-monotonic/unknown columns use range scans when normalized and comparable;
- raw price, raw flow, categorical, and count columns are excluded from automatic
  threshold scans.

Avoid adding per-feature Python templates. If a new feature requires special
handling, first express it through config: polarity, normalization metadata, or
an explicit candidate set.

## Sanity Checklist

Before trusting a feature-selection result, verify:

- the FeatureStore layer exists for the strategy, symbols, timeframe, and date
  range;
- the selected rule columns are present in the backtest dataframe;
- strict FeatureStore mode is enabled for research comparisons that depend on
  materialized columns;
- zero-trade candidates are caused by real thresholds, not missing columns;
- live providers and backtests consume the same feature names and timeframe.

