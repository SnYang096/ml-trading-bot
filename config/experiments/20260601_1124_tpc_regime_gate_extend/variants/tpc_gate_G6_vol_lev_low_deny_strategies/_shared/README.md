# Shared tree-core FeatureStore (core-4 union)

Union of `requested_features` from **bpc / tpc / me / srb** (~95 nodes).  
Build once; each strategy runs `--prepare-only` with the same layer for labels only.

## Build

```bash
mlbot feature-store build -c config/strategies/_shared
# Note resolved layer name from output, e.g. features_tree_core_120T_<hash10>
```

Or resolve programmatically:

```bash
python -c "from src.feature_store.tree_core_layer import resolve_tree_core_layer; print(resolve_tree_core_layer())"
```

## Prepare-only (per strategy, shared features cache)

```bash
LAYER=$(python -c "from src.feature_store.tree_core_layer import resolve_tree_core_layer; print(resolve_tree_core_layer())")

mlbot train final --no-docker --prepare-only \
  -c config/strategies/bpc \
  --feature-store-layer "$LAYER" \
  --output-dir results/train_final/bpc/<run_id>
```

Repeat for `tpc`, `me`, `srb` with the **same** `--feature-store-layer`.
