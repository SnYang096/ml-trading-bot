#!/usr/bin/env bash
# Build shared tree_full FeatureStore (~289 nodes / ~940 output columns).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${ROOT}/scripts:${ROOT}"

LAYER="$(python -c "from src.feature_store.tree_full_layer import resolve_tree_full_layer; print(resolve_tree_full_layer())")"
echo "tree_full layer: ${LAYER}"

SYMBOLS="${SYMBOLS:-BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT}"
START="${START:-2024-01-01}"
END="${END:-2026-04-01}"

python scripts/build_feature_store_from_config.py \
  --config config/strategies/_shared \
  --features-yaml config/strategies/_shared/features_all.yaml \
  --symbols "$SYMBOLS" \
  --timeframe 120T \
  --data-path data/parquet_data \
  --start-date "$START" \
  --end-date "$END" \
  --warmup-months 6 \
  --root feature_store

echo "Done. Use --feature-store-layer ${LAYER} in prepare/train/rd_loop."
