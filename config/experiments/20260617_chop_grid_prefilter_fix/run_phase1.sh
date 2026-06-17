#!/usr/bin/env bash
# chop_grid Phase 1: IC scan + Label scan
#
# Step 1: Merge feature_store + compute forward_rr
# Step 2: Run rd_loop (IC decay + feature plateau)
# Step 3: Build wall features incrementally (TODO)
# Step 4: Run wall IC scan (TODO)
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP_DIR=config/experiments/20260617_chop_grid_prefilter_fix
OUT_DIR=results/rd_loop/chop_grid_prefilter_fix_phase1

echo "=== Step 1: Prepare parquet (merge feature_store + forward_rr) ==="
python scripts/prepare_chop_grid_phase1_parquet.py \
  --feature-store-layer features_chop_grid_120T_c5a8c96e46 \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
  --horizon 50 \
  --direction bidir \
  --output "$OUT_DIR/features_with_fwd_rr.parquet"

echo ""
echo "=== Step 2: Run rd_loop Phase 1 (IC + plateau) ==="
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml "$EXP_DIR/rd_loop_chop_grid_phase1.yaml"

echo ""
echo "=== Step 3: Results ==="
echo "IC/plateau reports: $OUT_DIR/quick_scan/"
ls -la "$OUT_DIR/quick_scan/" 2>/dev/null || echo "(no quick_scan dir yet)"
