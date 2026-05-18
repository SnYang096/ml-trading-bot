#!/bin/bash
# 四 Archetype 全量训练脚本
# 用法: nohup bash scripts/train_all_archetypes.sh > /tmp/train_all.log 2>&1 &
set -e
cd "$(dirname "$0")/.."

# ─── 参数 ─────────────────────────────────────────────────
UNIVERSE="config/download/crypto_4h_token_universe_groups.yaml"
SYMBOLS=$(python -c "
from src.data_tools.universe_config import load_universe_config
cfg = load_universe_config('$UNIVERSE')
syms = cfg.resolve_symbols_usdt(universe_set='starter_a')
print(','.join(syms))
")

TIMEFRAME_4H="240T"
TIMEFRAME_1H="60T"
TIMEFRAME_15M="15T"
START_DATE="2023-01-01"
END_DATE="2025-12-31"
HOLDOUT_START="2025-07-01"
HOLDOUT_END="2025-12-31"
DATA_PATH="data/parquet_data"
FS_ROOT="feature_store"
FS_LAYER_4H="unified_4h_2023_2025"
FS_LAYER_1H="unified_1h_2023_2025"
FS_LAYER_15M="unified_15m_2023_2025"
OUTPUT_ROOT="docs/z实验_005_统一研究/reports/train_$(date +%Y%m%d_%H%M%S)"

echo "================================================================"
echo "  四 Archetype 全量训练"
echo "  $(date)"
echo "  Symbols: $SYMBOLS"
echo "  4H strategies: BPC, FER"
echo "  1H strategies: ME"
echo "  15M strategies: LV"
echo "  Date range: $START_DATE to $END_DATE"
echo "  Holdout: $HOLDOUT_START to $HOLDOUT_END"
echo "  Output: $OUTPUT_ROOT"
echo "================================================================"

mkdir -p "$OUTPUT_ROOT"

# ─── Step 0: Feature Store 构建 (4H) ────────────────────
echo ""
echo "=== $(date) === Step 0a: Build Feature Store (4H) ==="
mlbot feature-store build \
  --config config/strategies/bad-candidates/bpc \
  --universe-config "$UNIVERSE" \
  --universe-set starter_a \
  --timeframe "$TIMEFRAME_4H" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --warmup-months 3 \
  --no-docker \
  2>&1 | tee "$OUTPUT_ROOT/featurestore_4h.log"

# ─── Step 0b: Feature Store 构建 (1H) ────────────────────
echo ""
echo "=== $(date) === Step 0b: Build Feature Store (1H) ==="
mlbot feature-store build \
  --config config/strategies/bad-candidates/me \
  --universe-config "$UNIVERSE" \
  --universe-set starter_a \
  --timeframe "$TIMEFRAME_1H" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --warmup-months 3 \
  --no-docker \
  2>&1 | tee "$OUTPUT_ROOT/featurestore_1h.log"

# ─── Step 0c: Feature Store 构建 (15min) ─────────────────
echo ""
echo "=== $(date) === Step 0c: Build Feature Store (15min) ==="
mlbot feature-store build \
  --config config/strategies/lv \
  --universe-config "$UNIVERSE" \
  --universe-set starter_a \
  --timeframe "$TIMEFRAME_15M" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --warmup-months 3 \
  --no-docker \
  2>&1 | tee "$OUTPUT_ROOT/featurestore_15m.log"

# ─── Step 1: BPC (4H) ─────────────────────────────────────
echo ""
echo "=== $(date) === Step 1: Train BPC (4H) ==="

# 1a. Train with rr_extreme labels
mlbot train final \
  --config config/strategies/bad-candidates/bpc \
  --symbol "$SYMBOLS" \
  --timeframe "$TIMEFRAME_4H" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --holdout-start-date "$HOLDOUT_START" \
  --holdout-end-date "$HOLDOUT_END" \
  --output-root "$OUTPUT_ROOT/bpc" \
  --labels config/strategies/bad-candidates/bpc/labels_rr_extreme.yaml \
  --deterministic \
  --no-docker \
  2>&1 | tee "$OUTPUT_ROOT/bpc/train.log"

# 1b. Apply archetype gate
echo "=== $(date) === Step 1b: Apply BPC Gate ==="
mlbot gate apply-archetype \
  --strategy bpc \
  --logs "$OUTPUT_ROOT/bpc/predictions.parquet" \
  --out "$OUTPUT_ROOT/bpc/logs_gated.parquet" \
  --features-store-root "$FS_ROOT" \
  --no-docker \
  2>&1 | tee "$OUTPUT_ROOT/bpc/gate.log"

# 1c. Optimize gate
echo "=== $(date) === Step 1c: Optimize BPC Gate ==="
python scripts/optimize_gate_unified.py \
  --strategy bpc \
  --logs "$OUTPUT_ROOT/bpc/logs_gated.parquet" \
  --output "$OUTPUT_ROOT/bpc/gate_optimized.json" \
  --write-back-intervals \
  2>&1 | tee "$OUTPUT_ROOT/bpc/gate_optimize.log"

# 1d. Optimize evidence
echo "=== $(date) === Step 1d: Optimize BPC Evidence ==="
python scripts/optimize_evidence_plateau.py \
  --strategy bpc \
  --logs "$OUTPUT_ROOT/bpc/logs_gated.parquet" \
  --output "$OUTPUT_ROOT/bpc/evidence_optimized.json" \
  2>&1 | tee "$OUTPUT_ROOT/bpc/evidence_optimize.log"

echo "=== $(date) === BPC training complete ==="

# ─── Step 2: ME (4H) ─────────────────────────────────────
echo ""
echo "=== $(date) === Step 2: Train ME (1H) ==="

mlbot train final \
  --config config/strategies/me \
  --symbol "$SYMBOLS" \
  --timeframe "$TIMEFRAME_1H" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --holdout-start-date "$HOLDOUT_START" \
  --holdout-end-date "$HOLDOUT_END" \
  --output-root "$OUTPUT_ROOT/me" \
  --labels config/strategies/me/labels_rr_extreme.yaml \
  --deterministic \
  --no-docker \
  2>&1 | tee "$OUTPUT_ROOT/me/train.log"

echo "=== $(date) === Step 2b: Apply ME Gate ==="
mlbot gate apply-archetype \
  --strategy me \
  --logs "$OUTPUT_ROOT/me/predictions.parquet" \
  --out "$OUTPUT_ROOT/me/logs_gated.parquet" \
  --features-store-root "$FS_ROOT" \
  --no-docker \
  2>&1 | tee "$OUTPUT_ROOT/me/gate.log"

echo "=== $(date) === Step 2c: Optimize ME Gate ==="
python scripts/optimize_gate_unified.py \
  --strategy me \
  --logs "$OUTPUT_ROOT/me/logs_gated.parquet" \
  --output "$OUTPUT_ROOT/me/gate_optimized.json" \
  --write-back-intervals \
  2>&1 | tee "$OUTPUT_ROOT/me/gate_optimize.log"

echo "=== $(date) === Step 2d: Optimize ME Evidence ==="
python scripts/optimize_evidence_plateau.py \
  --strategy me \
  --logs "$OUTPUT_ROOT/me/logs_gated.parquet" \
  --output "$OUTPUT_ROOT/me/evidence_optimized.json" \
  2>&1 | tee "$OUTPUT_ROOT/me/evidence_optimize.log"

echo "=== $(date) === ME training complete ==="

# ─── Step 3: FER (4H) ─────────────────────────────────────
echo ""
echo "=== $(date) === Step 3: Train FER (4H) ==="

mlbot train final \
  --config config/strategies/fer \
  --symbol "$SYMBOLS" \
  --timeframe "$TIMEFRAME_4H" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --holdout-start-date "$HOLDOUT_START" \
  --holdout-end-date "$HOLDOUT_END" \
  --output-root "$OUTPUT_ROOT/fer" \
  --labels config/strategies/fer/labels_rr_extreme.yaml \
  --deterministic \
  --no-docker \
  2>&1 | tee "$OUTPUT_ROOT/fer/train.log"

echo "=== $(date) === Step 3b: Apply FER Gate ==="
mlbot gate apply-archetype \
  --strategy fer \
  --logs "$OUTPUT_ROOT/fer/predictions.parquet" \
  --out "$OUTPUT_ROOT/fer/logs_gated.parquet" \
  --features-store-root "$FS_ROOT" \
  --no-docker \
  2>&1 | tee "$OUTPUT_ROOT/fer/gate.log"

echo "=== $(date) === Step 3c: Optimize FER Gate ==="
python scripts/optimize_gate_unified.py \
  --strategy fer \
  --logs "$OUTPUT_ROOT/fer/logs_gated.parquet" \
  --output "$OUTPUT_ROOT/fer/gate_optimized.json" \
  --write-back-intervals \
  2>&1 | tee "$OUTPUT_ROOT/fer/gate_optimize.log"

echo "=== $(date) === Step 3d: Optimize FER Evidence ==="
python scripts/optimize_evidence_plateau.py \
  --strategy fer \
  --logs "$OUTPUT_ROOT/fer/logs_gated.parquet" \
  --output "$OUTPUT_ROOT/fer/evidence_optimized.json" \
  2>&1 | tee "$OUTPUT_ROOT/fer/evidence_optimize.log"

echo "=== $(date) === FER training complete ==="

# ─── Step 4: LV (15min) ──────────────────────────────────
echo ""
echo "=== $(date) === Step 4: Train LV (15min) ==="

mlbot train final \
  --config config/strategies/lv \
  --symbol "$SYMBOLS" \
  --timeframe "$TIMEFRAME_15M" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --holdout-start-date "$HOLDOUT_START" \
  --holdout-end-date "$HOLDOUT_END" \
  --output-root "$OUTPUT_ROOT/lv" \
  --labels config/strategies/lv/labels_rr_extreme.yaml \
  --deterministic \
  --no-docker \
  2>&1 | tee "$OUTPUT_ROOT/lv/train.log"

echo "=== $(date) === Step 4b: Apply LV Gate ==="
mlbot gate apply-archetype \
  --strategy lv \
  --logs "$OUTPUT_ROOT/lv/predictions.parquet" \
  --out "$OUTPUT_ROOT/lv/logs_gated.parquet" \
  --features-store-root "$FS_ROOT" \
  --no-docker \
  2>&1 | tee "$OUTPUT_ROOT/lv/gate.log"

echo "=== $(date) === Step 4c: Optimize LV Gate ==="
python scripts/optimize_gate_unified.py \
  --strategy lv \
  --logs "$OUTPUT_ROOT/lv/logs_gated.parquet" \
  --output "$OUTPUT_ROOT/lv/gate_optimized.json" \
  --write-back-intervals \
  2>&1 | tee "$OUTPUT_ROOT/lv/gate_optimize.log"

echo "=== $(date) === Step 4d: Optimize LV Evidence ==="
python scripts/optimize_evidence_plateau.py \
  --strategy lv \
  --logs "$OUTPUT_ROOT/lv/logs_gated.parquet" \
  --output "$OUTPUT_ROOT/lv/evidence_optimized.json" \
  2>&1 | tee "$OUTPUT_ROOT/lv/evidence_optimize.log"

echo "=== $(date) === LV training complete ==="

# ─── Step 5: PCM 联合回测 ─────────────────────────────────
echo ""
echo "=== $(date) === Step 5: PCM Combined Backtest ==="

python scripts/backtest_execution_layer.py \
  --pcm \
    "bpc:$OUTPUT_ROOT/bpc/logs_gated.parquet" \
    "me:$OUTPUT_ROOT/me/logs_gated.parquet" \
    "fer:$OUTPUT_ROOT/fer/logs_gated.parquet" \
    "lv:$OUTPUT_ROOT/lv/logs_gated.parquet" \
  2>&1 | tee "$OUTPUT_ROOT/pcm_backtest.log"

echo "=== $(date) === Step 5b: PCM KPI Evaluation ==="
python scripts/evaluate_pcm_allocation.py \
  --pcm-report "$OUTPUT_ROOT/pcm_backtest_report.csv" \
  --output-dir "$OUTPUT_ROOT/pcm_kpi" \
  2>&1 | tee "$OUTPUT_ROOT/pcm_kpi_eval.log"

echo ""
echo "================================================================"
echo "  $(date) - ALL TRAINING COMPLETE!"
echo "  Results: $OUTPUT_ROOT"
echo "================================================================"
