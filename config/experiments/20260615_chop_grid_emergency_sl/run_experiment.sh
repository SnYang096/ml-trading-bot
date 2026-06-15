#!/usr/bin/env bash
# chop_grid Emergency SL backtest runner
# Usage: ./run_experiment.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_DIR="${REPO_ROOT}/results/chop_grid/experiments/emergency_sl_20260615"
mkdir -p "${EXPERIMENT_DIR}"

SYMBOLS="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT"
START="2022-01-01"
END="2026-03-31"
TIMEFRAME="2h"
EXEC_TF="1min"
CONFIG="config/strategies/chop_grid/research/calibrate_roll.default.yaml"

echo "=== chop_grid Emergency SL Experiment ==="
echo "Symbols: ${SYMBOLS}"
echo "Period:  ${START} to ${END}"
echo "Output:  ${EXPERIMENT_DIR}"
echo ""

# Variant 1: Baseline (no SL)
echo "[1/4] Running baseline (no SL)..."
PYTHONPATH=src:scripts python -m scripts.chop_grid_backtest \
  --config "${CONFIG}" \
  --symbols "${SYMBOLS}" \
  --start "${START}" \
  --end "${END}" \
  --timeframe "${TIMEFRAME}" \
  --execution-timeframe "${EXEC_TF}" \
  --no-per-leg-stop-loss \
  --out-dir "${EXPERIMENT_DIR}/baseline_no_sl" \
  >"${EXPERIMENT_DIR}/baseline_no_sl.log" 2>&1

# Variant 2: Weak SL (spacing_mult=4)
echo "[2/4] Running weak SL (spacing_mult=4)..."
PYTHONPATH=src:scripts python -m scripts.chop_grid_backtest \
  --config "${CONFIG}" \
  --symbols "${SYMBOLS}" \
  --start "${START}" \
  --end "${END}" \
  --timeframe "${TIMEFRAME}" \
  --execution-timeframe "${EXEC_TF}" \
  --per-leg-stop-loss \
  --per-leg-sl-spacing-mult 4.0 \
  --out-dir "${EXPERIMENT_DIR}/sl_4x" \
  >"${EXPERIMENT_DIR}/sl_4x.log" 2>&1

# Variant 3: Medium SL (spacing_mult=6)
echo "[3/4] Running medium SL (spacing_mult=6)..."
PYTHONPATH=src:scripts python -m scripts.chop_grid_backtest \
  --config "${CONFIG}" \
  --symbols "${SYMBOLS}" \
  --start "${START}" \
  --end "${END}" \
  --timeframe "${TIMEFRAME}" \
  --execution-timeframe "${EXEC_TF}" \
  --per-leg-stop-loss \
  --per-leg-sl-spacing-mult 6.0 \
  --out-dir "${EXPERIMENT_DIR}/sl_6x" \
  >"${EXPERIMENT_DIR}/sl_6x.log" 2>&1

# Variant 4: Strong SL (spacing_mult=8)
echo "[4/4] Running strong SL (spacing_mult=8)..."
PYTHONPATH=src:scripts python -m scripts.chop_grid_backtest \
  --config "${CONFIG}" \
  --symbols "${SYMBOLS}" \
  --start "${START}" \
  --end "${END}" \
  --timeframe "${TIMEFRAME}" \
  --execution-timeframe "${EXEC_TF}" \
  --per-leg-stop-loss \
  --per-leg-sl-spacing-mult 8.0 \
  --out-dir "${EXPERIMENT_DIR}/sl_8x" \
  >"${EXPERIMENT_DIR}/sl_8x.log" 2>&1

echo ""
echo "=== Experiment Complete ==="
echo "Results: ${EXPERIMENT_DIR}"
