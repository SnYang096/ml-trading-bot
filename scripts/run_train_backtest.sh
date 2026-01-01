#!/bin/bash
# Quick pipeline: Train + Predict + Backtest (FeatureStore already built)
set -e
LOG_FILE="/workspaces/ml_trading_bot/results/train_backtest_log.txt"
RESULT_DIR="/workspaces/ml_trading_bot/results/real_btc_eth_2024"

rm -rf "$RESULT_DIR"
mkdir -p "$RESULT_DIR"
echo "Started at $(date)" | tee "$LOG_FILE"

cd /workspaces/ml_trading_bot

# Step 1: Train Model
echo "[1/4] Training..." | tee -a "$LOG_FILE"
python3 scripts/train_path_primitives_mlp.py \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --data-path data/parquet_data \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2024-09-30 \
  --epochs 30 \
  --output-dir "$RESULT_DIR" \
  --features-store-root feature_store 2>&1 | tee -a "$LOG_FILE"

MODEL_PATH=$(find "$RESULT_DIR" -name "model.pt" | head -1)
echo "Model: $MODEL_PATH" | tee -a "$LOG_FILE"

# Step 2: Predict
echo "[2/4] Predicting..." | tee -a "$LOG_FILE"
python3 scripts/predict_path_primitives_mlp.py \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2024-10-01 \
  --end-date 2024-12-31 \
  --model "$MODEL_PATH" \
  --output "$RESULT_DIR/preds" \
  --features-store-root feature_store 2>&1 | tee -a "$LOG_FILE"

# Step 3: Rule Router + Logs
echo "[3/4] Building logs..." | tee -a "$LOG_FILE"
python3 scripts/rule_mode_3action.py \
  --preds "$RESULT_DIR/preds" \
  --output "$RESULT_DIR/mode_3action.parquet" 2>&1 | tee -a "$LOG_FILE"

python3 scripts/rl_build_logs_3action.py \
  --preds "$RESULT_DIR/preds" \
  --mode "$RESULT_DIR/mode_3action.parquet" \
  --data-path data/parquet_data \
  --timeframe 240T \
  --output "$RESULT_DIR/logs_3action.parquet" \
  --returns-source momentum_proxy 2>&1 | tee -a "$LOG_FILE"

# Step 4: Backtest
echo "[4/4] Backtesting..." | tee -a "$LOG_FILE"
mkdir -p "$RESULT_DIR/rl_e2e"
python3 scripts/rl_counterfactual_eval_3action.py \
  --logs "$RESULT_DIR/logs_3action.parquet" \
  --out "$RESULT_DIR/rl_e2e/counterfactual" \
  --train_ratio 0.0 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== RESULTS ===" | tee -a "$LOG_FILE"
python3 -c "
import json
with open('$RESULT_DIR/rl_e2e/counterfactual/metrics.json') as f:
    m = json.load(f)
print(f'Sharpe:      {m.get(\"pred_sharpe_mean\", 0):.4f}')
print(f'Sortino:     {m.get(\"pred_sortino_mean\", 0):.4f}')
print(f'Ann Return:  {m.get(\"pred_ann_return_mean\", 0):.2%}')
print(f'Max DD:      {m.get(\"pred_avg_max_dd\", 0):.2%}')
print(f'Total Ret:   {m.get(\"pred_avg_total_return\", 0):.2%}')
" 2>&1 | tee -a "$LOG_FILE"

echo "DONE at $(date)" | tee -a "$LOG_FILE"

