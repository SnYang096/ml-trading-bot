#!/bin/bash
# Full E2E Pipeline for NN Multihead Model
# Log file: /workspaces/ml_trading_bot/results/pipeline_log.txt

set -e  # Exit on error

LOG_FILE="/workspaces/ml_trading_bot/results/pipeline_log.txt"
RESULT_DIR="/workspaces/ml_trading_bot/results/real_btc_eth_2024"

mkdir -p "$RESULT_DIR"
echo "========================================" | tee -a "$LOG_FILE"
echo "Pipeline started at $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

cd /workspaces/ml_trading_bot

# Step 1: Build FeatureStore
echo "" | tee -a "$LOG_FILE"
echo "[STEP 1/6] Building FeatureStore (BTC+ETH, 2024)..." | tee -a "$LOG_FILE"
echo "Started at $(date)" | tee -a "$LOG_FILE"

python3 scripts/build_feature_store_from_config.py \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --timeframe 240T \
  --data-path data/parquet_data \
  --root feature_store \
  --symbols BTCUSDT,ETHUSDT \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 2>&1 | tee -a "$LOG_FILE"

echo "[STEP 1/6] FeatureStore DONE at $(date)" | tee -a "$LOG_FILE"

# Step 2: Train Model
echo "" | tee -a "$LOG_FILE"
echo "[STEP 2/6] Training NN Multihead Model..." | tee -a "$LOG_FILE"
echo "Started at $(date)" | tee -a "$LOG_FILE"

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

echo "[STEP 2/6] Training DONE at $(date)" | tee -a "$LOG_FILE"

# Find model path
MODEL_PATH=$(find "$RESULT_DIR" -name "model.pt" | head -1)
echo "Model path: $MODEL_PATH" | tee -a "$LOG_FILE"

# Step 3: OOS Prediction
echo "" | tee -a "$LOG_FILE"
echo "[STEP 3/6] Running OOS Predictions (2024-10 ~ 2024-12)..." | tee -a "$LOG_FILE"
echo "Started at $(date)" | tee -a "$LOG_FILE"

python3 scripts/predict_path_primitives_mlp.py \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2024-10-01 \
  --end-date 2024-12-31 \
  --model "$MODEL_PATH" \
  --output "$RESULT_DIR/preds" \
  --features-store-root feature_store 2>&1 | tee -a "$LOG_FILE"

echo "[STEP 3/6] Prediction DONE at $(date)" | tee -a "$LOG_FILE"

# Step 4: Rule Router Mode
echo "" | tee -a "$LOG_FILE"
echo "[STEP 4/6] Running Rule Router (mode-3action)..." | tee -a "$LOG_FILE"
echo "Started at $(date)" | tee -a "$LOG_FILE"

python3 scripts/rule_mode_3action.py \
  --preds "$RESULT_DIR/preds" \
  --output "$RESULT_DIR/mode_3action.parquet" 2>&1 | tee -a "$LOG_FILE"

echo "[STEP 4/6] Rule Router DONE at $(date)" | tee -a "$LOG_FILE"

# Step 5: Build RL Logs
echo "" | tee -a "$LOG_FILE"
echo "[STEP 5/6] Building RL Logs..." | tee -a "$LOG_FILE"
echo "Started at $(date)" | tee -a "$LOG_FILE"

python3 scripts/rl_build_logs_3action.py \
  --preds "$RESULT_DIR/preds" \
  --mode "$RESULT_DIR/mode_3action.parquet" \
  --data-path data/parquet_data \
  --timeframe 240T \
  --output "$RESULT_DIR/logs_3action.parquet" \
  --returns-source momentum_proxy 2>&1 | tee -a "$LOG_FILE"

echo "[STEP 5/6] RL Logs DONE at $(date)" | tee -a "$LOG_FILE"

# Step 6: E2E Evaluation
echo "" | tee -a "$LOG_FILE"
echo "[STEP 6/6] Running E2E Evaluation (Shadow + Counterfactual + FSM)..." | tee -a "$LOG_FILE"
echo "Started at $(date)" | tee -a "$LOG_FILE"

mkdir -p "$RESULT_DIR/rl_e2e"

# Shadow eval
python3 scripts/rl_shadow_eval_3action.py \
  --logs "$RESULT_DIR/logs_3action.parquet" \
  --out "$RESULT_DIR/rl_e2e/shadow" \
  --train_ratio 0.7 2>&1 | tee -a "$LOG_FILE"

# Counterfactual eval
python3 scripts/rl_counterfactual_eval_3action.py \
  --logs "$RESULT_DIR/logs_3action.parquet" \
  --out "$RESULT_DIR/rl_e2e/counterfactual" \
  --train_ratio 0.7 2>&1 | tee -a "$LOG_FILE"

# FSM decide
python3 scripts/rl_fsm_decide.py \
  --metrics "$RESULT_DIR/rl_e2e/counterfactual/metrics.json" \
  --state RL_CANDIDATE \
  --out "$RESULT_DIR/rl_e2e/fsm_decision.json" 2>&1 | tee -a "$LOG_FILE"

echo "[STEP 6/6] E2E Evaluation DONE at $(date)" | tee -a "$LOG_FILE"

# Summary
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Pipeline COMPLETED at $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results:" | tee -a "$LOG_FILE"
echo "  Model: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "  Predictions: $RESULT_DIR/preds/" | tee -a "$LOG_FILE"
echo "  Metrics: $RESULT_DIR/rl_e2e/counterfactual/metrics.json" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Print key metrics
if [ -f "$RESULT_DIR/rl_e2e/counterfactual/metrics.json" ]; then
    echo "=== Key Backtest Metrics ===" | tee -a "$LOG_FILE"
    python3 -c "
import json
with open('$RESULT_DIR/rl_e2e/counterfactual/metrics.json') as f:
    m = json.load(f)
print(f\"  Sharpe (pred): {m.get('pred_sharpe_mean', 'N/A'):.4f}\")
print(f\"  Sortino (pred): {m.get('pred_sortino_mean', 'N/A'):.4f}\")
print(f\"  Ann. Return (pred): {m.get('pred_ann_return_mean', 'N/A'):.4f}\")
print(f\"  Ann. Vol (pred): {m.get('pred_ann_vol_mean', 'N/A'):.4f}\")
print(f\"  Max DD (pred): {m.get('pred_avg_max_dd', 'N/A'):.4f}\")
print(f\"  Total Return (pred): {m.get('pred_avg_total_return', 'N/A'):.4f}\")
" 2>&1 | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "PIPELINE_STATUS=SUCCESS" | tee -a "$LOG_FILE"

