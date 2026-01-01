#!/bin/bash
# EXP 002: Extended training (2023-2024, more epochs, tuned hyperparams)
set -e
LOG_FILE="/workspaces/ml_trading_bot/results/exp002_log.txt"
RESULT_DIR="/workspaces/ml_trading_bot/results/exp002_btc_eth_2023_2024"
LAYER="features_291404fba6"

rm -rf "$RESULT_DIR"
mkdir -p "$RESULT_DIR"
echo "=== EXP 002: Extended Training ===" | tee "$LOG_FILE"
echo "Started at $(date)" | tee -a "$LOG_FILE"

cd /workspaces/ml_trading_bot

# Step 1: Build FeatureStore for 2023 (if needed)
echo "[1/5] Building FeatureStore (2023)..." | tee -a "$LOG_FILE"
python3 scripts/build_feature_store_from_config.py \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --timeframe 240T \
  --data-path data/parquet_data \
  --root feature_store \
  --layer "$LAYER" \
  --symbols BTCUSDT,ETHUSDT \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 2>&1 | tee -a "$LOG_FILE"

echo "[1/5] FeatureStore DONE" | tee -a "$LOG_FILE"

# Step 2: Train Model (extended: 2023-01 ~ 2024-06, more epochs, adjusted params)
echo "[2/5] Training (2023-01 ~ 2024-06, 50 epochs, hidden=512)..." | tee -a "$LOG_FILE"
python3 scripts/train_path_primitives_mlp.py \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --data-path data/parquet_data \
  --timeframe 240T \
  --start-date 2023-01-01 \
  --end-date 2024-06-30 \
  --epochs 50 \
  --hidden 512 \
  --depth 3 \
  --dropout 0.2 \
  --lr 0.0001 \
  --output-dir "$RESULT_DIR" \
  --features-store-root feature_store \
  --features-store-layer "$LAYER" 2>&1 | tee -a "$LOG_FILE"

MODEL_PATH=$(find "$RESULT_DIR" -name "model.pt" | head -1)
echo "Model: $MODEL_PATH" | tee -a "$LOG_FILE"

# Step 3: OOS Predict (2024-07 ~ 2024-12)
echo "[3/5] Predicting OOS (2024-07 ~ 2024-12)..." | tee -a "$LOG_FILE"
python3 scripts/predict_path_primitives_mlp.py \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2024-07-01 \
  --end-date 2024-12-31 \
  --model "$MODEL_PATH" \
  --output "$RESULT_DIR/preds" \
  --features-store-root feature_store \
  --features-store-layer "$LAYER" 2>&1 | tee -a "$LOG_FILE"

# Step 4: Rule Router + Logs
echo "[4/5] Building logs..." | tee -a "$LOG_FILE"
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

# Step 5: Backtest
echo "[5/5] Backtesting..." | tee -a "$LOG_FILE"
mkdir -p "$RESULT_DIR/rl_e2e"
python3 scripts/rl_counterfactual_eval_3action.py \
  --logs "$RESULT_DIR/logs_3action.parquet" \
  --out "$RESULT_DIR/rl_e2e/counterfactual" \
  --train_ratio 0.0 2>&1 | tee -a "$LOG_FILE"

# Results
echo "" | tee -a "$LOG_FILE"
echo "=== TRAINING METRICS ===" | tee -a "$LOG_FILE"
python3 -c "
import json
from pathlib import Path
model_dir = list(Path('$RESULT_DIR').glob('path_primitives_*'))[0]
m = json.load(open(model_dir / 'metrics.json'))
print(f'Dir Acc:       {m.get(\"dir_acc\", 0):.4f}')
print(f'Dir AUC:       {m.get(\"dir_auc\", 0):.4f}')
print(f'MFE Spearman:  {m.get(\"mfe_atr_spearman\", 0):.4f}')
print(f'MAE Spearman:  {m.get(\"mae_atr_spearman\", 0):.4f}')
" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== BACKTEST RESULTS (OOS: 2024-07 ~ 2024-12) ===" | tee -a "$LOG_FILE"
python3 << 'PYEOF' 2>&1 | tee -a "$LOG_FILE"
import pandas as pd
import numpy as np

for sym in ["BTCUSDT", "ETHUSDT"]:
    df = pd.read_parquet(f"$RESULT_DIR/preds/preds_{sym}.parquet")
    df['signal'] = 0
    df.loc[df['pred_dir_prob'] > 0.55, 'signal'] = 1
    df.loc[df['pred_dir_prob'] < 0.45, 'signal'] = -1
    df['returns'] = df['close'].pct_change()
    df['strat_ret'] = df['signal'].shift(1) * df['returns']
    
    ret = df['strat_ret'].dropna()
    sharpe = (ret.mean() / ret.std()) * np.sqrt(252 * 6) if ret.std() > 0 else 0
    total = (1 + ret).prod() - 1
    
    print(f"{sym}: Sharpe={sharpe:.4f}, Total={total:.2%}, Signals={df['signal'].value_counts().to_dict()}")
PYEOF

echo "" | tee -a "$LOG_FILE"
echo "DONE at $(date)" | tee -a "$LOG_FILE"

