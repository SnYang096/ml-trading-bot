#!/bin/bash
# 全量数据下载脚本：OI + Funding Rate
# 用法: nohup bash scripts/download_all_data.sh > /tmp/download_all.log 2>&1 &
set -e
cd "$(dirname "$0")/.."

UNIVERSE="config/download/crypto_4h_token_universe_groups.yaml"

# 从 universe config 中提取所有 symbols
SYMBOLS=$(python -c "
from src.data_tools.universe_config import load_universe_config
cfg = load_universe_config('$UNIVERSE')
syms = cfg.resolve_symbols_usdt(universe_set='starter_a')
print(' '.join(syms))
")

echo "=== $(date) === Starting data download ==="
echo "Symbols: $SYMBOLS"
echo ""

# Step 1: OI 下载 (via Binance Data Vision, 5m 精度)
echo "=== $(date) === Step 1: OI download (Data Vision, 5m) ==="
python -u scripts/download_oi_from_data_vision.py \
  --universe-config $UNIVERSE \
  --universe-set starter_a \
  --start-date 2023-01-01 \
  --parquet-dir data/open_interest/parquet \
  --progress-every 100 \
  --sleep-sec 0.1

echo ""

# Step 2: Funding Rate 下载
echo "=== $(date) === Step 2: Funding Rate download ==="
python -u src/data_tools/download_funding_rate.py \
  --symbols $SYMBOLS \
  --start-year 2023 --start-month 1 \
  --parquet-dir data/funding_rate/parquet \
  --progress-every 25 \
  --sleep-sec 0.25

echo ""
echo "=== $(date) === All downloads complete ==="
