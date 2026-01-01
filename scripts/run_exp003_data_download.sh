#!/bin/bash
# EXP003: 下载 20 tokens 的 4 年数据 (2021-2024)

set -e

echo "=========================================="
echo "EXP003: 下载训练数据"
echo "=========================================="

# 20 tokens 列表 (HighCap/Alt/Meme 混合)
TOKENS=(
  # HighCap (8个)
  "BTC" "ETH" "BNB" "SOL" "XRP" "ADA" "AVAX" "LTC"
  # Alt (7个)
  "LINK" "DOT" "ATOM" "NEAR" "UNI" "AAVE" "FTM"
  # Meme (5个)
  "DOGE" "SHIB" "PEPE" "WIF" "FLOKI"
)

SYMBOLS=""
for t in "${TOKENS[@]}"; do
  SYMBOLS+="${t}USDT,"
done
SYMBOLS=${SYMBOLS%,}  # 移除最后的逗号

echo "Tokens: $SYMBOLS"
echo "Date range: 2021-01-01 to 2024-12-31"
echo ""

# 使用 mlbot data pipeline-universe 下载
mlbot data download \
  --symbols "$SYMBOLS" \
  --start-date 2021-01-01 \
  --end-date 2024-12-31 \
  --no-docker

echo ""
echo "下载完成！"
