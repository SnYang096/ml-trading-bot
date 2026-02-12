#!/bin/bash
# 本地实盘测试启动脚本
# 使用DEGRADED模式（观察不交易）

set -e

# 设置PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "============================================================"
echo "🧪 本地实盘测试部署 - DEGRADED模式（只观察不交易）"
echo "============================================================"

# 配置环境变量
export MLBOT_LIVE_SYMBOLS="BTCUSDT"
export MLBOT_LIVE_STORAGE_BASE="data/live_storage_test"
export MLBOT_LIVE_WARMUP_DAYS="30"
export MLBOT_LIVE_TRADE_SIZE="0.0"  # 0表示不交易
export MLBOT_LIVE_USE_FUTURES="true"
export MLBOT_LIVE_GAP_FILL="true"

# BPC策略配置
export MLBOT_STRATEGIES_ROOT="config/strategies"
export MLBOT_BPC_FEATURE_PLAN_YAML="config/live/live_feature_plan.yaml"
export MLBOT_BPC_BAR_MINUTES="240"  # 4小时
export MLBOT_BPC_WINDOW_MINUTES="15"  # 15分钟

# 特征存储（可选）
export MLBOT_FEATURE_STORE_DIR="feature_store"
export MLBOT_FEATURE_STORE_LAYER=""

# 订单管理器配置（测试模式）
export MLBOT_ORDER_MODE="test"  # test/paper/live

echo ""
echo "📋 配置信息:"
echo "   Symbol: $MLBOT_LIVE_SYMBOLS"
echo "   Storage: $MLBOT_LIVE_STORAGE_BASE"
echo "   Warmup: $MLBOT_LIVE_WARMUP_DAYS days"
echo "   Trade Size: $MLBOT_LIVE_TRADE_SIZE (0=只观察)"
echo "   Bar: ${MLBOT_BPC_BAR_MINUTES}min, Window: ${MLBOT_BPC_WINDOW_MINUTES}min"
echo "   Order Mode: $MLBOT_ORDER_MODE"
echo ""

# 创建存储目录
mkdir -p "$MLBOT_LIVE_STORAGE_BASE"

echo "🚀 启动实盘系统..."
echo ""

# 运行
python scripts/run_live.py
