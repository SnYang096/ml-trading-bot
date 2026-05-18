#!/bin/bash
# spot_accum_simple 实盘启动脚本（spot-only 进程）

set -e

UNIVERSE="${1:-highcap}"
LIVE_ROOT="live/${UNIVERSE}"

echo "============================================================"
echo "🚀 spot_accum_simple live 启动"
echo "============================================================"
echo "Universe: $UNIVERSE"
echo "Live Root: $LIVE_ROOT"
echo ""

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 优先加载 spot 专用 key；不存在则回退到通用 env（便于 CI 注入）
if [ -f "live/binance_spot_mainnet.env" ]; then
  echo "📝 加载 Spot API 密钥..."
  set -a
  source live/binance_spot_mainnet.env
  set +a
fi

# Feature bus 只读依赖（由 quant-feature-bus 提供）
FEATURE_BUS_ROOT="${MLBOT_FEATURE_BUS_ROOT:-live/shared_feature_bus}"
if [ ! -d "$FEATURE_BUS_ROOT" ]; then
  echo "❌ Feature Bus 目录不存在: $FEATURE_BUS_ROOT"
  echo "   请先启动 quant-feature-bus publisher"
  exit 1
fi

export MLBOT_FEATURE_BUS_ROOT="$FEATURE_BUS_ROOT"
export MLBOT_SPOT_STRATEGY="${MLBOT_SPOT_STRATEGY:-spot_accum_simple}"
export MLBOT_SPOT_STRATEGIES_ROOT="${MLBOT_SPOT_STRATEGIES_ROOT:-$LIVE_ROOT/config/strategies}"
export MLBOT_SPOT_CONSTITUTION_YAML="${MLBOT_SPOT_CONSTITUTION_YAML:-$LIVE_ROOT/config/constitution/constitution.yaml}"
export MLBOT_SPOT_DB_PATH="${MLBOT_SPOT_DB_PATH:-$LIVE_ROOT/data/spot_order_management.db}"
export MLBOT_SPOT_LEDGER_DB_PATH="${MLBOT_SPOT_LEDGER_DB_PATH:-$LIVE_ROOT/data/spot_accum_ledger.db}"
export MLBOT_SPOT_ORDER_MANAGER_ENABLED="${MLBOT_SPOT_ORDER_MANAGER_ENABLED:-true}"
export MLBOT_SPOT_SHADOW_MODE="${MLBOT_SPOT_SHADOW_MODE:-false}"
export MLBOT_SPOT_FEATURE_BUS_POLL_SECONDS="${MLBOT_SPOT_FEATURE_BUS_POLL_SECONDS:-5}"
export MLBOT_METRICS_PORT="${MLBOT_METRICS_PORT:-9193}"

if [ "$MLBOT_SPOT_SHADOW_MODE" = "false" ] && { [ -z "$BINANCE_SPOT_API_KEY" ] || [ -z "$BINANCE_SPOT_API_SECRET" ]; }; then
  echo "❌ 实盘模式需要 BINANCE_SPOT_API_KEY / BINANCE_SPOT_API_SECRET"
  exit 1
fi

echo "✅ 配置完成: strategy=$MLBOT_SPOT_STRATEGY shadow=$MLBOT_SPOT_SHADOW_MODE metrics_port=$MLBOT_METRICS_PORT"
echo ""

python scripts/run_spot_accum_live.py
