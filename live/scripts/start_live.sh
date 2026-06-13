#!/bin/bash
# 实盘启动脚本（带依赖检查，支持 universe）

set -e

UNIVERSE="${1:-highcap}"           # 第一个参数：universe 名称（默认 highcap）
SYMBOLS_ARG="${2:-}"               # 第二个参数：可选，手动指定 symbol 列表

LIVE_ROOT="live/${UNIVERSE}"
export LIVE_ROOT  # 提前导出，供后续 Python 脚本使用

echo "============================================================"
echo "🚀 Directional Trend / Fat-tail consumer 启动"
echo "============================================================"
echo "Universe: $UNIVERSE"
echo "Live Root: $LIVE_ROOT"
echo ""

# 设置PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 加载 Binance API 密钥（从 live/binance_mainnet.env）
if [ -f "live/binance_mainnet.env" ]; then
  echo "📝 加载 Binance API 密钥..."
  set -a  # 自动导出所有变量
  source live/binance_mainnet.env
  set +a
  echo "   ✅ API 密钥已加载"
  # 打印 Key（不打印 Secret）便于确认 CICD 注入是否正确
  echo "   🔑 BINANCE_API_KEY=${BINANCE_API_KEY:0:8}...${BINANCE_API_KEY: -4} (len=${#BINANCE_API_KEY})"
else
  echo "   ⚠️  未找到 live/binance_mainnet.env，将使用系统环境变量中的 API 密钥"
fi
# 最终确认 Key 来源（环境变量可能来自 env 文件或 CICD 注入）
if [ -n "$BINANCE_API_KEY" ]; then
  echo "   🔑 最终 API_KEY=${BINANCE_API_KEY:0:8}...${BINANCE_API_KEY: -4} (len=${#BINANCE_API_KEY})"
else
  echo "   ❌ BINANCE_API_KEY 未设置！实盘下单将失败"
fi
echo ""

# 从 universe.yaml 加载 bus 全集（策略 meta 在 run_live.py 内再过滤）
if [ -z "$SYMBOLS_ARG" ]; then
  SYMBOLS=$(python - << 'PY'
from src.live_data_stream.universe_symbols import resolve_symbols_csv
import os

live_root = os.environ.get("LIVE_ROOT", "live/highcap")
universe = live_root.rsplit("/", 1)[-1]
print(resolve_symbols_csv(universe=universe, env_symbols=""))
PY
  )
else
  SYMBOLS="$SYMBOLS_ARG"
fi

# 1. Feature Bus 数据路径检查：行情 WebSocket 只在 quant-feature-bus 进程内运行。
echo "📦 第1步：Feature Bus 数据路径检查..."
FEATURE_SOURCE="${MLBOT_FEATURE_SOURCE:-bus}"
if [ "$FEATURE_SOURCE" != "bus" ] && [ "$FEATURE_SOURCE" != "feature-bus" ] && [ "$FEATURE_SOURCE" != "feature-store" ]; then
  echo "   ❌ MLBOT_FEATURE_SOURCE=$FEATURE_SOURCE 已不支持；趋势消费者只允许 bus / feature-bus / feature-store"
  exit 1
fi
FEATURE_BUS_ROOT="${MLBOT_FEATURE_BUS_ROOT:-live/shared_feature_bus}"
if [ ! -d "$FEATURE_BUS_ROOT" ]; then
  echo "   ❌ Feature Bus 目录不存在: $FEATURE_BUS_ROOT"
  echo "      请先启动 quant-feature-bus publisher。"
  exit 1
fi
echo "   ✅ Feature Bus: $FEATURE_BUS_ROOT"
echo ""

# 2. 依赖自检
echo "🔍 第2步：依赖自检..."
python "live/scripts/check_dependencies.py" --symbols "$SYMBOLS" --live-root "$LIVE_ROOT"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 依赖检查失败，启动中止！"
    exit 1
fi

echo ""
echo "✅ 依赖检查通过"
echo "Symbol(s): $SYMBOLS"
echo ""

# 3. 配置环境变量
echo "⚙️  第3步：配置环境变量..."

export MLBOT_LIVE_SYMBOLS="$SYMBOLS"
export MLBOT_LIVE_STORAGE_BASE="$LIVE_ROOT/data"
export MLBOT_LIVE_WARMUP_DAYS="${MLBOT_LIVE_WARMUP_DAYS:-0}"
export MLBOT_LIVE_TRADE_SIZE="0.001"  # 最小开仓量 fallback（风险反算 qty 太小时使用）
# risk_per_slot 已经在 constitution.yaml 中配置 (slots.risk_per_slot = 0.01 = 1%)
# MLBOT_RISK_PER_TRADE 作为备用 fallback（无 equity 时用固定美元）
export MLBOT_RISK_PER_TRADE="${MLBOT_RISK_PER_TRADE:-10.0}"
export MLBOT_LIVE_USE_FUTURES="true"
export MLBOT_LIVE_GAP_FILL="true"

# 策略配置（使用live目录）
export MLBOT_STRATEGIES_ROOT="$LIVE_ROOT/config/strategies"
export MLBOT_BPC_WINDOW_MINUTES="15"  # 15分钟

# Constitution 配置（全局配置在 live/highcap/config/ 下；含 multi_leg 节）
export MLBOT_CONSTITUTION_YAML="$LIVE_ROOT/config/constitution/constitution.yaml"

# 唯一数据路径：quant-feature-bus 写盘，本进程只读 Feature Bus
export MLBOT_FEATURE_SOURCE="${MLBOT_FEATURE_SOURCE:-bus}"

# 订单管理器配置
export MLBOT_ORDER_MODE="test"  # test/paper/live
export MLBOT_ORDER_MANAGER_ENABLED="true"  # 启用 OrderManager（需要 BINANCE_API_KEY）

echo "   ✅ 环境变量已配置"
echo ""

# 4. 刷新 Funding Rate / OI 数据 (增量, 从 Binance API 下载最近60天)
echo "📊 第4步：刷新 Funding Rate / OI 数据..."
python scripts/refresh_funding_oi_data.py --symbols "$SYMBOLS" --lookback-days 60 || {
  echo "   ⚠️  Funding/OI 刷新失败（非致命，使用已有历史数据继续）"
}
echo ""

# 5. 启动实盘系统
echo "🚀 第5步：启动实盘系统..."
echo ""

python scripts/run_live.py
