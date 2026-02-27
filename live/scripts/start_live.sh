#!/bin/bash
# 实盘启动脚本（带依赖检查，支持 universe）

set -e

UNIVERSE="${1:-highcap}"           # 第一个参数：universe 名称（默认 highcap）
SYMBOLS_ARG="${2:-}"               # 第二个参数：可选，手动指定 symbol 列表

LIVE_ROOT="live/${UNIVERSE}"
export LIVE_ROOT  # 提前导出，供后续 Python 脚本使用

echo "============================================================"
echo "🚀 实盘系统启动"
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
else
  echo "   ⚠️  未找到 live/binance_mainnet.env，将使用系统环境变量中的 API 密钥"
fi
echo ""

# 从 universe.yaml 加载默认 symbols（若未通过参数指定）
if [ -z "$SYMBOLS_ARG" ]; then
  if [ -f "$LIVE_ROOT/universe.yaml" ]; then
    SYMBOLS=$(python - << 'PY'
import yaml
from pathlib import Path
import os

live_root = os.environ.get("LIVE_ROOT")
path = Path(live_root) / "universe.yaml"
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
syms = list((cfg.get("symbols") or {}).keys())
print(",".join(sorted(syms)))
PY
    )
  else
    SYMBOLS="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT"
  fi
else
  SYMBOLS="$SYMBOLS_ARG"
fi

# 1. Warmup 数据检查（仅补充 daily 数据，不做完整下载）
echo "📦 第1步：Warmup 数据检查..."
TICKS_DIR="$LIVE_ROOT/data/ticks"
BARS_DIR="$LIVE_ROOT/data/bars"
if [ -d "$TICKS_DIR" ] && [ "$(ls -A $TICKS_DIR 2>/dev/null)" ]; then
  echo "   ✅ 已有 ticks warmup 数据，补充缺失的 daily 数据..."
  bash live/scripts/prepare_warmup_ticks.sh "$UNIVERSE" 6 --fill-gap
else
  echo "   ⚠️  未找到 warmup 数据！请先执行以下命令之一："
  echo "      方式 A: 本地下载后 rsync 上传"
  echo "        bash live/scripts/prepare_warmup_ticks.sh $UNIVERSE 6"
  echo "        rsync -avz live/highcap/data/ ubuntu@SERVER:/opt/quant-engine/live/highcap/data/"
  echo "      方式 B: 服务器上独立执行"
  echo "        docker run --rm -v /opt/quant-engine/live/highcap/data:/app/live/highcap/data quant-engine:latest bash live/scripts/prepare_warmup_ticks.sh highcap 6"
  echo ""
  echo "   ❌ Warmup 数据缺失，启动中止！"
  exit 1
fi

# 检查 bars 目录（bars 是特征计算的必要数据）
if [ -d "$BARS_DIR" ]; then
  BARS_FILES=$(find "$BARS_DIR" -name "*.parquet" 2>/dev/null | wc -l)
  echo "   📊 bars 目录: $BARS_FILES 个 parquet 文件"
  if [ "$BARS_FILES" -lt 100 ]; then
    echo "   ⚠️  bars 数据不足 ($BARS_FILES 个文件，需要 100+)！"
    echo "      特征计算需要 150+ 天 1min bars。请上传历史 bars 数据："
    echo "      rsync -avz live/highcap/data/bars/ remote:$LIVE_ROOT/data/bars/"
    echo "   ❌ bars warmup 数据不足，启动中止！"
    exit 1
  fi
else
  echo "   ❌ bars 目录不存在: $BARS_DIR"
  echo "      请先运行 warmup 或上传 bars 数据"
  exit 1
fi
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
export MLBOT_LIVE_WARMUP_DAYS="30"
export MLBOT_LIVE_TRADE_SIZE="0.001"  # 最小开仓量 fallback（风险反算 qty 太小时使用）
# risk_per_slot 已经在 constitution.yaml 中配置 (slots.risk_per_slot = 0.01 = 1%)
# MLBOT_RISK_PER_TRADE 作为备用 fallback（无 equity 时用固定美元）
export MLBOT_RISK_PER_TRADE="${MLBOT_RISK_PER_TRADE:-10.0}"
export MLBOT_LIVE_USE_FUTURES="true"
export MLBOT_LIVE_GAP_FILL="true"

# 策略配置（使用live目录）
export MLBOT_STRATEGIES_ROOT="$LIVE_ROOT/config/strategies"
export MLBOT_BPC_WINDOW_MINUTES="15"  # 15分钟

# PCM / Constitution 配置（全局配置在 live/highcap/config/ 下）
export MLBOT_PCM_REGIME_CONFIG="$LIVE_ROOT/config/pcm_regime.yaml"
export MLBOT_CONSTITUTION_YAML="$LIVE_ROOT/config/constitution/constitution.yaml"

# 启动模式: bpc (单策略) 或 three_strategies (三策略多时间框架)
export MLBOT_LIVE_MODE="${MLBOT_LIVE_MODE:-three_strategies}"

# 策略B：live 不再依赖 Feature Store，所有特征基于 ticks/bars 实时重算
# export MLBOT_FEATURE_STORE_DIR="$LIVE_ROOT/feature_store"  # 已废弃
# export MLBOT_FEATURE_STORE_LAYER="bpc_live_240T"           # 已废弃

# 订单管理器配置
export MLBOT_ORDER_MODE="test"  # test/paper/live
export MLBOT_ORDER_MANAGER_ENABLED="true"  # 启用 OrderManager（需要 BINANCE_API_KEY）

echo "   ✅ 环境变量已配置"
echo ""

# 4. 启动实盘系统
echo "🚀 第4步：启动实盘系统..."
echo ""

python scripts/run_live.py
