#!/bin/bash
# 实盘启动依赖检查脚本

set -e

UNIVERSE="${1:-highcap}"
LIVE_ROOT="live/${UNIVERSE}"

echo "============================================================"
echo "🔍 实盘启动依赖检查"
echo "============================================================"
echo "Universe: $UNIVERSE"
echo ""

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LIVE_ROOT

# 检查计数器
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

check_item() {
    local desc="$1"
    local condition="$2"
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if eval "$condition"; then
        echo "✅ $desc"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        echo "❌ $desc"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi
}

echo "📋 1. 目录结构检查"
echo "-----------------------------------------------------------"
check_item "universe.yaml 存在" "[ -f ${LIVE_ROOT}/universe.yaml ]"
check_item "config 目录存在" "[ -d ${LIVE_ROOT}/config ]"
echo ""

echo "📦 2. feature_store 检查"
echo "-----------------------------------------------------------"
check_item "feature_store 目录存在" "[ -d ${LIVE_ROOT}/feature_store ]"

if [ -d "${LIVE_ROOT}/feature_store/bpc_live_240T" ]; then
    SYMBOL_COUNT=$(find "${LIVE_ROOT}/feature_store/bpc_live_240T" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "  ℹ️  币种数量: $SYMBOL_COUNT"
    check_item "至少有 1 个币种的 feature_store" "[ $SYMBOL_COUNT -ge 1 ]"
else
    echo "  ⚠️  feature_store layer 不存在，请先运行: ./live/scripts/build_feature_store.sh ${UNIVERSE} 3"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi
echo ""

echo "🔑 3. Binance API 密钥检查"
echo "-----------------------------------------------------------"

# 先尝试加载 live/binance_mainnet.env
if [ -f "live/binance_mainnet.env" ]; then
    echo "ℹ️  找到 live/binance_mainnet.env，正在加载..."
    set -a
    source live/binance_mainnet.env
    set +a
fi

check_item "BINANCE_API_KEY 已配置" "[ -n \"${BINANCE_API_KEY}\" ]" || {
    echo "  💡 请确保 live/binance_mainnet.env 文件存在并包含 BINANCE_API_KEY"
    echo "  或者手动设置: export BINANCE_API_KEY=your_api_key"
}
check_item "BINANCE_API_SECRET 已配置" "[ -n \"${BINANCE_API_SECRET}\" ]" || {
    echo "  💡 请确保 live/binance_mainnet.env 文件存在并包含 BINANCE_API_SECRET"
    echo "  或者手动设置: export BINANCE_API_SECRET=your_api_secret"
}
echo ""

echo "📂 4. 数据目录检查"
echo "-----------------------------------------------------------"
check_item "data/parquet_data 存在" "[ -d data/parquet_data ]"

# 检查最近月份的 tick 数据
CURRENT_MONTH=$(date +%Y-%m)
PREV_MONTH=$(date -d "last month" +%Y-%m)

TICK_FILES_CURRENT=$(find data/parquet_data -name "*${CURRENT_MONTH}.parquet" 2>/dev/null | wc -l)
TICK_FILES_PREV=$(find data/parquet_data -name "*${PREV_MONTH}.parquet" 2>/dev/null | wc -l)

echo "  ℹ️  当前月 ($CURRENT_MONTH) tick 文件: $TICK_FILES_CURRENT"
echo "  ℹ️  上月 ($PREV_MONTH) tick 文件: $TICK_FILES_PREV"

if [ $TICK_FILES_PREV -ge 6 ]; then
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    echo "  ✅ 历史 tick 数据充足"
else
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
    echo "  ⚠️  历史 tick 数据不足，建议下载最近 3 个月数据"
    echo "     mlbot data download --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT"
fi
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
echo ""

echo "⚙️  5. 策略配置检查"
echo "-----------------------------------------------------------"
check_item "config/strategies/bpc 存在" "[ -d config/strategies/bpc ]"
check_item "meta.yaml 存在" "[ -f config/strategies/bpc/meta.yaml ]"
check_item "archetypes/gate.yaml 存在" "[ -f config/strategies/bpc/archetypes/gate.yaml ]"
check_item "archetypes/evidence.yaml 存在" "[ -f config/strategies/bpc/archetypes/evidence.yaml ]"
check_item "archetypes/execution.yaml 存在" "[ -f config/strategies/bpc/archetypes/execution.yaml ]"
echo ""

echo "============================================================"
echo "📊 检查结果汇总"
echo "============================================================"
echo "总检查项: $TOTAL_CHECKS"
echo "✅ 通过: $PASSED_CHECKS"
echo "❌ 失败: $FAILED_CHECKS"
echo ""

if [ $FAILED_CHECKS -eq 0 ]; then
    echo "🎉 所有依赖检查通过！"
    echo ""
    echo "可以启动实盘测试（观察模式）："
    echo "  export MLBOT_LIVE_MODE=bpc"
    echo "  export MLBOT_LIVE_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT"
    echo "  export MLBOT_LIVE_TRADE_SIZE=0.0  # 观察模式，不下单"
    echo "  export MLBOT_LIVE_USE_FUTURES=true"
    echo "  export MLBOT_BPC_BAR_MINUTES=240"
    echo "  export MLBOT_BPC_WINDOW_MINUTES=15"
    echo "  python scripts/run_live.py"
    echo ""
    exit 0
else
    echo "⚠️  有 $FAILED_CHECKS 项检查失败，请先解决上述问题。"
    echo ""
    exit 1
fi
