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

echo "📦 2. Feature Store 状态（已废弃）"
echo "-----------------------------------------------------------"
echo "  ⚠️  [已废弃] Feature Store 不再使用。"
echo "  原因: 当前 pipeline 无法基于特征流式计算，只能基于 ticks/bars 流式计算。"
echo "  现已改为: 基于 ticks/bars 数据实时重算全部特征。"
echo "  数据来源: ${LIVE_ROOT}/data/ticks/ 和 ${LIVE_ROOT}/data/bars/"
if [ -d "${LIVE_ROOT}/data/ticks" ]; then
    TICK_SYMS=$(ls -d ${LIVE_ROOT}/data/ticks/*/ 2>/dev/null | wc -l)
    echo "  ✅ ticks 数据目录存在: $TICK_SYMS 个币种"
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    echo "  ❌ ${LIVE_ROOT}/data/ticks/ 不存在"
    echo "  💡 运行: bash live/scripts/prepare_warmup_ticks.sh ${UNIVERSE} 6 --from-local"
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
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

echo "⚙️  5. 策略配置检查（实盘目录）"
echo "-----------------------------------------------------------"
check_item "${LIVE_ROOT}/config/strategies/tpc 存在" "[ -d ${LIVE_ROOT}/config/strategies/tpc ]"
check_item "meta.yaml 存在" "[ -f ${LIVE_ROOT}/config/strategies/tpc/meta.yaml ]"
check_item "archetypes/gate.yaml 存在" "[ -f ${LIVE_ROOT}/config/strategies/tpc/archetypes/gate.yaml ]"
check_item "archetypes/execution.yaml 存在" "[ -f ${LIVE_ROOT}/config/strategies/tpc/archetypes/execution.yaml ]"
check_item "archetypes/entry_filters.yaml 存在" "[ -f ${LIVE_ROOT}/config/strategies/tpc/archetypes/entry_filters.yaml ]"
echo ""

echo "📅 6. Warmup 数据时间覆盖检查 (至少 6 个月)"
echo "-----------------------------------------------------------"
SIX_MONTHS_AGO=$(date -d "6 months ago" +%Y-%m-%d)
echo "  ℹ️  要求最早数据日期: $SIX_MONTHS_AGO"

WARMUP_OK=true
if [ -d "${LIVE_ROOT}/data/bars" ]; then
    for SYM_DIR in "${LIVE_ROOT}/data/bars/"*/; do
        if [ ! -d "$SYM_DIR" ]; then continue; fi
        SYM_NAME=$(basename "$SYM_DIR")
        EARLIEST_FILE=$(ls "$SYM_DIR" 2>/dev/null | sort | head -1)
        LATEST_FILE=$(ls "$SYM_DIR" 2>/dev/null | sort | tail -1)
        if [ -z "$EARLIEST_FILE" ]; then
            echo "  ⚠️  $SYM_NAME: 无 bars 数据"
            WARMUP_OK=false
            continue
        fi
        # 提取日期 (文件名格式: YYYY-MM-DD.parquet)
        EARLIEST_DATE=$(echo "$EARLIEST_FILE" | sed 's/\.parquet$//')
        LATEST_DATE=$(echo "$LATEST_FILE" | sed 's/\.parquet$//')
        if [[ "$EARLIEST_DATE" > "$SIX_MONTHS_AGO" ]]; then
            echo "  ⚠️  $SYM_NAME: 最早=$EARLIEST_DATE, 不足 6 个月 (需要 <= $SIX_MONTHS_AGO)"
            WARMUP_OK=false
        else
            echo "  ✅ $SYM_NAME: $EARLIEST_DATE ~ $LATEST_DATE"
        fi
    done
else
    echo "  ⚠️  ${LIVE_ROOT}/data/bars 目录不存在"
    WARMUP_OK=false
fi

TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
if $WARMUP_OK; then
    echo "  ✅ 所有币种 warmup 数据覆盖充足 (≥ 6 个月)"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    echo "  ❌ 部分币种 warmup 数据不足 6 个月"
    echo "  💡 补全命令: bash live/scripts/prepare_warmup_ticks.sh ${UNIVERSE} 6 --from-local"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi
echo ""

echo "📊 7. Evidence / Gate（TPC）"
echo "-----------------------------------------------------------"
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
if [ -f ${LIVE_ROOT}/config/strategies/tpc/archetypes/evidence.yaml ]; then
    EV_COUNT=$(grep -c '^ *- id:' ${LIVE_ROOT}/config/strategies/tpc/archetypes/evidence.yaml 2>/dev/null || echo 0)
    if [ "$EV_COUNT" -ge 1 ]; then
        echo "  ✅ evidence.yaml: $EV_COUNT 个 evidence 特征"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo "  ❌ evidence.yaml 存在但无 evidence 特征配置"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
else
    echo "  ⚠️  evidence.yaml 不存在（TPC 可无；视为通过）"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
fi

TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
if [ -f ${LIVE_ROOT}/config/strategies/tpc/archetypes/gate.yaml ]; then
    GATE_COUNT=$(grep -c '^ *- id:' ${LIVE_ROOT}/config/strategies/tpc/archetypes/gate.yaml 2>/dev/null || echo 0)
    if [ "$GATE_COUNT" -ge 1 ]; then
        echo "  ✅ gate.yaml: $GATE_COUNT 个 gate 规则"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo "  ❌ gate.yaml 存在但无 gate 规则"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
else
    echo "  ❌ gate.yaml 不存在"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi
echo ""
echo "📊 检查结果汇总"
echo "============================================================"
echo "总检查项: $TOTAL_CHECKS"
echo "✅ 通过: $PASSED_CHECKS"
echo "❌ 失败: $FAILED_CHECKS"
echo ""

if [ $FAILED_CHECKS -eq 0 ]; then
    echo "🎉 所有依赖检查通过！"
    echo ""
    echo "可以启动实盘测试（先起 quant-feature-bus，再 quant-trend-fattail）："
    echo "  export MLBOT_FEATURE_SOURCE=bus"
    echo "  export MLBOT_LIVE_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT"
    echo "  export MLBOT_LIVE_TRADE_SIZE=0.0  # 观察模式，不下单"
    echo "  python scripts/run_market_feature_publisher.py ...   # 终端 1"
    echo "  bash live/scripts/start_live.sh highcap              # 终端 2"
    echo ""
    exit 0
else
    echo "⚠️  有 $FAILED_CHECKS 项检查失败，请先解决上述问题。"
    echo ""
    exit 1
fi
