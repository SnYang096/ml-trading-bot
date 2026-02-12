#!/bin/bash
# 准备实盘 warmup ticks 数据
# 替代旧的 build_feature_store.sh（live 不再需要 Feature Store）
#
# 用法:
#   bash live/scripts/prepare_warmup_ticks.sh [universe] [months]
#   bash live/scripts/prepare_warmup_ticks.sh highcap 6
#
# 功能:
#   1. 下载最近 N 个月 monthly aggTrades（Binance UM 期货）
#   2. 下载最近 1 个月 daily aggTrades（补齐到昨天）
#   3. 转换为 1min 聚合 ticks + bars
#   4. 按日期拆分写入 live/{universe}/data/ticks/ 和 bars/

set -e

UNIVERSE="${1:-highcap}"
MONTHS="${2:-6}"

echo "============================================================"
echo "🚀 准备实盘 warmup ticks 数据"
echo "============================================================"
echo "Universe: $UNIVERSE"
echo "Months:   $MONTHS (最近N个月 monthly + 最近1个月 daily)"
echo ""

# 切换到项目根目录
cd "$(dirname "$0")/../.."

# 设置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 执行 Python 脚本
python live/scripts/prepare_warmup_ticks.py \
    --universe "$UNIVERSE" \
    --months "$MONTHS"

echo ""
echo "============================================================"
echo "✅ warmup 数据准备完成！"
echo ""
echo "下一步:"
echo "  bash live/scripts/start_live.sh $UNIVERSE"
echo ""
echo "系统启动后约 4h 进入 NORMAL 模式"
echo "============================================================"
