#!/bin/bash
# 准备实盘 warmup ticks 数据
# 替代旧的 build_feature_store.sh（live 不再需要 Feature Store）
#
# 用法:
#   bash live/scripts/prepare_warmup_ticks.sh [universe] [months] [--from-local|--fill-gap]
#   bash live/scripts/prepare_warmup_ticks.sh highcap 6               # 从 Binance 下载（6个月，覆盖 atr_percentile 等长 lookback）
#   bash live/scripts/prepare_warmup_ticks.sh highcap 6 --from-local  # 从 data/parquet_data/ 读取
#   bash live/scripts/prepare_warmup_ticks.sh highcap 6 --fill-gap   # 只补全缺失的 daily 数据
#
# 说明:
#   特征计算通过 compute_features_batch() 批量计算，直接从磁盘读取：
#   - 90+ 天 1min bars → 重采样 4h → OHLCV 特征 (atr_percentile=540, sma_200, SR 等)
#   - 7 天 1min ticks → VPIN 自适应桶
#   因此需要准备 6 个月历史数据
#
# 功能:
#   --from-local 模式:
#     直接从 data/parquet_data/ 读取已有 1min 聚合数据（快速）
#   默认模式:
#     1. 下载最近 N 个月 monthly aggTrades（Binance UM 期货）
#     2. 下载最近 1 个月 daily aggTrades（补齐到昨天）
#     3. 转换为 1min 聚合 ticks + bars
#   共同:
#     4. 按日期拆分写入 live/{universe}/data/ticks/ 和 bars/

set -e

UNIVERSE="${1:-highcap}"
MONTHS="${2:-6}"
# 收集额外参数（如 --from-local）
shift 2 2>/dev/null || true
EXTRA_ARGS="$@"

echo "============================================================"
echo "🚀 准备实盘 warmup ticks 数据"
echo "============================================================"
echo "Universe: $UNIVERSE"
echo "Months:   $MONTHS"
echo "Args:     $EXTRA_ARGS"
echo ""

# 切换到项目根目录
cd "$(dirname "$0")/../.."

# 设置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 执行 Python 脚本
python live/scripts/prepare_warmup_ticks.py \
    --universe "$UNIVERSE" \
    --months "$MONTHS" \
    $EXTRA_ARGS

echo ""
echo "============================================================"
echo "✅ warmup 数据准备完成！"
echo ""
echo "下一步:"
echo "  bash live/scripts/start_live.sh $UNIVERSE"
echo ""
echo "系统启动后约 4h 进入 NORMAL 模式"
echo "============================================================"
