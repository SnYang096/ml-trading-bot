#!/bin/bash
# 构建实盘feature_store（按 universe，最近N个月）

set -e

UNIVERSE="${1:-highcap}"            # 第一个参数：universe 名称（默认 highcap）
MONTHS="${2:-3}"                     # 第二个参数：最近 N 个月

LIVE_ROOT="live/${UNIVERSE}"

echo "============================================================"
echo "🏗️  构建实盘feature_store"
echo "============================================================"
echo "Universe: $UNIVERSE"
echo "Months: $MONTHS (最近N个月)"
echo ""

# 计算起止日期（最近N个月）
# 当前是 2026-02，上个月是 2026-01
# 最近3个月 = 2025-11, 2025-12, 2026-01
LAST_MONTH_START=$(date -d "$(date +%Y-%m-01) -1 month" +%Y-%m-01)  # 上个月1号 (2026-01-01)
START_DATE=$(date -d "$LAST_MONTH_START -$((MONTHS-1)) months" +%Y-%m-01)  # 往前推 (N-1) 个月 (2025-11-01)
END_DATE=$(date -d "$LAST_MONTH_START +1 month -1 day" +%Y-%m-%d)  # 上个月最后一天 (2026-01-31)

echo "📅 时间范围: $START_DATE ~ $END_DATE"
echo ""

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LIVE_ROOT

# 选择 CLI 命令（优先使用 mlbot）
if command -v mlbot >/dev/null 2>&1; then
  ML_CMD="mlbot"
else
  ML_CMD="python -m src.cli.main"
fi

# 从 universe.yaml 读取 symbols
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

echo "Symbols: $SYMBOLS"

# 执行feature-store build
echo "🔨 执行 feature-store build..."
echo ""

$ML_CMD feature-store build \
    --config "config/strategies/bad-candidates/bpc" \
    --symbols "$SYMBOLS" \
    --timeframe "240T" \
    --data-path "data/parquet_data" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --root "$LIVE_ROOT/feature_store" \
    --layer "bpc_live_240T" \
    --warmup-months 0 \
    --warmup-bars 0 \
    --no-docker

echo ""
echo "✅ feature_store构建完成！"
echo "   输出目录: $LIVE_ROOT/feature_store/bpc_live_240T"
