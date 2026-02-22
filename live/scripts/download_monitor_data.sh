#!/bin/bash
# ============================================================
# 从实盘服务器下载监控数据到本地
#
# 用法:
#   bash live/scripts/download_monitor_data.sh              # 默认下载最近 30 天
#   bash live/scripts/download_monitor_data.sh --days 7     # 最近 7 天
#   bash live/scripts/download_monitor_data.sh --all        # 全量同步
#
# 环境变量（可在 live/server.env 中配置）:
#   LIVE_SERVER_HOST  - 服务器 IP/域名
#   LIVE_SERVER_USER  - SSH 用户名
#   LIVE_SERVER_PORT  - SSH 端口（默认 22）
#   LIVE_SERVER_KEY   - SSH 私钥路径（可选）
#   LIVE_REMOTE_ROOT  - 服务器上项目根目录
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── 加载服务器配置 ──
if [ -f "$PROJECT_ROOT/live/server.env" ]; then
    set -a
    source "$PROJECT_ROOT/live/server.env"
    set +a
fi

# 默认值
LIVE_SERVER_HOST="${LIVE_SERVER_HOST:-}"
LIVE_SERVER_USER="${LIVE_SERVER_USER:-root}"
LIVE_SERVER_PORT="${LIVE_SERVER_PORT:-22}"
LIVE_SERVER_KEY="${LIVE_SERVER_KEY:-}"
LIVE_REMOTE_ROOT="${LIVE_REMOTE_ROOT:-/root/ml_trading_bot}"
UNIVERSE="${UNIVERSE:-highcap}"

# 本地下载目标目录
LOCAL_DATA_DIR="$PROJECT_ROOT/data/downloaded_live"

# ── 解析参数 ──
DAYS=30
SYNC_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --days)
            DAYS="$2"
            shift 2
            ;;
        --all)
            SYNC_ALL=true
            shift
            ;;
        --universe)
            UNIVERSE="$2"
            shift 2
            ;;
        --host)
            LIVE_SERVER_HOST="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: bash live/scripts/download_monitor_data.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --days N        下载最近 N 天的数据（默认 30）"
            echo "  --all           全量同步所有数据"
            echo "  --universe NAME universe 名称（默认 highcap）"
            echo "  --host HOST     服务器地址（覆盖 server.env）"
            echo "  -h, --help      显示帮助"
            echo ""
            echo "配置文件: live/server.env（参考 live/server.env.example）"
            exit 0
            ;;
        *)
            echo "❌ 未知参数: $1"
            exit 1
            ;;
    esac
done

# ── 校验 ──
if [ -z "$LIVE_SERVER_HOST" ]; then
    echo "❌ 未配置服务器地址"
    echo "   请设置 LIVE_SERVER_HOST 环境变量，或创建 live/server.env 文件"
    echo "   参考: cp live/server.env.example live/server.env"
    exit 1
fi

# 构建 SSH 选项
SSH_OPTS="-p $LIVE_SERVER_PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
if [ -n "$LIVE_SERVER_KEY" ]; then
    SSH_OPTS="$SSH_OPTS -i $LIVE_SERVER_KEY"
fi

REMOTE_DATA="$LIVE_REMOTE_ROOT/live/$UNIVERSE/data"
RSYNC_SSH="ssh $SSH_OPTS"

echo "============================================================"
echo "📥 下载实盘监控数据"
echo "============================================================"
echo "服务器: $LIVE_SERVER_USER@$LIVE_SERVER_HOST:$LIVE_SERVER_PORT"
echo "远程路径: $REMOTE_DATA"
echo "本地路径: $LOCAL_DATA_DIR"
echo "天数: ${SYNC_ALL:+全量}${SYNC_ALL:-$DAYS 天}"
echo ""

# ── 创建本地目录 ──
mkdir -p "$LOCAL_DATA_DIR"

# ── 构建 rsync 过滤规则 ──
# 只下载 features_15min / features_4h 目录（特征快照）
# 以及 bars / ticks 如果需要
RSYNC_COMMON_OPTS="-avz --progress --human-readable"

if [ "$SYNC_ALL" = true ]; then
    # 全量同步
    RSYNC_FILTER=""
else
    # 只下载最近 N 天的文件
    # 生成日期列表用于 --include 过滤
    echo "📅 生成最近 ${DAYS} 天的日期过滤..."
    DATE_INCLUDES=""
    for i in $(seq 0 $((DAYS - 1))); do
        d=$(date -d "$i days ago" +%Y-%m-%d 2>/dev/null || date -v-${i}d +%Y-%m-%d)
        DATE_INCLUDES="$DATE_INCLUDES --include=$d.parquet"
    done
    RSYNC_FILTER="$DATE_INCLUDES --exclude=*.parquet"
fi

# ── 1. 下载 features_15min（特征快照，L0 对比用） ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 [1/3] 下载 15min 特征快照 (features_15min/)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

rsync $RSYNC_COMMON_OPTS \
    -e "$RSYNC_SSH" \
    --include='*/' $RSYNC_FILTER \
    "$LIVE_SERVER_USER@$LIVE_SERVER_HOST:$REMOTE_DATA/features_15min/" \
    "$LOCAL_DATA_DIR/features_15min/" \
    2>&1 || echo "⚠️  features_15min 下载失败（目录可能不存在）"

# ── 2. 下载 features_4h ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 [2/3] 下载 4h 特征快照 (features_4h/)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

rsync $RSYNC_COMMON_OPTS \
    -e "$RSYNC_SSH" \
    --include='*/' $RSYNC_FILTER \
    "$LIVE_SERVER_USER@$LIVE_SERVER_HOST:$REMOTE_DATA/features_4h/" \
    "$LOCAL_DATA_DIR/features_4h/" \
    2>&1 || echo "⚠️  features_4h 下载失败（目录可能不存在）"

# ── 3. 下载 bars（1min K线，本地 batch 重算用） ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 [3/3] 下载 1min bars (bars/)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

rsync $RSYNC_COMMON_OPTS \
    -e "$RSYNC_SSH" \
    --include='*/' $RSYNC_FILTER \
    "$LIVE_SERVER_USER@$LIVE_SERVER_HOST:$REMOTE_DATA/bars/" \
    "$LOCAL_DATA_DIR/bars/" \
    2>&1 || echo "⚠️  bars 下载失败（目录可能不存在）"

# ── 统计 ──
echo ""
echo "============================================================"
echo "✅ 下载完成"
echo "============================================================"

if [ -d "$LOCAL_DATA_DIR/features_15min" ]; then
    F15_COUNT=$(find "$LOCAL_DATA_DIR/features_15min" -name "*.parquet" | wc -l)
    F15_SIZE=$(du -sh "$LOCAL_DATA_DIR/features_15min" 2>/dev/null | cut -f1)
    echo "  features_15min: ${F15_COUNT} 个文件, ${F15_SIZE}"
fi

if [ -d "$LOCAL_DATA_DIR/features_4h" ]; then
    F4H_COUNT=$(find "$LOCAL_DATA_DIR/features_4h" -name "*.parquet" | wc -l)
    F4H_SIZE=$(du -sh "$LOCAL_DATA_DIR/features_4h" 2>/dev/null | cut -f1)
    echo "  features_4h:    ${F4H_COUNT} 个文件, ${F4H_SIZE}"
fi

if [ -d "$LOCAL_DATA_DIR/bars" ]; then
    BAR_COUNT=$(find "$LOCAL_DATA_DIR/bars" -name "*.parquet" | wc -l)
    BAR_SIZE=$(du -sh "$LOCAL_DATA_DIR/bars" 2>/dev/null | cut -f1)
    echo "  bars:           ${BAR_COUNT} 个文件, ${BAR_SIZE}"
fi

echo ""
echo "本地数据目录: $LOCAL_DATA_DIR"
echo ""
echo "下一步: 运行线上/线下特征对比"
echo "  python scripts/compare_live_vs_batch_features.py \\"
echo "    --live-features $LOCAL_DATA_DIR/features_15min \\"
echo "    --bars $LOCAL_DATA_DIR/bars"
