#!/bin/bash
# 检查冒烟测试状态

LOG_FILE="logs/smoke_test_mainnet.log"
PID=$(ps aux | grep "test_testnet_smoke_with_gap_fill.py" | grep -v grep | awk '{print $2}' | head -1)

echo "=========================================="
echo "冒烟测试状态检查"
echo "=========================================="
echo ""

if [ -z "$PID" ]; then
    echo "❌ 测试进程未运行"
    exit 1
else
    echo "✅ 测试进程正在运行 (PID: $PID)"
fi

echo ""
echo "📊 最新日志 (最后30行):"
echo "----------------------------------------"
if [ -f "$LOG_FILE" ]; then
    tail -30 "$LOG_FILE"
else
    echo "⚠️  日志文件不存在: $LOG_FILE"
fi

echo ""
echo "📈 统计信息:"
echo "----------------------------------------"
if [ -f "$LOG_FILE" ]; then
    echo "接收tick数: $(grep -c "收到tick\|Received tick" "$LOG_FILE" 2>/dev/null || echo "0")"
    echo "重连次数: $(grep -c "重连成功\|Reconnect success" "$LOG_FILE" 2>/dev/null || echo "0")"
    echo "连接错误: $(grep -c "connection error\|连接错误" "$LOG_FILE" 2>/dev/null || echo "0")"
    echo "健康状态变化: $(grep -c "健康状态变化\|Health status" "$LOG_FILE" 2>/dev/null || echo "0")"
fi

echo ""
echo "⏱️  运行时间:"
if [ -n "$PID" ] && [ "$PID" != "" ]; then
    ps -p "$PID" -o etime= 2>/dev/null | awk '{print "   " $0}' || echo "   无法获取运行时间"
fi

echo ""
echo "=========================================="
