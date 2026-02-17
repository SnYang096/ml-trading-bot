#!/bin/bash
# FER Feature Store 构建监控脚本
# 每30分钟检查一次进度

LOG_FILE="/tmp/fer_fs_build.log"
CHECK_INTERVAL=1800  # 30分钟

echo "=== FER Feature Store 构建监控 ==="
echo "日志文件: $LOG_FILE"
echo "检查间隔: 30分钟"
echo "开始时间: $(date)"
echo ""

while true; do
    echo "=== 检查时间: $(date) ==="
    
    # 检查进程是否还在运行
    if pgrep -f "feature-store build.*config/strategies/fer" > /dev/null; then
        echo "✅ Feature Store 构建进程运行中"
        
        # 显示最后30行日志
        echo ""
        echo "📋 最新日志（最后30行）:"
        echo "----------------------------------------"
        tail -30 "$LOG_FILE"
        echo "----------------------------------------"
        
        # 检查是否有错误
        if grep -i "error\|exception\|failed" "$LOG_FILE" | tail -5; then
            echo ""
            echo "⚠️  发现错误信息"
        fi
        
    else
        echo "🏁 Feature Store 构建进程已结束"
        echo ""
        echo "📋 完整日志查看:"
        echo "   tail -100 $LOG_FILE"
        echo ""
        
        # 检查是否成功
        if grep -q "✅.*completed\|Feature store built successfully" "$LOG_FILE"; then
            echo "✅ 构建成功！"
        else
            echo "❌ 构建可能失败，请检查日志"
        fi
        
        break
    fi
    
    echo ""
    echo "⏳ 等待30分钟后下次检查..."
    sleep $CHECK_INTERVAL
done

echo ""
echo "监控结束时间: $(date)"
