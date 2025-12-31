#!/bin/bash
# 任务监控脚本 - 定时检查后台任务状态
# 用法: ./scripts/monitor_tasks.sh [间隔秒数，默认30]

INTERVAL=${1:-30}

echo "🔄 任务监控已启动（每 ${INTERVAL} 秒检查一次）"
echo "   按 Ctrl+C 退出"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "📊 任务状态监控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Feature Group Search 任务
    echo "🔍 Feature Group Search 任务:"
    echo ""
    
    for strategy in sr_breakout compression_breakout trend_following sr_reversal; do
        LOG="/tmp/fgs_${strategy}_expanded.log"
        if [ -f "$LOG" ]; then
            # 检查是否完成
            if grep -q "✅ Search complete" "$LOG" 2>/dev/null; then
                echo "  ✅ $strategy: 已完成"
                grep -E "Best Sharpe|Final features" "$LOG" 2>/dev/null | tail -2 | sed 's/^/     /'
            elif grep -q "Error\|Exception\|Traceback" "$LOG" 2>/dev/null; then
                echo "  ❌ $strategy: 出错"
                tail -1 "$LOG" | sed 's/^/     /'
            else
                echo "  🔄 $strategy: 运行中"
                tail -1 "$LOG" 2>/dev/null | sed 's/^/     /'
            fi
        else
            echo "  ⏸️  $strategy: 未启动"
        fi
        echo ""
    done
    
    # 运行中的进程数
    RUNNING=$(ps aux | grep "feature_group_search" | grep -v grep | wc -l)
    echo "----------------------------------------"
    echo "📈 运行中进程数: $RUNNING"
    echo ""
    echo "💡 提示: 任务完成后结果在 results/feature_group_search/"
    echo "   按 Ctrl+C 退出监控"
    
    sleep $INTERVAL
done

