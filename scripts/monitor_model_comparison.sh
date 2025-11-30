#!/bin/bash
# 监控模型比较运行状态

echo "🔍 模型比较运行状态监控"
echo "========================"
echo ""

# 检查进程
echo "📊 运行中的进程："
ps aux | grep "sr_reversal_model_comparison" | grep -v grep | awk '{printf "  PID: %-8s CPU: %-6s MEM: %-6s Timeframe: %s\n", $2, $3"%", $4"%", $NF}' | head -5
echo ""

# 检查日志文件
echo "📝 最新日志状态："
for log in /tmp/*comparison*.log; do
    if [ -f "$log" ]; then
        echo "  $(basename $log):"
        echo "    - 行数: $(wc -l < "$log" 2>/dev/null || echo 0)"
        echo "    - 大小: $(du -h "$log" 2>/dev/null | cut -f1)"
        echo "    - 最后更新: $(stat -c %y "$log" 2>/dev/null | cut -d. -f1)"
        echo "    - 最后一行: $(tail -1 "$log" 2>/dev/null | cut -c1-80)"
        echo ""
    fi
done

# 检查输出文件
echo "📁 输出文件："
find results/model_comparison -type f \( -name "*.html" -o -name "*.csv" \) 2>/dev/null | while read f; do
    echo "  $(basename $f): $(stat -c %y "$f" 2>/dev/null | cut -d. -f1)"
done | head -10
echo ""

# 检查关键进度指标
echo "🎯 关键进度指标："
echo "  4h 比较："
tail -200 /tmp/4h_comparison_no_tick_check.log 2>/dev/null | grep -E "Training|Evaluating|Report|Win Rate|Sharpe|Total R|n_trades|Error" | tail -5
echo ""
echo "  1h 比较："
tail -200 /tmp/1h_comparison_fixed.log 2>/dev/null | grep -E "Training|Evaluating|Report|Win Rate|Sharpe|Total R|n_trades|Error" | tail -5
echo ""

