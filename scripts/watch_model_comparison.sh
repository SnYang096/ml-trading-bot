#!/bin/bash
# 持续监控模型比较运行状态

INTERVAL=30  # 每30秒检查一次
MAX_ITERATIONS=120  # 最多监控1小时（120次 * 30秒）

echo "🔍 开始监控模型比较运行状态（每${INTERVAL}秒更新一次，最多${MAX_ITERATIONS}次）"
echo "按 Ctrl+C 停止监控"
echo ""

for i in $(seq 1 $MAX_ITERATIONS); do
    clear
    echo "=========================================="
    echo "📊 模型比较运行状态监控 - 第 $i 次检查"
    echo "=========================================="
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 检查进程
    echo "🔄 运行中的进程："
    ps aux | grep "sr_reversal_model_comparison" | grep -v grep | awk '{printf "  PID: %-8s CPU: %-6s MEM: %-6s %s\n", $2, $3"%", $4"%", substr($0, index($0,$11))}' | head -5
    echo ""
    
    # 检查4h进度
    echo "📈 4h 比较进度："
    if [ -f /tmp/4h_comparison_no_tick_check.log ]; then
        tail -20 /tmp/4h_comparison_no_tick_check.log 2>/dev/null | grep -E "Level|Computing|Training|Evaluating|Report|Error|error|✅|❌|completed|完成|VPIN" | tail -5
        echo "  日志行数: $(wc -l < /tmp/4h_comparison_no_tick_check.log 2>/dev/null || echo 0)"
    else
        echo "  日志文件不存在"
    fi
    echo ""
    
    # 检查1h进度
    echo "📈 1h 比较进度："
    if [ -f /tmp/1h_comparison_fixed.log ]; then
        tail -20 /tmp/1h_comparison_fixed.log 2>/dev/null | grep -E "Level|Computing|Training|Evaluating|Report|Error|error|✅|❌|completed|完成|VPIN" | tail -5
        echo "  日志行数: $(wc -l < /tmp/1h_comparison_fixed.log 2>/dev/null || echo 0)"
    else
        echo "  日志文件不存在"
    fi
    echo ""
    
    # 检查输出文件
    echo "📁 最新输出文件："
    find results/model_comparison -type f \( -name "*.html" -o -name "*.csv" \) -newermt "10 minutes ago" 2>/dev/null | while read f; do
        echo "  $(basename $f): $(stat -c %y "$f" 2>/dev/null | cut -d. -f1)"
    done | head -5
    if [ -z "$(find results/model_comparison -type f \( -name "*.html" -o -name "*.csv" \) -newermt "10 minutes ago" 2>/dev/null)" ]; then
        echo "  (最近10分钟内无新文件)"
    fi
    echo ""
    
    # 检查是否完成
    if ! ps aux | grep -q "[s]r_reversal_model_comparison"; then
        echo "✅ 所有进程已完成！"
        echo ""
        echo "📊 最终结果："
        find results/model_comparison -type f -name "*.html" -o -name "*.csv" 2>/dev/null | xargs ls -lht 2>/dev/null | head -5
        break
    fi
    
    echo "下次更新: ${INTERVAL}秒后..."
    sleep $INTERVAL
done

echo ""
echo "监控结束"

