#!/bin/bash

# PCM联合回测训练监控脚本
# 监控BPC和ME训练完成后自动执行PCM回测

RESULTS_DIR="/home/yin/trading/ml_trading_bot/results"
FER_PREDICTIONS="/home/yin/trading/ml_trading_bot/results/train_final_20260216_184525_return_tree/fer/predictions_fixed.parquet"

# 查找最新的BPC和ME训练目录
find_latest_training() {
    local strategy=$1
    local latest=$(ls -t ${RESULTS_DIR}/train_final_*_return_tree 2>/dev/null | while read dir; do
        if [ -f "$dir/$strategy/predictions.parquet" ]; then
            echo "$dir"
            break
        fi
    done)
    echo "$latest"
}

echo "=========================================="
echo "PCM联合回测训练监控"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

while true; do
    clear
    echo "=========================================="
    echo "PCM训练进度检查 - $(date '+%H:%M:%S')"
    echo "=========================================="
    
    # 检查BPC训练
    BPC_DIR=$(find_latest_training "bpc")
    if [ -n "$BPC_DIR" ] && [ -f "$BPC_DIR/bpc/predictions.parquet" ]; then
        echo "✅ BPC训练完成: $BPC_DIR"
        BPC_PREDICTIONS="$BPC_DIR/bpc/predictions.parquet"
        BPC_DONE=1
    else
        echo "⏳ BPC训练进行中..."
        BPC_DONE=0
    fi
    
    # 检查ME训练
    ME_DIR=$(find_latest_training "me")
    if [ -n "$ME_DIR" ] && [ -f "$ME_DIR/me/predictions.parquet" ]; then
        echo "✅ ME训练完成: $ME_DIR"
        ME_PREDICTIONS="$ME_DIR/me/predictions.parquet"
        ME_DONE=1
    else
        echo "⏳ ME训练进行中..."
        ME_DONE=0
    fi
    
    # 检查FER
    if [ -f "$FER_PREDICTIONS" ]; then
        echo "✅ FER predictions已就绪"
        FER_DONE=1
    else
        echo "❌ FER predictions缺失"
        FER_DONE=0
    fi
    
    echo ""
    
    # 如果全部完成，执行PCM回测
    if [ $BPC_DONE -eq 1 ] && [ $ME_DONE -eq 1 ] && [ $FER_DONE -eq 1 ]; then
        echo "=========================================="
        echo "✅ 所有训练完成，开始PCM联合回测"
        echo "=========================================="
        
        echo "BPC: $BPC_PREDICTIONS"
        echo "ME:  $ME_PREDICTIONS"
        echo "FER: $FER_PREDICTIONS"
        echo ""
        
        # 执行PCM回测
        cd /home/yin/trading/ml_trading_bot
        python scripts/backtest_execution_layer.py \
            --pcm bpc:$BPC_PREDICTIONS \
                 me:$ME_PREDICTIONS \
                 fer:$FER_PREDICTIONS \
            --quantile-train-start 2025-02-01 \
            --quantile-train-end 2025-08-01 \
            2>&1 | tee pcm_backtest_result_$(date '+%Y%m%d_%H%M%S').log
        
        EXIT_CODE=${PIPESTATUS[0]}
        
        echo ""
        echo "=========================================="
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ PCM回测完成！"
        else
            echo "❌ PCM回测失败，退出码: $EXIT_CODE"
        fi
        echo "=========================================="
        
        break
    fi
    
    echo "等待30秒后再次检查..."
    sleep 30
done

echo ""
echo "监控脚本结束: $(date '+%Y-%m-%d %H:%M:%S')"
