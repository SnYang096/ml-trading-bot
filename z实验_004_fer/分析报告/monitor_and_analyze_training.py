#!/usr/bin/env python3
"""
监控训练完成并自动分析结果
"""
import time
import pandas as pd
from pathlib import Path
import subprocess
import sys

def find_latest_me_training():
    """查找最新的ME训练结果"""
    results_dir = Path("/home/yin/trading/ml_trading_bot/results")
    pattern = "train_final_*_return_tree"
    
    me_dirs = []
    for p in results_dir.glob(pattern):
        me_pred = p / "me" / "predictions.parquet"
        if me_pred.exists():
            me_dirs.append((p, me_pred.stat().st_mtime))
    
    if not me_dirs:
        return None
    
    # 返回最新的
    latest = max(me_dirs, key=lambda x: x[1])
    return latest[0] / "me" / "predictions.parquet"

def wait_for_new_training():
    """等待新的ME训练完成"""
    print("🔍 等待新的ME训练完成...")
    print("   当前正在运行的训练: PID 463 (19:35启动)")
    
    # 记录当前最新的ME
    current_latest = find_latest_me_training()
    print(f"   当前最新ME: {current_latest}")
    
    check_count = 0
    while True:
        time.sleep(60)  # 每分钟检查一次
        check_count += 1
        
        new_latest = find_latest_me_training()
        if new_latest and new_latest != current_latest:
            print(f"\n✅ 发现新的ME训练结果: {new_latest}")
            return new_latest
        
        # 每5分钟报告一次
        if check_count % 5 == 0:
            print(f"   等待中... (已等待 {check_count} 分钟)")

def analyze_me_results(pred_file):
    """分析ME训练结果"""
    print("\n" + "="*80)
    print("📊 ME训练结果分析")
    print("="*80)
    
    df = pd.read_parquet(pred_file)
    
    # 基本统计
    print(f"\n【数据概况】")
    print(f"  总行数: {len(df):,}")
    print(f"  币种数: {df['_symbol'].nunique() if '_symbol' in df.columns else 'N/A'}")
    print(f"  时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    
    # 检查split分布
    if 'split' in df.columns:
        print(f"\n【训练/测试分布】")
        for split in df['split'].unique():
            count = (df['split'] == split).sum()
            pct = count / len(df) * 100
            print(f"  {split}: {count:,} ({pct:.1f}%)")
    
    # 预测分布
    if 'pred' in df.columns:
        print(f"\n【预测分布】")
        print(f"  mean: {df['pred'].mean():.4f}")
        print(f"  std: {df['pred'].std():.4f}")
        print(f"  min: {df['pred'].min():.4f}")
        print(f"  max: {df['pred'].max():.4f}")
    
    # 标签分布（如果有）
    if 'forward_rr' in df.columns:
        print(f"\n【标签分布 (forward_rr)】")
        print(f"  mean: {df['forward_rr'].mean():.4f}")
        print(f"  std: {df['forward_rr'].std():.4f}")
        
        # 胜率估算（forward_rr > 0的比例）
        win_rate = (df['forward_rr'] > 0).mean() * 100
        print(f"  胜率: {win_rate:.1f}% (forward_rr > 0)")
    
    print("\n" + "="*80)
    
    return df

def run_pcm_backtest(me_pred_file):
    """运行PCM回测"""
    print("\n🎯 准备运行PCM回测...")
    
    # 添加direction列
    print("   添加entry_direction列...")
    df = pd.read_parquet(me_pred_file)
    df['entry_direction'] = 1.0
    fixed_file = me_pred_file.parent / "predictions_fixed.parquet"
    df.to_parquet(fixed_file, index=False)
    print(f"   ✅ 已保存: {fixed_file}")
    
    # 运行回测
    cmd = [
        "python", "scripts/backtest_execution_layer.py",
        "--pcm",
        "bpc:/home/yin/trading/ml_trading_bot/results/train_final_20260208_220616_return_tree/bpc/predictions.parquet",
        f"me:{fixed_file}",
        "fer:/home/yin/trading/ml_trading_bot/results/train_final_20260216_184525_return_tree/fer/predictions_fixed.parquet",
        "--quantile-train-start", "2025-02-01",
        "--quantile-train-end", "2025-08-01"
    ]
    
    print(f"\n执行命令:")
    print(" ".join(cmd))
    print()
    
    result = subprocess.run(
        cmd,
        cwd="/home/yin/trading/ml_trading_bot",
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode

def main():
    print("="*80)
    print("🚀 ME训练监控与分析")
    print("="*80)
    
    # 等待新训练
    new_pred_file = wait_for_new_training()
    
    # 分析结果
    df = analyze_me_results(new_pred_file)
    
    # 运行PCM回测
    run_pcm_backtest(new_pred_file)
    
    print("\n✅ 分析完成！")

if __name__ == "__main__":
    main()
