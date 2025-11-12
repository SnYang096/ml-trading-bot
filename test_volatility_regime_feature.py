#!/usr/bin/env python3
"""测试波动率状态特征在 baseline 特征工程中的表现"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
from data_tools.comprehensive_feature_engineering import ComprehensiveFeatureEngineer
from data_tools.baseline_feature_engineering import get_baseline_feature_columns

print("=" * 100)
print("测试波动率状态特征 (volatility_regime)")
print("=" * 100)

# 读取实际数据
data_dir = Path('data/parquet_data')
btc_files = sorted(data_dir.glob('BTCUSDT_2024-11.parquet'))

if btc_files:
    df = pd.read_parquet(btc_files[0])
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    df['symbol'] = 'BTCUSDT'
    
    print(f"\n✅ 数据加载成功: {len(df)} 行")
    print(f"   日期范围: {df.index.min()} 到 {df.index.max()}")
    print(f"   列: {df.columns.tolist()}")
    
    # 使用 baseline 特征工程
    print(f"\n{'=' * 100}")
    print("使用 baseline 特征工程")
    print(f"{'=' * 100}")
    
    engineer = ComprehensiveFeatureEngineer(feature_types='baseline')
    result = engineer.engineer_features(df.head(2000), fit=True)
    
    print(f"\n✅ 特征工程成功")
    print(f"   总特征数: {len(result.columns)}")
    print(f"   volatility_regime 在特征列表中: {'volatility_regime' in result.columns}")
    
    # 获取 baseline 特征列表
    features = get_baseline_feature_columns(result)
    print(f"   Baseline 特征数: {len(features)}")
    print(f"   volatility_regime 在 baseline 特征中: {'volatility_regime' in features}")
    
    # 分析波动率状态特征
    if 'volatility_regime' in result.columns:
        print(f"\n{'=' * 100}")
        print("波动率状态特征分析")
        print(f"{'=' * 100}")
        
        vr = result['volatility_regime']
        print(f"\n特征统计:")
        print(f"   值范围: [{vr.min()}, {vr.max()}]")
        print(f"   数据类型: {vr.dtype}")
        print(f"   高波动状态比例: {vr.mean():.2%}")
        print(f"   低波动状态比例: {(1 - vr.mean()):.2%}")
        print(f"   唯一值: {vr.unique()}")
        print(f"   NaN 数量: {vr.isna().sum()}")
        
        print(f"\n详细统计:")
        print(vr.describe())
        
        # 分析波动率状态与 ATR 的关系
        if 'atr' in result.columns:
            print(f"\n{'=' * 100}")
            print("波动率状态与 ATR 的关系")
            print(f"{'=' * 100}")
            
            atr = result['atr']
            high_vol_atr = atr[vr == 1]
            low_vol_atr = atr[vr == 0]
            
            print(f"\n高波动状态 (volatility_regime = 1):")
            print(f"   样本数: {len(high_vol_atr)}")
            print(f"   ATR 均值: {high_vol_atr.mean():.4f}")
            print(f"   ATR 中位数: {high_vol_atr.median():.4f}")
            print(f"   ATR 范围: [{high_vol_atr.min():.4f}, {high_vol_atr.max():.4f}]")
            
            print(f"\n低波动状态 (volatility_regime = 0):")
            print(f"   样本数: {len(low_vol_atr)}")
            print(f"   ATR 均值: {low_vol_atr.mean():.4f}")
            print(f"   ATR 中位数: {low_vol_atr.median():.4f}")
            print(f"   ATR 范围: [{low_vol_atr.min():.4f}, {low_vol_atr.max():.4f}]")
            
            print(f"\n对比:")
            print(f"   ATR 均值比: {high_vol_atr.mean() / low_vol_atr.mean():.2f}x")
            print(f"   ATR 中位数比: {high_vol_atr.median() / low_vol_atr.median():.2f}x")
        
        # 分析波动率状态的分布
        print(f"\n{'=' * 100}")
        print("波动率状态分布分析")
        print(f"{'=' * 100}")
        
        # 计算连续高波动/低波动状态的持续时间
        vr_series = vr.fillna(0).astype(int)
        regime_changes = (vr_series != vr_series.shift()).cumsum()
        regime_durations = regime_changes.value_counts().sort_index()
        
        print(f"\n波动率状态变化:")
        print(f"   状态变化次数: {len(regime_changes.unique())}")
        print(f"   平均持续时间: {regime_durations.mean():.2f} bars")
        print(f"   最长持续时间: {regime_durations.max()} bars")
        print(f"   最短持续时间: {regime_durations.min()} bars")
        
        # 分析波动率状态的转换
        print(f"\n波动率状态转换:")
        transitions = (vr_series != vr_series.shift()).sum()
        print(f"   状态转换次数: {transitions}")
        print(f"   平均转换频率: {transitions / len(vr_series):.2%}")
        
        # 验证特征逻辑
        print(f"\n{'=' * 100}")
        print("特征逻辑验证")
        print(f"{'=' * 100}")
        
        if 'atr' in result.columns:
            # 计算 200 个周期的 70 分位数
            atr_quantile_70 = result['atr'].rolling(window=200, min_periods=1).quantile(0.7)
            
            # 验证特征是否正确
            expected_vr = (result['atr'] > atr_quantile_70).astype(int).fillna(0)
            matches = (vr == expected_vr).sum()
            total = len(vr)
            
            print(f"\n特征逻辑验证:")
            print(f"   匹配数量: {matches}/{total} ({100*matches/total:.2f}%)")
            print(f"   特征正确性: {'✅ 正确' if matches == total else '❌ 错误'}")
        
        print(f"\n{'=' * 100}")
        print("结论")
        print(f"{'=' * 100}")
        print(f"\n✅ 波动率状态特征已成功添加到 baseline 特征工程中")
        print(f"   - 特征名: volatility_regime")
        print(f"   - 特征类型: 二进制 (0/1)")
        print(f"   - 特征含义: 1=高波动状态, 0=低波动状态")
        print(f"   - 计算方式: atr > atr.rolling(200).quantile(0.7)")
        print(f"   - 用途: 帮助模型在高波动时更信任趋势信号，低波动时减少开仓")
        
    else:
        print(f"\n❌ volatility_regime 特征未找到")
else:
    print(f"\n❌ 未找到 BTCUSDT 数据文件")

print(f"\n{'=' * 100}")
print("测试完成")
print(f"{'=' * 100}")

