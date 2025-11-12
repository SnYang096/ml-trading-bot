#!/usr/bin/env python3
"""分析指定 Alpha101 因子在 BTC 和 ETH 上的表现"""

import pandas as pd
import numpy as np

# 读取报告数据
df = pd.read_csv('results/timeframe_forward/alpha101_fixed/timeframe_forward_details.csv')

# 目标因子
target_factors = ['alpha101_001', 'alpha101_022', 'alpha101_043', 'alpha101_066']

print("=" * 100)
print("Alpha101 因子有效性验证 - BTC & ETH")
print("=" * 100)

# 检查哪些因子存在
available_factors = df['feature'].unique()
print(f"\n可用的 Alpha101 因子数量: {len([f for f in available_factors if f.startswith('alpha101_')])}")
print(f"\n查找目标因子: {target_factors}")

found_factors = [f for f in target_factors if f in available_factors]
missing_factors = [f for f in target_factors if f not in available_factors]

if missing_factors:
    print(f"\n⚠️  未找到的因子: {missing_factors}")
    # 检查是否有类似的因子名称
    print("\n检查类似的因子名称:")
    for missing in missing_factors:
        similar = [f for f in available_factors if f.startswith('alpha101_') and missing.replace('alpha101_', '') in f]
        if similar:
            print(f"  {missing} -> 可能的相关因子: {similar[:5]}")

if found_factors:
    print(f"\n✅ 找到的因子: {found_factors}")
    
    # 筛选目标因子数据
    factors_df = df[df['feature'].isin(found_factors)].copy()
    factors_df['abs_pearson'] = factors_df['pearson_corr'].abs()
    
    print(f"\n总记录数: {len(factors_df)}")
    
    # 按符号和因子分析
    print("\n" + "=" * 100)
    print("1. 整体表现 (按符号和因子分组)")
    print("=" * 100)
    
    summary = factors_df.groupby(['symbol', 'feature']).agg({
        'pearson_corr': ['mean', 'std'],
        'abs_pearson': 'mean',
        'pearson_p': 'mean',
        'samples': 'mean'
    }).round(6)
    
    summary.columns = ['mean_corr', 'std_corr', 'abs_mean_corr', 'mean_pvalue', 'mean_samples']
    summary = summary.sort_values(['symbol', 'abs_mean_corr'], ascending=[False, False])
    print(summary.to_string())
    
    # 详细分析每个因子
    print("\n" + "=" * 100)
    print("2. 详细表现 (最佳时间框架和前向周期组合)")
    print("=" * 100)
    
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        print(f"\n{'=' * 100}")
        print(f"{symbol}")
        print(f"{'=' * 100}")
        
        symbol_data = factors_df[factors_df['symbol'] == symbol]
        
        for factor in found_factors:
            factor_data = symbol_data[symbol_data['feature'] == factor]
            
            if len(factor_data) == 0:
                print(f"\n❌ {factor}: 无数据")
                continue
            
            # 找到最佳组合
            best_idx = factor_data['abs_pearson'].idxmax()
            best_row = factor_data.loc[best_idx]
            
            # 统计信息
            mean_abs_corr = factor_data['abs_pearson'].mean()
            std_abs_corr = factor_data['abs_pearson'].std()
            significant_count = len(factor_data[factor_data['pearson_p'] < 0.05])
            
            print(f"\n📊 {factor}:")
            print(f"   平均绝对相关性: {mean_abs_corr:.4f} ± {std_abs_corr:.4f}")
            print(f"   显著相关性数量 (p < 0.05): {significant_count}/{len(factor_data)} ({100*significant_count/len(factor_data):.1f}%)")
            print(f"   最佳组合:")
            print(f"     - 时间框架: {best_row['timeframe']}")
            print(f"     - 前向周期: {best_row['forward_bars']} bars")
            print(f"     - Pearson 相关性: {best_row['pearson_corr']:.4f}")
            print(f"     - p-value: {best_row['pearson_p']:.4e} {'✅ 显著' if best_row['pearson_p'] < 0.05 else '❌ 不显著'}")
            print(f"     - 样本数: {int(best_row['samples'])}")
            
            # 显示所有组合的表现
            print(f"   所有组合的相关性范围: [{factor_data['pearson_corr'].min():.4f}, {factor_data['pearson_corr'].max():.4f}]")
            
            # 按时间框架分组
            print(f"   按时间框架的平均绝对相关性:")
            tf_summary = factor_data.groupby('timeframe')['abs_pearson'].mean().sort_values(ascending=False)
            for tf, val in tf_summary.items():
                print(f"     {tf}: {val:.4f}")
            
            # 按前向周期分组
            print(f"   按前向周期的平均绝对相关性:")
            fb_summary = factor_data.groupby('forward_bars')['abs_pearson'].mean().sort_values(ascending=False)
            for fb, val in fb_summary.items():
                print(f"     {fb} bars: {val:.4f}")
    
    # 对比分析
    print("\n" + "=" * 100)
    print("3. 因子排名对比")
    print("=" * 100)
    
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        print(f"\n{symbol} - 按平均绝对相关性排名:")
        symbol_data = factors_df[factors_df['symbol'] == symbol]
        ranking = symbol_data.groupby('feature')['abs_pearson'].mean().sort_values(ascending=False)
        for i, (factor, score) in enumerate(ranking.items(), 1):
            print(f"   {i}. {factor}: {score:.4f}")
    
    # 显著性分析
    print("\n" + "=" * 100)
    print("4. 显著性分析 (p < 0.05)")
    print("=" * 100)
    
    significant_df = factors_df[factors_df['pearson_p'] < 0.05].copy()
    
    if len(significant_df) > 0:
        print(f"\n显著相关性记录数: {len(significant_df)}/{len(factors_df)} ({100*len(significant_df)/len(factors_df):.1f}%)")
        
        sig_summary = significant_df.groupby(['symbol', 'feature']).agg({
            'abs_pearson': 'mean',
            'pearson_corr': 'mean',
            'pearson_p': 'mean'
        }).round(6)
        sig_summary.columns = ['mean_abs_corr', 'mean_corr', 'mean_pvalue']
        sig_summary = sig_summary.sort_values(['symbol', 'mean_abs_corr'], ascending=[False, False])
        print("\n显著相关性的平均表现:")
        print(sig_summary.to_string())
    else:
        print("\n⚠️  没有找到显著的相关性 (p < 0.05)")
    
    # 与所有 Alpha101 因子对比
    print("\n" + "=" * 100)
    print("5. 与所有 Alpha101 因子对比")
    print("=" * 100)
    
    all_alpha101 = df[df['feature'].str.startswith('alpha101_')].copy()
    all_alpha101['abs_pearson'] = all_alpha101['pearson_corr'].abs()
    
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        symbol_all = all_alpha101[all_alpha101['symbol'] == symbol]
        symbol_target = factors_df[factors_df['symbol'] == symbol]
        
        if len(symbol_all) > 0:
            all_mean = symbol_all.groupby('feature')['abs_pearson'].mean().sort_values(ascending=False)
            
            print(f"\n{symbol}:")
            print(f"   所有 Alpha101 因子平均绝对相关性中位数: {all_mean.median():.4f}")
            print(f"   所有 Alpha101 因子平均绝对相关性最大值: {all_mean.max():.4f}")
            
            if len(symbol_target) > 0:
                target_mean = symbol_target.groupby('feature')['abs_pearson'].mean()
                print(f"   目标因子平均绝对相关性中位数: {target_mean.median():.4f}")
                print(f"   目标因子平均绝对相关性最大值: {target_mean.max():.4f}")
                
                # 计算排名
                combined_ranking = all_mean.sort_values(ascending=False)
                print(f"\n   目标因子在全部 Alpha101 因子中的排名:")
                for factor in found_factors:
                    if factor in combined_ranking.index:
                        rank = combined_ranking.index.get_loc(factor) + 1
                        total = len(combined_ranking)
                        percentile = 100 * (1 - rank / total)
                        score = combined_ranking[factor]
                        print(f"     {factor}: 排名 {rank}/{total} (前 {percentile:.1f}%), 得分: {score:.4f}")

print("\n" + "=" * 100)
print("分析完成")
print("=" * 100)

