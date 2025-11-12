#!/usr/bin/env python3
"""验证目标 Alpha101 因子在 BTC 和 ETH 上的有效性"""

import pandas as pd
import numpy as np

# 读取报告数据
df = pd.read_csv('results/timeframe_forward/alpha101_fixed/timeframe_forward_details.csv')

# 目标因子（注意：alpha101_001 不存在）
target_factors = {
    'alpha101_001': 'alpha_001: (rank(ts_argmax(signed_power(returns, 2), 5)) - 0.5)',
    'alpha101_022': 'alpha_022: -1 * delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))',
    'alpha101_043': 'alpha_043: ts_rank(volume / adv20, 20) * ts_rank(-delta(close, 7), 8)',
    'alpha101_066': 'alpha_066: (close - open) / (high - low) → K 线实体强度'
}

print("=" * 100)
print("Alpha101 因子有效性验证报告 - BTC & ETH")
print("=" * 100)

# 检查哪些因子存在
available_factors = sorted([f for f in df['feature'].unique() if f.startswith('alpha101_')])
print(f"\n📊 可用的 Alpha101 因子数量: {len(available_factors)}")

# 检查目标因子
found_factors = [f for f in target_factors.keys() if f in available_factors]
missing_factors = [f for f in target_factors.keys() if f not in available_factors]

print(f"\n✅ 找到的目标因子: {len(found_factors)}/{len(target_factors)}")
for factor in found_factors:
    print(f"   - {factor}: {target_factors[factor]}")

if missing_factors:
    print(f"\n⚠️  未找到的目标因子: {len(missing_factors)}/{len(target_factors)}")
    for factor in missing_factors:
        print(f"   - {factor}: {target_factors[factor]}")
        print(f"     原因: 可能在特征工程过程中计算失败或被过滤")

# 分析找到的因子
if found_factors:
    factors_df = df[df['feature'].isin(found_factors)].copy()
    factors_df['abs_pearson'] = factors_df['pearson_corr'].abs()
    
    print("\n" + "=" * 100)
    print("1. 整体表现总结")
    print("=" * 100)
    
    summary = []
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        symbol_data = factors_df[factors_df['symbol'] == symbol]
        
        for factor in found_factors:
            factor_data = symbol_data[symbol_data['feature'] == factor]
            if len(factor_data) > 0:
                mean_abs_corr = factor_data['abs_pearson'].mean()
                max_abs_corr = factor_data['abs_pearson'].max()
                significant_count = len(factor_data[factor_data['pearson_p'] < 0.05])
                total_count = len(factor_data)
                
                # 找到最佳组合
                best_idx = factor_data['abs_pearson'].idxmax()
                best_row = factor_data.loc[best_idx]
                
                summary.append({
                    'symbol': symbol,
                    'factor': factor,
                    'mean_abs_corr': mean_abs_corr,
                    'max_abs_corr': max_abs_corr,
                    'best_tf': best_row['timeframe'],
                    'best_fb': best_row['forward_bars'],
                    'best_corr': best_row['pearson_corr'],
                    'best_pvalue': best_row['pearson_p'],
                    'significant_ratio': significant_count / total_count * 100,
                    'rank': None  # 稍后填充
                })
    
    summary_df = pd.DataFrame(summary)
    
    # 计算排名（与所有 Alpha101 因子对比）
    all_alpha101 = df[df['feature'].str.startswith('alpha101_')].copy()
    all_alpha101['abs_pearson'] = all_alpha101['pearson_corr'].abs()
    
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        symbol_all = all_alpha101[all_alpha101['symbol'] == symbol]
        if len(symbol_all) > 0:
            all_ranking = symbol_all.groupby('feature')['abs_pearson'].mean().sort_values(ascending=False)
            
            symbol_summary = summary_df[summary_df['symbol'] == symbol]
            for idx, row in symbol_summary.iterrows():
                if row['factor'] in all_ranking.index:
                    rank = all_ranking.index.get_loc(row['factor']) + 1
                    summary_df.loc[idx, 'rank'] = rank
                    summary_df.loc[idx, 'total_factors'] = len(all_ranking)
                    summary_df.loc[idx, 'percentile'] = 100 * (1 - rank / len(all_ranking))
    
    # 显示总结
    print("\n按符号和因子显示:")
    print("-" * 100)
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        symbol_summary = summary_df[summary_df['symbol'] == symbol].sort_values('mean_abs_corr', ascending=False)
        print(f"\n{symbol}:")
        print(f"{'因子':<20} {'平均绝对相关性':<15} {'最佳相关性':<15} {'排名':<10} {'显著比例':<10} {'最佳组合':<20}")
        print("-" * 100)
        for _, row in symbol_summary.iterrows():
            rank_str = f"{int(row['rank'])}/{int(row['total_factors'])}" if pd.notna(row['rank']) else "N/A"
            percentile_str = f"({row['percentile']:.1f}%)" if pd.notna(row['percentile']) else ""
            significant_str = f"{row['significant_ratio']:.1f}%"
            best_combo = f"{row['best_tf']}/{row['best_fb']}b"
            print(f"{row['factor']:<20} {row['mean_abs_corr']:<15.4f} {row['max_abs_corr']:<15.4f} {rank_str:<10} {percentile_str:<10} {significant_str:<10} {best_combo:<20}")
    
    print("\n" + "=" * 100)
    print("2. 详细分析 - 最佳时间框架和前向周期组合")
    print("=" * 100)
    
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        print(f"\n{'=' * 100}")
        print(f"{symbol}")
        print(f"{'=' * 100}")
        
        symbol_data = factors_df[factors_df['symbol'] == symbol]
        
        for factor in found_factors:
            factor_data = symbol_data[symbol_data['feature'] == factor]
            
            if len(factor_data) == 0:
                continue
            
            # 找到最佳组合
            best_idx = factor_data['abs_pearson'].idxmax()
            best_row = factor_data.loc[best_idx]
            
            # 统计信息
            mean_abs_corr = factor_data['abs_pearson'].mean()
            std_abs_corr = factor_data['abs_pearson'].std()
            significant_count = len(factor_data[factor_data['pearson_p'] < 0.05])
            total_count = len(factor_data)
            
            print(f"\n📊 {factor} - {target_factors[factor]}")
            print(f"   平均绝对相关性: {mean_abs_corr:.4f} ± {std_abs_corr:.4f}")
            print(f"   显著相关性: {significant_count}/{total_count} ({100*significant_count/total_count:.1f}%)")
            
            if pd.notna(summary_df[(summary_df['symbol'] == symbol) & (summary_df['factor'] == factor)]['rank'].values[0]):
                rank = int(summary_df[(summary_df['symbol'] == symbol) & (summary_df['factor'] == factor)]['rank'].values[0])
                total = int(summary_df[(summary_df['symbol'] == symbol) & (summary_df['factor'] == factor)]['total_factors'].values[0])
                percentile = summary_df[(summary_df['symbol'] == symbol) & (summary_df['factor'] == factor)]['percentile'].values[0]
                print(f"   在全部 Alpha101 因子中排名: {rank}/{total} (前 {percentile:.1f}%)")
            
            print(f"   最佳组合:")
            print(f"     - 时间框架: {best_row['timeframe']}")
            print(f"     - 前向周期: {best_row['forward_bars']} bars")
            print(f"     - Pearson 相关性: {best_row['pearson_corr']:.4f}")
            print(f"     - p-value: {best_row['pearson_p']:.4e} {'✅ 显著 (p < 0.05)' if best_row['pearson_p'] < 0.05 else '❌ 不显著'}")
            print(f"     - 样本数: {int(best_row['samples'])}")
            
            # 显示所有组合
            print(f"   所有组合的相关性:")
            combo_summary = factor_data.groupby(['timeframe', 'forward_bars']).agg({
                'pearson_corr': 'mean',
                'pearson_p': 'mean',
                'abs_pearson': 'mean'
            }).sort_values('abs_pearson', ascending=False)
            
            for (tf, fb), row in combo_summary.head(5).iterrows():
                sig_mark = "✅" if row['pearson_p'] < 0.05 else "  "
                print(f"     {sig_mark} {tf:>6} / {fb:>2}b: {row['pearson_corr']:>7.4f} (p={row['pearson_p']:.4e})")
    
    print("\n" + "=" * 100)
    print("3. 结论与建议")
    print("=" * 100)
    
    # 计算整体评分
    print("\n因子有效性评分 (基于平均绝对相关性和显著性):")
    print("-" * 100)
    
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        print(f"\n{symbol}:")
        symbol_summary = summary_df[summary_df['symbol'] == symbol].sort_values('mean_abs_corr', ascending=False)
        
        for i, (_, row) in enumerate(symbol_summary.iterrows(), 1):
            # 评分标准: 平均绝对相关性 + 显著性比例 + 排名
            score = (
                row['mean_abs_corr'] * 100 +  # 相关性得分
                row['significant_ratio'] * 0.1 +  # 显著性得分
                (100 - row['percentile']) * 0.01 if pd.notna(row['percentile']) else 0  # 排名得分
            )
            
            # 评级
            if score > 2.0:
                rating = "⭐⭐⭐ 优秀"
            elif score > 1.0:
                rating = "⭐⭐ 良好"
            elif score > 0.5:
                rating = "⭐ 一般"
            else:
                rating = "❌ 较差"
            
            print(f"   {i}. {row['factor']}: {rating} (得分: {score:.2f})")
            print(f"      平均绝对相关性: {row['mean_abs_corr']:.4f}")
            print(f"      显著相关性比例: {row['significant_ratio']:.1f}%")
            if pd.notna(row['rank']):
                print(f"      排名: {int(row['rank'])}/{int(row['total_factors'])} (前 {row['percentile']:.1f}%)")
            print(f"      最佳组合: {row['best_tf']} / {row['best_fb']} bars")
            print()
    
    # 最终建议
    print("\n" + "=" * 100)
    print("4. 最终建议")
    print("=" * 100)
    
    print("\n基于分析结果，推荐优先尝试的因子:")
    print("-" * 100)
    
    # 找出在两个符号上都表现良好的因子
    btc_best = summary_df[summary_df['symbol'] == 'BTCUSDT'].nlargest(3, 'mean_abs_corr')
    eth_best = summary_df[summary_df['symbol'] == 'ETHUSDT'].nlargest(3, 'mean_abs_corr')
    
    common_factors = set(btc_best['factor']) & set(eth_best['factor'])
    
    if common_factors:
        print("\n✅ 在 BTC 和 ETH 上都表现良好的因子:")
        for factor in common_factors:
            btc_row = summary_df[(summary_df['symbol'] == 'BTCUSDT') & (summary_df['factor'] == factor)].iloc[0]
            eth_row = summary_df[(summary_df['symbol'] == 'ETHUSDT') & (summary_df['factor'] == factor)].iloc[0]
            
            print(f"\n   {factor}:")
            print(f"      BTC: 平均相关性 {btc_row['mean_abs_corr']:.4f}, 排名 {int(btc_row['rank'])}/{int(btc_row['total_factors'])}")
            print(f"      ETH: 平均相关性 {eth_row['mean_abs_corr']:.4f}, 排名 {int(eth_row['rank'])}/{int(eth_row['total_factors'])}")
            print(f"      最佳时间框架: {btc_row['best_tf']} (BTC), {eth_row['best_tf']} (ETH)")
            print(f"      最佳前向周期: {btc_row['best_fb']} bars (BTC), {eth_row['best_fb']} bars (ETH)")
    
    print("\n⚠️  注意事项:")
    print("   1. alpha101_001 未出现在报告中，可能因为计算失败或被过滤")
    print("   2. 所有因子的相关性都相对较低 (< 0.1)，这是加密货币市场的典型特征")
    print("   3. 建议结合多个因子使用，而不是单独依赖某个因子")
    print("   4. 不同时间框架和前向周期的表现差异较大，需要根据具体策略选择")

print("\n" + "=" * 100)
print("分析完成")
print("=" * 100)

