#!/usr/bin/env python3
"""测试时序适配版本的 Alpha101 因子在 BTC/ETH 上的表现"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_tools.alpha_factors.alpha101_timeseries_adapted import (
    compute_adapted_alpha101_factors,
    alpha001_ts,
    alpha022_ts,
    alpha043_ts,
    alpha066_ts
)

print("=" * 100)
print("测试时序适配版本的 Alpha101 因子")
print("=" * 100)

# 读取数据
data_dir = Path('data/parquet_data')
symbols = ['BTCUSDT', 'ETHUSDT']

results = []

for symbol in symbols:
    print(f"\n{'=' * 100}")
    print(f"分析 {symbol}")
    print(f"{'=' * 100}")
    
    # 读取数据
    files = sorted(data_dir.glob(f'{symbol}_2024-11.parquet'))
    if not files:
        print(f"❌ 未找到 {symbol} 数据文件")
        continue
    
    df = pd.read_parquet(files[0])
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    # 确保有必要的列
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺少列: {missing_cols}")
        continue
    
    print(f"✅ 数据加载成功: {len(df)} 行")
    print(f"   日期范围: {df.index.min()} 到 {df.index.max()}")
    
    # 计算时序适配的 Alpha101 因子
    print(f"\n计算时序适配的 Alpha101 因子...")
    try:
        factors = compute_adapted_alpha101_factors(df, use_ts_rank=True)
        print(f"✅ 因子计算成功")
        print(f"   生成的因子: {factors.columns.tolist()}")
        print(f"   因子统计:")
        print(factors.describe())
    except Exception as e:
        print(f"❌ 因子计算失败: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # 计算未来收益率（不同周期）
    forward_bars = [3, 6, 12, 24]
    returns = df['close'].pct_change().fillna(0)
    log_price = np.log(df['close'])
    
    print(f"\n{'=' * 100}")
    print(f"计算相关性")
    print(f"{'=' * 100}")
    
    for forward_bar in forward_bars:
        # 计算未来收益率
        future_return = (log_price.shift(-forward_bar) - log_price).fillna(0)
        
        # 对齐数据
        aligned_data = pd.concat([factors, future_return], axis=1)
        aligned_data.columns = list(factors.columns) + ['future_return']
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) < 100:
            print(f"  ⚠️  前向 {forward_bar} bars: 数据不足 ({len(aligned_data)} 行)")
            continue
        
        print(f"\n  前向 {forward_bar} bars (样本数: {len(aligned_data)}):")
        print(f"  {'因子':<25} {'相关性':<12} {'p-value':<12} {'显著性':<8}")
        print(f"  {'-' * 70}")
        
        for factor_col in factors.columns:
            factor_values = aligned_data[factor_col].values
            target_values = aligned_data['future_return'].values
            
            # 检查标准差
            if np.isclose(np.std(factor_values), 0) or np.isclose(np.std(target_values), 0):
                print(f"  {factor_col:<25} {'N/A':<12} {'N/A':<12} {'常数'}")
                continue
            
            # 计算相关性
            corr, p_value = pearsonr(factor_values, target_values)
            is_significant = p_value < 0.05
            
            significance = "✅ 显著" if is_significant else "  "
            print(f"  {factor_col:<25} {corr:>10.4f}  {p_value:>10.4e}  {significance}")
            
            # 保存结果
            results.append({
                'symbol': symbol,
                'factor': factor_col,
                'forward_bars': forward_bar,
                'pearson_corr': corr,
                'pearson_p': p_value,
                'samples': len(aligned_data),
                'is_significant': is_significant
            })

# 生成总结报告
print(f"\n{'=' * 100}")
print("总结报告")
print(f"{'=' * 100}")

if results:
    results_df = pd.DataFrame(results)
    results_df['abs_pearson'] = results_df['pearson_corr'].abs()
    
    print(f"\n总体统计:")
    print(f"  总记录数: {len(results_df)}")
    print(f"  显著相关性数量: {results_df['is_significant'].sum()} ({100*results_df['is_significant'].sum()/len(results_df):.1f}%)")
    print(f"  平均绝对相关性: {results_df['abs_pearson'].mean():.4f}")
    print(f"  最大绝对相关性: {results_df['abs_pearson'].max():.4f}")
    
    print(f"\n按因子分组:")
    print(f"  {'因子':<25} {'平均绝对相关性':<15} {'显著数量':<12} {'最佳相关性':<12}")
    print(f"  {'-' * 70}")
    
    factor_summary = results_df.groupby('factor').agg({
        'abs_pearson': 'mean',
        'is_significant': 'sum',
        'pearson_corr': lambda x: x.loc[x.abs().idxmax()]
    }).sort_values('abs_pearson', ascending=False)
    
    for factor, row in factor_summary.iterrows():
        print(f"  {factor:<25} {row['abs_pearson']:>13.4f}  {int(row['is_significant']):>10}/{len(results_df[results_df['factor']==factor]):<2}  {row['pearson_corr']:>10.4f}")
    
    print(f"\n按符号分组:")
    for symbol in symbols:
        symbol_data = results_df[results_df['symbol'] == symbol]
        if len(symbol_data) > 0:
            print(f"\n  {symbol}:")
            print(f"    平均绝对相关性: {symbol_data['abs_pearson'].mean():.4f}")
            print(f"    显著相关性数量: {symbol_data['is_significant'].sum()}/{len(symbol_data)} ({100*symbol_data['is_significant'].sum()/len(symbol_data):.1f}%)")
            print(f"    最佳因子:")
            best_factor = symbol_data.loc[symbol_data['abs_pearson'].idxmax()]
            print(f"      {best_factor['factor']}: {best_factor['pearson_corr']:.4f} (p={best_factor['pearson_p']:.4e}, {best_factor['forward_bars']} bars)")
    
    # 保存结果
    output_file = 'results/timeframe_forward/alpha101_ts_adapted_test.csv'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ 结果已保存到: {output_file}")
else:
    print("❌ 没有结果可报告")

print(f"\n{'=' * 100}")
print("测试完成")
print(f"{'=' * 100}")

