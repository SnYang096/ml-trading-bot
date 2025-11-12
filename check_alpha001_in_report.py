#!/usr/bin/env python3
"""检查 alpha101_001 是否在报告中，以及为什么它可能被过滤"""

import pandas as pd
import numpy as np

# 读取报告数据
df = pd.read_csv('results/timeframe_forward/alpha101_fixed/timeframe_forward_details.csv')

print("=" * 100)
print("检查 alpha101_001 在报告中的情况")
print("=" * 100)

# 检查 alpha101_001 是否存在
alpha_factors = sorted([f for f in df['feature'].unique() if f.startswith('alpha101_')])
print(f"\n报告中的 Alpha101 因子数量: {len(alpha_factors)}")

if 'alpha101_001' in df['feature'].values:
    print("\n✅ alpha101_001 存在于报告中")
    alpha001_data = df[df['feature'] == 'alpha101_001']
    print(f"   记录数: {len(alpha001_data)}")
    print(f"   相关性范围: [{alpha001_data['pearson_corr'].min():.4f}, {alpha001_data['pearson_corr'].max():.4f}]")
    print(f"   平均绝对相关性: {alpha001_data['pearson_corr'].abs().mean():.4f}")
    print(f"   p-value 范围: [{alpha001_data['pearson_p'].min():.4e}, {alpha001_data['pearson_p'].max():.4e}]")
else:
    print("\n❌ alpha101_001 不在报告中")
    print("\n可能的原因:")
    print("   1. alpha101_001 的值是常数（所有值都是 -0.5）")
    print("   2. 在计算相关性时，如果特征的标准差为0，会被跳过")
    print("   3. 相关性计算代码中检查: if np.isclose(np.std(feature_valid), 0)")
    
    # 检查代码逻辑
    print("\n检查代码逻辑:")
    print("   在 compute_correlations 函数中:")
    print("   if np.isclose(np.std(feature_valid), 0) or np.isclose(np.std(target_valid), 0):")
    print("       continue  # 跳过标准差为0的特征")
    
    # 测试 alpha001 的计算
    print("\n" + "=" * 100)
    print("测试 alpha001 的计算结果")
    print("=" * 100)
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    from data_tools.alpha_factors.alpha101_feature_engineer import Alpha101FeatureEngineer
    
    # 读取数据
    data_dir = Path('data/parquet_data')
    btc_files = sorted(data_dir.glob('BTCUSDT_2024-11.parquet'))
    
    if btc_files:
        df_data = pd.read_parquet(btc_files[0])
        df_data = df_data.set_index('timestamp') if 'timestamp' in df_data.columns else df_data
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if all(col in df_data.columns for col in required_cols):
            engineer = Alpha101FeatureEngineer()
            
            # 重置索引以便特征工程
            df_reset = df_data.reset_index()
            if 'symbol' not in df_reset.columns:
                df_reset['symbol'] = 'BTCUSDT'
            
            result_df = engineer.compute(df_reset, symbol='BTCUSDT')
            
            if 'alpha101_001' in result_df.columns:
                alpha001_col = result_df['alpha101_001']
                print(f"\n✅ alpha101_001 在特征工程中生成")
                print(f"   数据行数: {len(alpha001_col)}")
                print(f"   非空值数量: {alpha001_col.notna().sum()}")
                print(f"   NaN 数量: {alpha001_col.isna().sum()}")
                print(f"   唯一值数量: {alpha001_col.nunique()}")
                print(f"   值范围: [{alpha001_col.min():.4f}, {alpha001_col.max():.4f}]")
                print(f"   标准差: {alpha001_col.std():.4f}")
                print(f"   是否接近常数: {np.isclose(alpha001_col.std(), 0)}")
                
                if np.isclose(alpha001_col.std(), 0):
                    print("\n   ⚠️  alpha101_001 的值是常数，标准差为0")
                    print("   这就是为什么它没有出现在相关性报告中")
                    print("   在 compute_correlations 函数中，常数特征会被跳过")
                    
                    # 检查值
                    unique_values = alpha001_col.unique()
                    print(f"\n   唯一值: {unique_values}")
                    print(f"   所有值是否相同: {len(unique_values) == 1}")
                    
                    if len(unique_values) == 1:
                        print(f"   所有值都是: {unique_values[0]}")
                        print("\n   这可能是因为 alpha001 的实现有问题，或者")
                        print("   在当前数据上，它总是返回相同的值")

print("\n" + "=" * 100)
print("检查完成")
print("=" * 100)

