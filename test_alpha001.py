#!/usr/bin/env python3
"""测试 alpha001 为什么没有被包含在特征中"""

import sys
import traceback
import pandas as pd
import numpy as np
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_tools.alpha_factors.alpha101_feature_engineer import Alpha101FeatureEngineer

# 创建测试数据
print("=" * 100)
print("测试 alpha001 计算")
print("=" * 100)

# 读取实际数据
data_dir = Path('data/parquet_data')
btc_files = sorted(data_dir.glob('BTCUSDT_2024-11.parquet'))

if btc_files:
    df = pd.read_parquet(btc_files[0])
    df = df.set_index('timestamp') if 'timestamp' in df.columns else df
    
    # 确保有必要的列
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ 缺少列: {missing_cols}")
        sys.exit(1)
    
    print(f"✅ 数据加载成功: {len(df)} 行")
    print(f"   列: {df.columns.tolist()}")
    print(f"   日期范围: {df.index.min()} 到 {df.index.max()}")
    
    # 测试 Alpha101 特征工程
    print("\n" + "=" * 100)
    print("测试 Alpha101 特征工程")
    print("=" * 100)
    
    try:
        engineer = Alpha101FeatureEngineer()
        print(f"✅ Alpha101FeatureEngineer 创建成功")
        print(f"   可用的 alpha 函数: {len(engineer._alpha_funcs)} 个")
        
        # 检查 alpha001 是否在函数列表中
        alpha001_name = None
        for name in engineer._alpha_funcs.keys():
            if '001' in name or name == 'alpha001':
                alpha001_name = name
                print(f"   ✅ 找到 alpha001: {name}")
                break
        
        if not alpha001_name:
            print("   ❌ 未找到 alpha001")
            print(f"   可用的 alpha 函数名称:")
            for name in sorted(engineer._alpha_funcs.keys()):
                print(f"      - {name}")
        else:
            # 检查 alpha001 的参数
            import inspect
            func = engineer._alpha_funcs[alpha001_name]
            sig = inspect.signature(func)
            print(f"   alpha001 参数: {list(sig.parameters.keys())}")
            
            # 尝试计算
            print("\n" + "=" * 100)
            print("尝试计算 alpha001")
            print("=" * 100)
            
            try:
                # 准备数据
                base = df[required_cols].astype(float).copy()
                base.columns = pd.Index(base.columns, name="field")
                
                data_frames = {
                    "o": engineer._to_panel(base["open"], "asset"),
                    "h": engineer._to_panel(base["high"], "asset"),
                    "l": engineer._to_panel(base["low"], "asset"),
                    "c": engineer._to_panel(base["close"], "asset"),
                    "v": engineer._to_panel(base["volume"], "asset"),
                }
                
                # 计算 returns
                data_frames["r"] = data_frames["c"].pct_change().replace([-np.inf, np.inf], np.nan).fillna(0.0)
                
                # 解析参数
                args = []
                for param in sig.parameters:
                    try:
                        arg = engineer._resolve_argument(param, data_frames)
                        args.append(arg)
                        print(f"   ✅ 参数 {param} 解析成功: shape {arg.shape}")
                    except KeyError as e:
                        print(f"   ❌ 参数 {param} 解析失败: {e}")
                        raise
                
                # 调用函数
                print(f"\n   调用 alpha001({', '.join([f'{p}={a.shape}' for p, a in zip(sig.parameters.keys(), args)])})")
                result = func(*args)
                print(f"   ✅ alpha001 计算成功")
                print(f"      结果类型: {type(result)}")
                if hasattr(result, 'shape'):
                    print(f"      结果形状: {result.shape}")
                if hasattr(result, 'index'):
                    print(f"      索引类型: {type(result.index)}")
                
                # 格式化输出
                formatted = engineer._format_output(result, "asset")
                if formatted is not None:
                    print(f"   ✅ 格式化成功")
                    print(f"      格式化结果类型: {type(formatted)}")
                    print(f"      格式化结果长度: {len(formatted)}")
                    print(f"      非空值数量: {formatted.notna().sum()}")
                    print(f"      NaN 数量: {formatted.isna().sum()}")
                    print(f"      值范围: [{formatted.min():.4f}, {formatted.max():.4f}]")
                else:
                    print(f"   ❌ 格式化返回 None")
                    
            except Exception as e:
                print(f"   ❌ 计算失败: {e}")
                print(f"   错误类型: {type(e).__name__}")
                traceback.print_exc()
            
            # 测试完整的特征工程
            print("\n" + "=" * 100)
            print("测试完整的特征工程流程")
            print("=" * 100)
            
            try:
                # 重置索引以便特征工程
                df_reset = df.reset_index()
                if 'symbol' not in df_reset.columns:
                    df_reset['symbol'] = 'BTCUSDT'
                
                result_df = engineer.compute(df_reset, symbol='BTCUSDT')
                print(f"✅ 特征工程成功")
                print(f"   生成的特征数: {len(result_df.columns)}")
                print(f"   数据行数: {len(result_df)}")
                
                # 检查 alpha101_001 是否存在
                if 'alpha101_001' in result_df.columns:
                    print(f"   ✅ alpha101_001 存在")
                    alpha001_col = result_df['alpha101_001']
                    print(f"      非空值数量: {alpha001_col.notna().sum()}")
                    print(f"      NaN 数量: {alpha001_col.isna().sum()}")
                    print(f"      值范围: [{alpha001_col.min():.4f}, {alpha001_col.max():.4f}]")
                else:
                    print(f"   ❌ alpha101_001 不存在")
                    print(f"   可用的 alpha101 特征:")
                    alpha101_cols = [col for col in result_df.columns if col.startswith('alpha101_')]
                    for col in sorted(alpha101_cols)[:10]:
                        print(f"      - {col}")
                    if len(alpha101_cols) > 10:
                        print(f"      ... 还有 {len(alpha101_cols) - 10} 个")
                        
            except Exception as e:
                print(f"❌ 特征工程失败: {e}")
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ 错误: {e}")
        traceback.print_exc()
else:
    print("❌ 未找到 BTCUSDT 数据文件")

print("\n" + "=" * 100)
print("测试完成")
print("=" * 100)

