#!/usr/bin/env python3
"""调试 alpha001 为什么返回常数"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_tools.alpha_factors.alpha101_feature_engineer import Alpha101FeatureEngineer
from data_tools.alpha_factors import alpha101_raw
from data_tools.alpha_factors.alpha_utils import rank, ts_argmax, ts_std, power

print("=" * 100)
print("调试 alpha001 计算")
print("=" * 100)

# 读取数据
data_dir = Path('data/parquet_data')
btc_files = sorted(data_dir.glob('BTCUSDT_2024-11.parquet'))

if btc_files:
    df = pd.read_parquet(btc_files[0])
    df = df.set_index('timestamp') if 'timestamp' in df.columns else df
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    base = df[required_cols].astype(float).copy()
    base.columns = pd.Index(base.columns, name="field")
    
    engineer = Alpha101FeatureEngineer()
    
    # 准备数据
    data_frames = {
        "o": engineer._to_panel(base["open"], "asset"),
        "h": engineer._to_panel(base["high"], "asset"),
        "l": engineer._to_panel(base["low"], "asset"),
        "c": engineer._to_panel(base["close"], "asset"),
        "v": engineer._to_panel(base["volume"], "asset"),
    }
    
    # 计算 returns
    data_frames["r"] = data_frames["c"].pct_change().replace([-np.inf, np.inf], np.nan).fillna(0.0)
    
    c = data_frames["c"].copy()
    r = data_frames["r"].copy()
    
    print(f"\n原始数据:")
    print(f"  c (close) shape: {c.shape}")
    print(f"  r (returns) shape: {r.shape}")
    print(f"  c 值范围: [{c.min().min():.2f}, {c.max().max():.2f}]")
    print(f"  r 值范围: [{r.min().min():.4f}, {r.max().max():.4f}]")
    print(f"  r < 0 的数量: {(r < 0).sum().sum()}")
    print(f"  r >= 0 的数量: {(r >= 0).sum().sum()}")
    
    # 执行 alpha001 的逻辑
    print(f"\n执行 alpha001 逻辑:")
    print(f"  c[r < 0] = ts_std(r, 20)")
    
    # 计算 ts_std(r, 20)
    r_std = ts_std(r, 20)
    print(f"  ts_std(r, 20) shape: {r_std.shape}")
    print(f"  ts_std(r, 20) 值范围: [{r_std.min().min():.6f}, {r_std.max().max():.6f}]")
    print(f"  ts_std(r, 20) 非空值数量: {r_std.notna().sum().sum()}")
    
    # 修改 c
    c_modified = c.copy()
    mask = r < 0
    print(f"  r < 0 的 mask shape: {mask.shape}")
    print(f"  mask True 数量: {mask.sum().sum()}")
    
    # 注意：这里需要广播
    c_modified[mask] = r_std[mask]
    print(f"  修改后的 c 值范围: [{c_modified.min().min():.2f}, {c_modified.max().max():.2f}]")
    
    # 计算 power(c, 2)
    c_powered = power(c_modified, 2)
    print(f"  power(c, 2) shape: {c_powered.shape}")
    print(f"  power(c, 2) 值范围: [{c_powered.min().min():.2f}, {c_powered.max().max():.2f}]")
    
    # 计算 ts_argmax(power(c, 2), 5)
    try:
        c_argmax = ts_argmax(c_powered, 5)
        print(f"  ts_argmax(power(c, 2), 5) shape: {c_argmax.shape}")
        print(f"  ts_argmax 值范围: [{c_argmax.min().min():.2f}, {c_argmax.max().max():.2f}]")
        print(f"  ts_argmax 唯一值数量: {c_argmax.nunique(axis=0).sum()}")
    except Exception as e:
        print(f"  ❌ ts_argmax 计算失败: {e}")
        import traceback
        traceback.print_exc()
        c_argmax = None
    
    if c_argmax is not None:
        # 计算 rank
        try:
            c_ranked = rank(c_argmax)
            print(f"  rank(ts_argmax) shape: {c_ranked.shape}")
            print(f"  rank 值范围: [{c_ranked.min().min():.4f}, {c_ranked.max().max():.4f}]")
            print(f"  rank 唯一值数量: {c_ranked.nunique(axis=0).sum()}")
        except Exception as e:
            print(f"  ❌ rank 计算失败: {e}")
            import traceback
            traceback.print_exc()
            c_ranked = None
        
        if c_ranked is not None:
            # 计算最终结果 (rank * -0.5)
            try:
                result = c_ranked.mul(-0.5)
                print(f"  result (rank * -0.5) shape: {result.shape}")
                print(f"  result 值范围: [{result.min().min():.4f}, {result.max().max():.4f}]")
                print(f"  result 唯一值数量: {result.nunique(axis=0).sum()}")
                
                # 尝试格式化
                formatted = engineer._format_output(result, "asset")
                if formatted is not None:
                    print(f"  formatted 值范围: [{formatted.min():.4f}, {formatted.max():.4f}]")
                    print(f"  formatted 唯一值数量: {formatted.nunique()}")
                    print(f"  formatted 标准差: {formatted.std():.4f}")
                    print(f"  是否接近常数: {np.isclose(formatted.std(), 0)}")
            except Exception as e:
                print(f"  ❌ 最终计算失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 直接调用 alpha001 函数
    print(f"\n直接调用 alpha001 函数:")
    try:
        result_direct = alpha101_raw.alpha001(c.copy(), r.copy())
        print(f"  结果类型: {type(result_direct)}")
        print(f"  结果形状: {result_direct.shape if hasattr(result_direct, 'shape') else 'N/A'}")
        
        formatted_direct = engineer._format_output(result_direct, "asset")
        if formatted_direct is not None:
            print(f"  格式化结果值范围: [{formatted_direct.min():.4f}, {formatted_direct.max():.4f}]")
            print(f"  格式化结果唯一值数量: {formatted_direct.nunique()}")
            print(f"  格式化结果标准差: {formatted_direct.std():.4f}")
            print(f"  是否接近常数: {np.isclose(formatted_direct.std(), 0)}")
    except Exception as e:
        print(f"  ❌ 直接调用失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 100)
print("调试完成")
print("=" * 100)

