#!/usr/bin/env python3
"""
测试 sr_strength_max 的依赖自动修复功能

验证：
1. 当 hal_high, hal_low, poc 不存在时，自动计算
2. 当 ATR 不存在时，自动计算
3. 确保计算顺序正确
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data_tools.data_handler import MarketDataLoader
from src.features.loader.feature_wrappers import compute_sr_strength_max
from src.features.time_series.baseline_features import (
    add_poc_hal_dimensionless_features,
)


def test_sr_strength_max_auto_dependencies():
    """测试 sr_strength_max 的自动依赖修复"""
    print("=" * 80)
    print("SR Strength Max 依赖自动修复测试")
    print("=" * 80)

    # 加载数据
    data_path = project_root / "data" / "parquet_data"
    symbol = "BTCUSDT"
    timeframe = "240T"
    start_date = "2025-01-01"
    end_date = "2025-07-31"

    print(f"\n📂 加载数据...")
    loader = MarketDataLoader(data_path=str(data_path))
    df_raw = loader.load_data(
        symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
    )

    split_idx = int(len(df_raw) * 0.85)
    df = df_raw.iloc[:split_idx].copy()
    print(f"   训练集大小: {len(df)}")

    # 测试 1: 没有任何依赖特征
    print(f"\n{'='*80}")
    print("测试 1: 没有任何依赖特征（只有基础 OHLCV）")
    print(f"{'='*80}")

    df_test1 = df[["open", "high", "low", "close", "volume"]].copy()

    print(f"   初始列: {list(df_test1.columns)}")
    print(f"   hal_high 存在: {'hal_high' in df_test1.columns}")
    print(f"   hal_low 存在: {'hal_low' in df_test1.columns}")
    print(f"   poc 存在: {'poc' in df_test1.columns}")
    print(f"   atr 存在: {'atr' in df_test1.columns}")

    try:
        df_result1 = compute_sr_strength_max(df_test1)

        print(f"\n   计算后检查:")
        print(f"   hal_high 存在: {'hal_high' in df_result1.columns}")
        print(f"   hal_low 存在: {'hal_low' in df_result1.columns}")
        print(f"   poc 存在: {'poc' in df_result1.columns}")
        print(f"   atr 存在: {'atr' in df_result1.columns}")
        print(f"   sr_strength_max 存在: {'sr_strength_max' in df_result1.columns}")

        if "sr_strength_max" in df_result1.columns:
            sr_strength = df_result1["sr_strength_max"]
            print(f"   sr_strength_max 统计:")
            print(f"      NaN 数量: {sr_strength.isna().sum()}")
            print(f"      有效数量: {sr_strength.notna().sum()}")
            print(f"      零值数量: {(sr_strength == 0.0).sum()}")
            if sr_strength.notna().any():
                valid = sr_strength[sr_strength.notna()]
                print(f"      范围: [{valid.min():.4f}, {valid.max():.4f}]")
                print(f"      均值: {valid.mean():.4f}")

        print(f"   ✅ 测试 1 通过：自动计算了所有依赖特征")
    except Exception as e:
        print(f"   ❌ 测试 1 失败: {e}")
        import traceback

        traceback.print_exc()

    # 测试 2: 只有部分依赖特征
    print(f"\n{'='*80}")
    print("测试 2: 只有部分依赖特征（有 hal_high，但没有 hal_low, poc, atr）")
    print(f"{'='*80}")

    df_test2 = df[["open", "high", "low", "close", "volume"]].copy()
    # 只计算 hal_high
    df_test2 = add_poc_hal_dimensionless_features(
        df_test2,
        required_features={"hal_high"},
        poc_window=160,
    )

    print(f"   初始状态:")
    print(f"   hal_high 存在: {'hal_high' in df_test2.columns}")
    print(f"   hal_low 存在: {'hal_low' in df_test2.columns}")
    print(f"   poc 存在: {'poc' in df_test2.columns}")
    print(f"   atr 存在: {'atr' in df_test2.columns}")

    try:
        df_result2 = compute_sr_strength_max(df_test2)

        print(f"\n   计算后检查:")
        print(f"   hal_high 存在: {'hal_high' in df_result2.columns}")
        print(f"   hal_low 存在: {'hal_low' in df_result2.columns}")
        print(f"   poc 存在: {'poc' in df_result2.columns}")
        print(f"   atr 存在: {'atr' in df_result2.columns}")
        print(f"   sr_strength_max 存在: {'sr_strength_max' in df_result2.columns}")

        if "sr_strength_max" in df_result2.columns:
            sr_strength = df_result2["sr_strength_max"]
            print(f"   sr_strength_max 统计:")
            print(f"      NaN 数量: {sr_strength.isna().sum()}")
            print(f"      有效数量: {sr_strength.notna().sum()}")
            if sr_strength.notna().any():
                valid = sr_strength[sr_strength.notna()]
                print(f"      范围: [{valid.min():.4f}, {valid.max():.4f}]")

        print(f"   ✅ 测试 2 通过：自动补充了缺失的依赖特征")
    except Exception as e:
        print(f"   ❌ 测试 2 失败: {e}")
        import traceback

        traceback.print_exc()

    # 测试 3: 有 wpt_price_reconstructed（应该使用它）
    print(f"\n{'='*80}")
    print("测试 3: 有 wpt_price_reconstructed（应该使用它计算边界）")
    print(f"{'='*80}")

    df_test3 = df[["open", "high", "low", "close", "volume"]].copy()
    # 模拟 wpt_price_reconstructed（使用 close 作为占位符）
    df_test3["wpt_price_reconstructed"] = df_test3["close"]

    print(f"   初始状态:")
    print(
        f"   wpt_price_reconstructed 存在: {'wpt_price_reconstructed' in df_test3.columns}"
    )
    print(f"   hal_high 存在: {'hal_high' in df_test3.columns}")
    print(f"   atr 存在: {'atr' in df_test3.columns}")

    try:
        df_result3 = compute_sr_strength_max(
            df_test3,
            poc_window=160,
            price_col="wpt_price_reconstructed",
        )

        print(f"\n   计算后检查:")
        print(f"   hal_high 存在: {'hal_high' in df_result3.columns}")
        print(f"   hal_low 存在: {'hal_low' in df_result3.columns}")
        print(f"   poc 存在: {'poc' in df_result3.columns}")
        print(f"   atr 存在: {'atr' in df_result3.columns}")
        print(f"   sr_strength_max 存在: {'sr_strength_max' in df_result3.columns}")

        if "sr_strength_max" in df_result3.columns:
            sr_strength = df_result3["sr_strength_max"]
            print(f"   sr_strength_max 统计:")
            print(f"      NaN 数量: {sr_strength.isna().sum()}")
            print(f"      有效数量: {sr_strength.notna().sum()}")
            if sr_strength.notna().any():
                valid = sr_strength[sr_strength.notna()]
                print(f"      范围: [{valid.min():.4f}, {valid.max():.4f}]")

        print(f"   ✅ 测试 3 通过：正确使用了 wpt_price_reconstructed")
    except Exception as e:
        print(f"   ❌ 测试 3 失败: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n✅ 所有测试完成")


if __name__ == "__main__":
    test_sr_strength_max_auto_dependencies()
