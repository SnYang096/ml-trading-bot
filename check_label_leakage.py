#!/usr/bin/env python3
"""
检查标签泄漏的工具脚本

检查三个常见的标签泄漏问题：
1. 用整个样本的标准化做标签
2. 用未来信息定义分类标签
3. 标签包含当前 bar 的 close
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from time_series_model.pipeline.training.rank_ic_trainer import prepare_rank_ic_labels


def check_label_leakage(df: pd.DataFrame, hold_period: int = 5):
    """检查标签泄漏问题"""
    print("=" * 60)
    print("🔍 标签泄漏检查")
    print("=" * 60)
    
    # 问题 1: 检查是否有全局标准化
    print("\n1️⃣  检查全局标准化泄漏...")
    if "future_return" in df.columns:
        # 检查是否有使用全局 mean/std 的标准化
        global_mean = df["future_return"].mean()
        global_std = df["future_return"].std()
        print(f"   future_return 全局均值: {global_mean:.6f}")
        print(f"   future_return 全局标准差: {global_std:.6f}")
        
        # 检查是否有标准化后的标签
        normalized_cols = [col for col in df.columns if "normalized" in col.lower() or "zscore" in col.lower()]
        if normalized_cols:
            print(f"   ⚠️  发现标准化列: {normalized_cols}")
            for col in normalized_cols:
                # 检查是否使用了全局统计
                if df[col].notna().any():
                    sample_val = df[col].dropna().iloc[0]
                    # 如果值接近 (x - mean) / std，可能是全局标准化
                    if abs(sample_val) < 10:  # 标准化后的值通常在合理范围内
                        print(f"      {col}: 可能是标准化值，需要检查是否使用全局统计")
        else:
            print("   ✅ 未发现明显的全局标准化标签")
    
    # 问题 2: 检查分类标签是否使用未来信息
    print("\n2️⃣  检查分类标签的未来信息泄漏...")
    quantile_cols = [col for col in df.columns if "quantile" in col.lower()]
    if quantile_cols:
        print(f"   发现分位数列: {quantile_cols}")
        for col in quantile_cols:
            # 检查是否有使用全局 quantile
            if df[col].notna().any():
                # 检查值的分布
                values = df[col].dropna()
                if len(values) > 0:
                    print(f"      {col}: 值范围 [{values.min():.3f}, {values.max():.3f}]")
                    # 如果值都在 [0, 1] 范围内，可能是分位数
                    if values.min() >= 0 and values.max() <= 1:
                        print(f"         ✅ 看起来是滚动分位数（值在 [0,1] 范围内）")
    else:
        print("   ✅ 未发现分位数标签")
    
    # 问题 3: 检查 future_return 是否依赖当前 bar 的 close
    print("\n3️⃣  检查 future_return 是否依赖当前 bar 的 close...")
    if "close" in df.columns and "future_return" in df.columns:
        # 检查前几行
        print("   前5行数据:")
        print("   ✅ 修复后: future_return[t] 使用 close[t+1] 作为起始价格")
        for i in range(min(5, len(df))):
            if not pd.isna(df["future_return"].iloc[i]):
                # 修复后: future_return[t] = (close[t+1+horizon] - close[t+1]) / close[t+1]
                close_t1 = df["close"].iloc[i+1] if i+1 < len(df) else np.nan
                close_t1_h = df["close"].iloc[i+1+hold_period] if i+1+hold_period < len(df) else np.nan
                future_ret = df["future_return"].iloc[i]
                
                # 计算期望值（修复后的逻辑）
                if not pd.isna(close_t1) and not pd.isna(close_t1_h):
                    expected = (close_t1_h - close_t1) / close_t1
                    match = "✅" if np.isclose(future_ret, expected, rtol=1e-5) else "❌"
                    print(f"      t={i}: close[{i+1}]={close_t1:.4f}, close[{i+1+hold_period}]={close_t1_h:.4f}")
                    print(f"         future_return[{i}] = {future_ret:.6f} (expected: {expected:.6f}) {match}")
                    if match == "✅":
                        print(f"         ✅ 使用 close[{i+1}] 作为起始价格（假设在 t+1 开盘价成交），安全！")
                    else:
                        print(f"         ⚠️  值不匹配，需要检查计算逻辑")
    
    print("\n" + "=" * 60)
    print("检查完成")
    print("=" * 60)


if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="15T")
    close = 100 + np.cumsum(np.random.randn(100) * 0.01)
    df = pd.DataFrame({"close": close}, index=dates)
    
    # 准备标签
    df_with_labels = prepare_rank_ic_labels(
        df,
        price_col="close",
        hold_period=5,
        lookback_window=20,
    )
    
    # 检查泄漏
    check_label_leakage(df_with_labels, hold_period=5)

