#!/usr/bin/env python3
"""
修复FER方向逻辑

FER语义: 单边失败 → 反向清算
- CVD正值(多头aggressor活跃) + 价格不涨 = 多头失败 → 做空
- CVD负值(空头aggressor活跃) + 价格不跌 = 空头失败 → 做多

方向规则: direction = -sign(cvd_change_5_normalized)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def fix_fer_direction(predictions_path: str) -> pd.DataFrame:
    """
    为FER predictions添加基于CVD的动态方向

    Args:
        predictions_path: FER predictions.parquet路径

    Returns:
        修复后的DataFrame
    """
    print(f"📂 读取FER predictions: {predictions_path}")
    df = pd.read_parquet(predictions_path)

    print(f"📊 原始数据: {len(df)} rows, {len(df.columns)} columns")
    print(f"📋 列名: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")

    # 检查是否有CVD列
    cvd_candidates = [
        "cvd_change_5_normalized",
        "cvd_change_10_normalized",
        "cvd_change_20_normalized",
        "cvd_normalized",
    ]

    cvd_col = None
    for col in cvd_candidates:
        if col in df.columns:
            cvd_col = col
            print(f"✅ 找到CVD列: {col}")
            break

    if cvd_col is None:
        print(f"⚠️  未找到CVD列，候选列: {cvd_candidates}")
        print(f"⚠️  降级为固定做多（direction=1.0）")
        df["entry_direction"] = 1.0
        return df

    # 计算动态方向
    print(f"\n🔄 基于{cvd_col}计算动态方向...")

    # CVD正 → 做空(-1)，CVD负 → 做多(1)
    direction = -np.sign(df[cvd_col])

    # 处理CVD=0的情况（默认做多）
    direction = direction.replace(0, 1)

    # 统计方向分布
    long_count = (direction == 1).sum()
    short_count = (direction == -1).sum()

    print(f"📊 方向分布:")
    print(f"   做多: {long_count} ({long_count/len(df)*100:.1f}%)")
    print(f"   做空: {short_count} ({short_count/len(df)*100:.1f}%)")

    # 添加方向列
    df["entry_direction"] = direction

    # 统计CVD分布（验证逻辑）
    cvd_stats = df[cvd_col].describe()
    print(f"\n📈 {cvd_col}统计:")
    print(f"   均值: {cvd_stats['mean']:.4f}")
    print(f"   中位数: {cvd_stats['50%']:.4f}")
    print(f"   最小值: {cvd_stats['min']:.4f}")
    print(f"   最大值: {cvd_stats['max']:.4f}")

    return df


def main():
    # 查找最新的FER训练结果
    results_dir = Path("/home/yin/trading/ml_trading_bot/results")

    # 按时间排序，找最新的
    train_dirs = sorted(
        list(results_dir.glob("train_final_*_return_tree"))
        + list((results_dir / "fer").glob("train_final_*_return_tree")),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    fer_path = None
    for train_dir in train_dirs:
        fer_subdir = train_dir / "fer"
        if fer_subdir.exists():
            pred_file = fer_subdir / "predictions.parquet"
            if pred_file.exists():
                fer_path = pred_file
                print(f"🔍 找到FER predictions: {train_dir.name}")
                break

    if fer_path is None:
        print("❌ 未找到FER predictions文件")
        sys.exit(1)

    # 修复方向
    df_fixed = fix_fer_direction(str(fer_path))

    # 保存修复后的文件
    output_path = fer_path.parent / "predictions_fixed.parquet"
    print(f"\n💾 保存修复后的predictions: {output_path}")
    df_fixed.to_parquet(output_path, index=False)

    print(f"\n✅ FER方向修复完成！")
    print(f"   输入: {fer_path}")
    print(f"   输出: {output_path}")

    # 验证文件
    df_verify = pd.read_parquet(output_path)
    assert "entry_direction" in df_verify.columns, "entry_direction列未添加"
    assert df_verify["entry_direction"].isin([-1, 1]).all(), "direction值必须是-1或1"
    print(f"\n✅ 验证通过！")


if __name__ == "__main__":
    main()
