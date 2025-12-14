#!/usr/bin/env python3
"""
诊断早期数据产生 inf 值的原因

从训练输出看，inf 值都出现在 1-2 月的数据上：
- sr_strength_max: 706个inf，样本在 2025-01-03
- hurst_price_rolling: 298个inf，样本在 2025-01-03
- rsi: 70个inf，样本在 2025-02-01

这个脚本会：
1. 加载 1-2 月的数据
2. 逐步计算特征，找出产生 inf 的具体步骤
3. 检查早期数据的边界条件
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_tools.data_utils import load_raw_data
from src.features.time_series.baseline_features import BaselineFeatureEngineer
from src.features.time_series.utils_hurst_features import extract_hurst_features
from src.features.utils.data_monitor import check_data_quality


def diagnose_early_data():
    """诊断早期数据的 inf 问题"""
    print("=" * 80)
    print("🔍 诊断早期数据产生 inf 值的原因")
    print("=" * 80)

    # 加载 1-2 月的数据（问题时间段）
    data_path = "data/parquet_data"
    symbol = "BTCUSDT"
    timeframe = "240T"
    start_date = "2025-01-01"
    end_date = "2025-02-28"

    print(f"\n📊 加载数据: {start_date} 到 {end_date}")
    df = load_raw_data(
        data_path=data_path,
        symbol=symbol,
        timeframe=timeframe,
    )

    # 裁剪到指定时间范围
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.loc[start_date:end_date]
    else:
        datetime_col = next(
            (col for col in ("datetime", "timestamp", "date") if col in df.columns),
            None,
        )
        if datetime_col:
            dt_idx = pd.to_datetime(df[datetime_col])
            mask = (dt_idx >= pd.to_datetime(start_date)) & (
                dt_idx <= pd.to_datetime(end_date)
            )
            df = df.loc[mask]

    print(f"   Shape: {df.shape}")
    print(f"   Time range: {df.index.min()} to {df.index.max()}")

    # 检查源数据质量
    print("\n" + "-" * 80)
    print("1️⃣ 检查源数据质量")
    print("-" * 80)
    check_data_quality(
        df,
        data_source="EARLY_DATA_DIAGNOSIS",
        stage="raw_data",
        raise_on_inf=False,
    )

    # 计算 ATR
    print("\n" + "-" * 80)
    print("2️⃣ 计算 ATR")
    print("-" * 80)
    engineer = BaselineFeatureEngineer()
    if "atr" not in df.columns:
        df["atr"] = engineer._compute_atr(df, window=14)

    atr_inf = np.isinf(df["atr"]).sum()
    atr_nan = df["atr"].isna().sum()
    print(f"   ATR: {atr_inf} inf, {atr_nan} NaN")

    if atr_inf > 0:
        inf_indices = df["atr"][np.isinf(df["atr"])].index[:5]
        print(f"   ⚠️  ATR inf 位置: {inf_indices.tolist()}")

    if atr_nan > 0:
        nan_indices = df["atr"][df["atr"].isna()].index[:5]
        print(f"   ℹ️  ATR NaN 位置（前14个是正常的）: {nan_indices.tolist()}")

    # 检查 ATR 在早期是否全为 0 或 NaN
    early_atr = df["atr"].iloc[:20]
    print(f"\n   前20个 ATR 值:")
    print(f"   - 全为 0: {(early_atr == 0).all()}")
    print(f"   - 全为 NaN: {early_atr.isna().all()}")
    print(f"   - 有 inf: {np.isinf(early_atr).any()}")
    print(f"   - 前5个值: {early_atr.head().tolist()}")

    # 计算 sr_strength_max
    print("\n" + "-" * 80)
    print("3️⃣ 计算 sr_strength_max")
    print("-" * 80)

    # 先计算基础特征
    df_features = engineer.engineer_features(
        df,
        required_features=["sr_strength_max"],
    )

    if "sr_strength_max" in df_features.columns:
        sr_strength = df_features["sr_strength_max"]
        inf_mask = np.isinf(sr_strength)
        inf_count = inf_mask.sum()

        print(f"   sr_strength_max: {inf_count} inf, {sr_strength.isna().sum()} NaN")

        if inf_count > 0:
            inf_indices = sr_strength[inf_mask].index[:10]
            print(f"   ⚠️  inf 位置: {inf_indices.tolist()}")

            # 检查这些位置的 sqs_* 列
            sqs_cols = [col for col in df_features.columns if col.startswith("sqs_")]
            print(f"\n   检查 sqs_* 列（共 {len(sqs_cols)} 个）:")
            for idx in inf_indices[:3]:
                print(f"\n   📍 位置 {idx}:")
                row = df_features.loc[idx]
                inf_sqs = [
                    col
                    for col in sqs_cols
                    if col in df_features.columns and not np.isfinite(row[col])
                ]
                if inf_sqs:
                    print(f"      ⚠️  包含 inf 的 sqs 列: {inf_sqs[:5]}")
                    for col in inf_sqs[:3]:
                        val = row[col]
                        print(f"         {col}: {val} (type: {type(val).__name__})")
                else:
                    print(f"      ✅ 所有 sqs 列都是有限的")

                # 检查 ATR
                if "atr" in df_features.columns:
                    atr_val = df_features.loc[idx, "atr"]
                    print(f"      ATR: {atr_val} (isfinite: {np.isfinite(atr_val)})")

                # 检查价格数据
                if "close" in df_features.columns:
                    close_val = df_features.loc[idx, "close"]
                    print(
                        f"      Close: {close_val} (isfinite: {np.isfinite(close_val)})"
                    )

    # 计算 Hurst 特征
    print("\n" + "-" * 80)
    print("4️⃣ 计算 Hurst 特征")
    print("-" * 80)

    hurst_features = extract_hurst_features(
        df,
        price_col="close",
        cvd_col="cvd" if "cvd" in df.columns else None,
        volume_col="volume",
    )

    hurst_cols = [col for col in hurst_features.columns if "hurst" in col.lower()]
    for col in hurst_cols:
        values = hurst_features[col]
        inf_count = np.isinf(values).sum()
        nan_count = values.isna().sum()
        print(f"   {col}: {inf_count} inf, {nan_count} NaN")

        if inf_count > 0:
            inf_indices = values[np.isinf(values)].index[:5]
            print(f"      ⚠️  inf 位置: {inf_indices.tolist()}")

    # 计算 RSI
    print("\n" + "-" * 80)
    print("5️⃣ 计算 RSI")
    print("-" * 80)

    rsi = BaselineFeatureEngineer.compute_rsi(df["close"], period=14)
    inf_count = np.isinf(rsi).sum()
    nan_count = rsi.isna().sum()
    print(f"   RSI: {inf_count} inf, {nan_count} NaN")

    if inf_count > 0:
        inf_indices = rsi[np.isinf(rsi)].index[:10]
        print(f"   ⚠️  inf 位置: {inf_indices.tolist()}")

        # 检查这些位置的源数据
        for idx in inf_indices[:3]:
            print(f"\n   📍 位置 {idx}:")
            idx_pos = df.index.get_loc(idx)
            window_start = max(0, idx_pos - 20)
            window_data = df.iloc[window_start : idx_pos + 1]["close"]
            print(
                f"      Close 价格范围: {window_data.min():.2f} to {window_data.max():.2f}"
            )
            print(f"      全为 0: {(window_data == 0).all()}")
            print(f"      全相同: {(window_data == window_data.iloc[0]).all()}")
            print(f"      有 inf: {np.isinf(window_data).any()}")
            print(f"      有 NaN: {window_data.isna().any()}")

    print("\n" + "=" * 80)
    print("✅ 诊断完成")
    print("=" * 80)


if __name__ == "__main__":
    diagnose_early_data()
