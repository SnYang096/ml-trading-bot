#!/usr/bin/env python3
"""
集成测试：复现 RSI 特征中的 inf 值问题

从训练日志看到：
- 训练集中 rsi 有 70 个 inf
- 出现在 2025-02-01 附近的时间点

这个测试会：
1. 加载实际数据
2. 计算 RSI 特征
3. 检查 inf 值的来源和原因
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.features.time_series.baseline_features import BaselineFeatureEngineer
from src.data_tools.data_handler import MarketDataLoader


def test_rsi_with_real_data():
    """使用真实数据测试 RSI 计算"""
    print("=" * 80)
    print("RSI Inf 值问题诊断测试")
    print("=" * 80)

    # 加载数据（使用与训练相同的数据路径）
    data_path = project_root / "data" / "parquet_data"
    symbol = "BTCUSDT"
    timeframe = "240T"
    start_date = "2025-01-01"
    end_date = "2025-07-31"

    print(f"\n📂 加载数据...")
    print(f"   Data path: {data_path}")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Date range: {start_date} to {end_date}")

    try:
        loader = MarketDataLoader(data_path=str(data_path))
        df_raw = loader.load_data(
            symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
        )

        if df_raw.empty:
            print("   ⚠️  数据为空，跳过测试")
            return

        print(f"   ✅ 加载了 {len(df_raw)} 条数据")
        print(f"   📅 时间范围: {df_raw.index.min()} 至 {df_raw.index.max()}")

        df = df_raw.copy()
        print(f"   ✅ 使用数据: {len(df)} 条")

        # 获取价格序列
        price_col = "close"
        if price_col not in df.columns:
            print(f"   ❌ 未找到 {price_col} 列")
            return

        prices = df[price_col]
        print(f"\n📊 价格数据统计:")
        print(f"   非空数量: {prices.notna().sum()}")
        print(f"   Inf 数量: {np.isinf(prices).sum()}")
        print(f"   NaN 数量: {prices.isna().sum()}")
        print(f"   最小值: {prices.min()}")
        print(f"   最大值: {prices.max()}")
        print(f"   均值: {prices.mean():.2f}")

        # 检查价格数据中的问题
        inf_mask = np.isinf(prices)
        if inf_mask.any():
            print(f"\n   ⚠️  发现价格数据中的 inf 值:")
            inf_indices = prices[inf_mask].index[:10]
            for idx in inf_indices:
                print(f"      {idx}: {prices.loc[idx]}")

        # 计算 RSI
        print(f"\n🔧 计算 RSI...")
        rsi = BaselineFeatureEngineer.compute_rsi(prices, period=14)

        # 测试 RSI 滚动 Z-score（可能是 inf 的来源）
        print(f"\n🔧 计算 RSI 滚动 Z-score...")
        rolling_zscore_windows = [50, 288, 500]
        for window in rolling_zscore_windows:
            zscore_col = f"rsi_zscore_w{window}"
            rsi_zscore = BaselineFeatureEngineer._rolling_zscore(
                rsi, window=window, min_periods=window
            )
            inf_count = np.isinf(rsi_zscore).sum()
            if inf_count > 0:
                print(f"   ⚠️  {zscore_col}: {inf_count} 个 inf 值")
                inf_indices = rsi_zscore[np.isinf(rsi_zscore)].index[:10]
                for idx in inf_indices:
                    print(
                        f"      {idx}: RSI={rsi.loc[idx]}, Z-score={rsi_zscore.loc[idx]}"
                    )
                    # 检查窗口内的数据
                    idx_pos = rsi.index.get_loc(idx)
                    window_start = max(0, idx_pos - window)
                    window_end = idx_pos + 1
                    window_rsi = rsi.iloc[window_start:window_end]
                    print(
                        f"         窗口 RSI: mean={window_rsi.mean():.2f}, std={window_rsi.std():.2f}, count={window_rsi.notna().sum()}"
                    )
            else:
                print(f"   ✅ {zscore_col}: 无 inf 值")

        print(f"\n📊 RSI 结果统计:")
        print(f"   总数量: {len(rsi)}")
        print(f"   非空数量: {rsi.notna().sum()}")
        print(f"   Inf 数量: {np.isinf(rsi).sum()}")
        print(f"   -Inf 数量: {(rsi == -np.inf).sum()}")
        print(f"   +Inf 数量: {(rsi == np.inf).sum()}")
        print(f"   NaN 数量: {rsi.isna().sum()}")

        if rsi.notna().any():
            rsi_valid = rsi[rsi.notna()]
            print(f"   有效值范围: [{rsi_valid.min():.2f}, {rsi_valid.max():.2f}]")
            print(f"   有效值均值: {rsi_valid.mean():.2f}")

        # 检查 inf 值的位置
        inf_mask = np.isinf(rsi)
        if inf_mask.any():
            print(f"\n   ⚠️  发现 RSI 中的 inf 值:")
            inf_indices = rsi[inf_mask].index[:20]
            for idx in inf_indices:
                rsi_val = rsi.loc[idx]
                price_val = prices.loc[idx]
                # 获取前后窗口的数据
                idx_pos = prices.index.get_loc(idx)
                window_start = max(0, idx_pos - 20)
                window_end = min(len(prices), idx_pos + 5)
                window_prices = prices.iloc[window_start:window_end]

                print(f"\n      📍 {idx}:")
                print(f"         RSI: {rsi_val}")
                print(f"         当前价格: {price_val}")
                print(f"         窗口价格统计:")
                print(f"           非空: {window_prices.notna().sum()}")
                print(f"           Inf: {np.isinf(window_prices).sum()}")
                print(
                    f"           范围: [{window_prices.min():.2f}, {window_prices.max():.2f}]"
                )
                print(f"           前15个值: {window_prices.head(15).tolist()}")

        # 检查特定日期（从日志看是 2025-02-01 附近）
        print(f"\n🔍 检查 2025-02-01 附近的数据...")
        if isinstance(rsi.index, pd.DatetimeIndex):
            target_date = "2025-02-01"
            date_mask = (rsi.index >= target_date) & (rsi.index < "2025-02-02")
            if date_mask.any():
                date_rsi = rsi[date_mask]
                date_prices = prices[date_mask]
                print(f"   📅 {target_date} 的数据:")
                print(f"      RSI 非空: {date_rsi.notna().sum()}")
                print(f"      RSI Inf: {np.isinf(date_rsi).sum()}")
                print(f"      价格统计:")
                print(f"         非空: {date_prices.notna().sum()}")
                print(f"         Inf: {np.isinf(date_prices).sum()}")
                print(
                    f"         范围: [{date_prices.min():.2f}, {date_prices.max():.2f}]"
                )

                # 显示具体值
                for idx in date_rsi.index[:10]:
                    print(f"      {idx}: RSI={rsi.loc[idx]}, Price={prices.loc[idx]}")

        # 测试 RSI 背离特征（可能产生 inf）
        print(f"\n🔬 测试 RSI 背离特征...")
        try:
            # 模拟 RSI 背离计算
            recent_high = prices.rolling(20, min_periods=5).max().ffill()
            recent_rsi_high = rsi.rolling(20, min_periods=5).max().ffill()
            tol = 1e-8
            top_divergence_mask = (
                recent_high.notna()
                & recent_rsi_high.notna()
                & (prices >= (recent_high - tol))
                & (rsi < (recent_rsi_high - tol))
            )

            recent_low = prices.rolling(20, min_periods=5).min().ffill()
            recent_rsi_low = rsi.rolling(20, min_periods=5).min().ffill()
            bottom_divergence_mask = (
                recent_low.notna()
                & recent_rsi_low.notna()
                & (prices <= (recent_low + tol))
                & (rsi > (recent_rsi_low + tol))
            )

            rsi_divergence = bottom_divergence_mask.astype(
                float
            ) - top_divergence_mask.astype(float)

            print(f"   RSI 背离特征统计:")
            print(f"      Inf 数量: {np.isinf(rsi_divergence).sum()}")
            print(f"      NaN 数量: {rsi_divergence.isna().sum()}")

            if np.isinf(rsi_divergence).any():
                print(f"   ⚠️  发现 RSI 背离特征中的 inf 值！")
                inf_indices = rsi_divergence[np.isinf(rsi_divergence)].index[:10]
                for idx in inf_indices:
                    print(
                        f"      {idx}: divergence={rsi_divergence.loc[idx]}, RSI={rsi.loc[idx]}"
                    )
        except Exception as e:
            print(f"   ⚠️  计算 RSI 背离特征时出错: {e}")

        # 尝试手动计算 RSI 来诊断问题
        print(f"\n🔬 手动计算 RSI 诊断...")
        try:
            import talib

            prices_clean = prices.replace([np.inf, -np.inf], np.nan)
            rsi_manual = talib.RSI(prices_clean.values, timeperiod=14)
            rsi_manual_series = pd.Series(rsi_manual, index=prices.index)

            print(f"   Talib 直接计算结果:")
            print(f"      Inf 数量: {np.isinf(rsi_manual_series).sum()}")
            print(f"      NaN 数量: {rsi_manual_series.isna().sum()}")

            # 检查是否有差异
            if not np.array_equal(
                np.isinf(rsi), np.isinf(rsi_manual_series), equal_nan=True
            ):
                print(f"   ⚠️  发现差异！")
                diff_mask = np.isinf(rsi) != np.isinf(rsi_manual_series)
                if diff_mask.any():
                    print(f"      差异位置: {rsi[diff_mask].index.tolist()[:10]}")
        except ImportError:
            print(f"   ⚠️  Talib 未安装，跳过手动计算")

        # 最后，使用完整的特征计算流程来复现问题
        print(f"\n🔬 使用完整特征计算流程...")
        try:
            # 准备数据
            df_test = df[["open", "high", "low", "close", "volume"]].copy()

            # 使用完整的特征计算流程
            engineer = BaselineFeatureEngineer()
            result = engineer.compute_features(
                df_test,
                required_features=["rsi"],
            )

            if "rsi" in result.columns:
                rsi_full = result["rsi"]
                inf_count = np.isinf(rsi_full).sum()
                print(f"   完整流程 RSI 统计:")
                print(f"      Inf 数量: {inf_count}")
                print(f"      NaN 数量: {rsi_full.isna().sum()}")

                if inf_count > 0:
                    print(f"   ⚠️  完整流程中发现了 {inf_count} 个 inf 值！")
                    inf_indices = rsi_full[np.isinf(rsi_full)].index[:10]
                    for idx in inf_indices:
                        print(
                            f"      {idx}: RSI={rsi_full.loc[idx]}, Price={prices.loc[idx]}"
                        )
        except Exception as e:
            print(f"   ⚠️  完整流程测试失败: {e}")
            import traceback

            traceback.print_exc()

        print(f"\n✅ 测试完成")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_rsi_with_real_data()
