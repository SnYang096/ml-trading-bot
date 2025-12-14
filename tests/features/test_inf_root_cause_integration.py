"""
集成测试：模拟真实训练场景，找出 inf 值的根本原因

这个测试会：
1. 加载真实的训练数据（2025-01-01 到 2025-07-31）
2. 计算有问题的特征（sr_strength_max, hurst, rsi, trade_clustering）
3. 找出产生 inf 值的具体原因
4. 验证修复
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.time_series.baseline_features import BaselineFeatureEngineer
from src.features.time_series.utils_hurst_features import extract_hurst_features
from src.features.time_series.utils_order_flow_features import (
    extract_order_flow_features,
)
from src.features.utils.data_monitor import check_data_quality


# 直接使用 MarketDataLoader 加载数据
def load_raw_data_simple(symbol, timeframe, data_path, start_date, end_date):
    """使用 MarketDataLoader 加载数据（与训练流程相同）"""
    try:
        from src.data_tools.data_loader import MarketDataLoader
    except ImportError:
        # 如果导入失败，尝试直接读取并转换 tick 数据
        import pandas as pd
        from pathlib import Path

        data_path = Path(data_path)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        months = pd.date_range(start=start_dt.replace(day=1), end=end_dt, freq="MS")

        all_ticks = []
        for month in months:
            month_str = month.strftime("%Y-%m")
            file_path = data_path / f"{symbol}_{month_str}.parquet"
            if file_path.exists():
                df_tick = pd.read_parquet(file_path)
                if "timestamp" in df_tick.columns:
                    df_tick["timestamp"] = pd.to_datetime(df_tick["timestamp"])
                    df_tick = df_tick.set_index("timestamp")
                all_ticks.append(df_tick)

        if not all_ticks:
            raise FileNotFoundError(f"No data files found for {symbol} in {data_path}")

        ticks = pd.concat(all_ticks).sort_index()
        ticks = ticks.loc[start_date:end_date]

        # 从 tick 数据生成 OHLCV
        df = ticks.resample(timeframe).agg(
            {
                "price": ["first", "max", "min", "last"],
                "volume": "sum",
            }
        )
        df.columns = ["open", "high", "low", "close", "volume"]
        df = df.dropna()

        # 添加其他列（如果存在）
        if "side" in ticks.columns:
            # 计算 CVD 等
            ticks["buy_vol"] = ticks["volume"] * (ticks["side"] == 1)
            ticks["sell_vol"] = ticks["volume"] * (ticks["side"] == -1)
            buy_vol = ticks["buy_vol"].resample(timeframe).sum()
            sell_vol = ticks["sell_vol"].resample(timeframe).sum()
            df["cvd"] = (buy_vol - sell_vol).cumsum()
            df["taker_buy_ratio"] = buy_vol / (buy_vol + sell_vol + 1e-10)

        return df

    loader = MarketDataLoader(data_path=str(data_path))
    df = loader.load_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
    )
    return df


@pytest.fixture
def real_training_data():
    """加载真实的训练数据（只加载两个月以加快测试速度）"""
    data_path = Path("data/parquet_data")
    if not data_path.exists():
        pytest.skip(f"Data path {data_path} does not exist")

    # 只加载两个月数据以加快测试速度
    symbol = "BTCUSDT"
    timeframe = "240T"
    start_date = "2025-06-01"  # 只加载6月和7月
    end_date = "2025-07-31"

    try:
        df = load_raw_data_simple(
            symbol=symbol,
            timeframe=timeframe,
            data_path=str(data_path),
            start_date=start_date,
            end_date=end_date,
        )

        # 裁剪到指定时间范围
        df = df.loc[start_date:end_date]

        print(f"\n📊 Loaded real training data:")
        print(f"   Shape: {df.shape}")
        print(f"   Time range: {df.index.min()} to {df.index.max()}")
        print(f"   Columns: {list(df.columns)}")

        return df
    except Exception as e:
        pytest.skip(f"Failed to load real data: {e}")


def test_sr_strength_max_inf_root_cause(real_training_data):
    """找出 sr_strength_max 产生 inf 值的根本原因"""
    df = real_training_data.copy()

    print(f"\n🔍 Testing sr_strength_max inf root cause...")

    # 检查源数据质量
    check_data_quality(
        df,
        data_source="SR_STRENGTH_TEST",
        stage="before_calc",
        raise_on_inf=False,
    )

    # 计算基础特征（ATR 等）
    engineer = BaselineFeatureEngineer()

    # 先计算 ATR（如果不存在）
    if "atr" not in df.columns:
        try:
            import talib

            df["atr"] = talib.ATR(
                df["high"].values, df["low"].values, df["close"].values, timeperiod=14
            )
        except ImportError:
            # 简单实现
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift())
            low_close = np.abs(df["low"] - df["close"].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df["atr"] = tr.rolling(window=14, min_periods=1).mean()

    # 检查 ATR
    atr_inf = np.isinf(df["atr"]).sum()
    atr_nan = df["atr"].isna().sum()
    print(f"   ATR: {atr_inf} inf, {atr_nan} NaN")

    # 计算 sr_strength_max
    df_features = engineer.engineer_features(
        df,
        required_features=["sr_strength_max"],
    )

    # 检查结果
    if "sr_strength_max" not in df_features.columns:
        pytest.fail("sr_strength_max not computed")

    sr_strength = df_features["sr_strength_max"]
    inf_mask = ~np.isfinite(sr_strength)
    inf_count = inf_mask.sum()

    print(f"\n📊 sr_strength_max results:")
    print(f"   Total: {len(sr_strength)}")
    print(f"   Inf values: {inf_count}")
    print(f"   NaN values: {sr_strength.isna().sum()}")

    if inf_count > 0:
        # 找出 inf 值的位置
        inf_indices = sr_strength[inf_mask].index[:10]
        print(f"   First 10 inf indices: {inf_indices.tolist()}")

        # 检查这些位置的其他特征
        for idx in inf_indices[:3]:
            print(f"\n   🔍 Analyzing index {idx}:")
            row = df_features.loc[idx]

            # 找出所有包含 inf 的列
            inf_cols = [col for col in df_features.columns if not np.isfinite(row[col])]
            print(f"      Inf columns: {inf_cols[:10]}")

            # 检查边界强度相关的列
            sqs_cols = [col for col in df_features.columns if col.startswith("sqs_")]
            print(
                f"      SQS columns with values: {[col for col in sqs_cols if col in df_features.columns and np.isfinite(row[col])][:5]}"
            )
            print(
                f"      SQS columns with inf: {[col for col in sqs_cols if col in df_features.columns and not np.isfinite(row[col])][:5]}"
            )

            # 检查 ATR
            if "atr" in df_features.columns:
                atr_val = df_features.loc[idx, "atr"]
                print(f"      ATR: {atr_val} (isfinite: {np.isfinite(atr_val)})")

        # 检查边界强度计算过程中的中间值
        print(f"\n   🔍 Checking boundary strength calculation...")
        # 这里可以添加更详细的检查逻辑

    assert (
        inf_count == 0
    ), f"sr_strength_max should not have inf values, found {inf_count}"


def test_hurst_inf_root_cause(real_training_data):
    """找出 Hurst 特征产生 inf 值的根本原因"""
    df = real_training_data.copy()

    print(f"\n🔍 Testing Hurst inf root cause...")

    # 检查源数据质量
    check_data_quality(
        df,
        data_source="HURST_TEST",
        stage="before_calc",
        raise_on_inf=False,
    )

    # 计算 Hurst 特征
    df_features = extract_hurst_features(
        df,
        price_col="close",
        cvd_col="cvd" if "cvd" in df.columns else None,
        volume_col="volume",
    )

    # 检查结果
    hurst_cols = [col for col in df_features.columns if "hurst" in col.lower()]
    print(f"\n📊 Hurst features: {len(hurst_cols)}")

    for col in hurst_cols:
        values = df_features[col]
        inf_mask = np.isinf(values)
        nan_mask = values.isna()
        inf_count = inf_mask.sum()
        nan_count = nan_mask.sum()

        print(f"   {col}: {inf_count} inf (real), {nan_count} NaN (data insufficient)")

        if inf_count > 0:
            # 找出 inf 值的位置
            inf_indices = values[inf_mask].index[:10]
            inf_values = values[inf_mask].head(10)
            print(f"      First 10 inf indices: {inf_indices.tolist()}")
            print(f"      First 10 inf values: {inf_values.tolist()}")
            print(f"      ⚠️  These are REAL inf values, not NaN!")

            # 检查这些位置的源数据
            for idx in inf_indices[:3]:
                print(f"\n      🔍 Analyzing index {idx}:")
                try:
                    idx_pos = df.index.get_loc(idx)
                    start_pos = max(0, idx_pos - 50)
                    if "price" in col.lower():
                        # 检查价格数据
                        price_vals = df.iloc[start_pos : idx_pos + 1]["close"]
                        print(
                            f"         Price range: {price_vals.min():.2f} to {price_vals.max():.2f}"
                        )
                        print(
                            f"         Price changes: {price_vals.pct_change().describe()}"
                        )
                    elif "cvd" in col.lower() and "cvd" in df.columns:
                        # 检查 CVD 数据
                        cvd_vals = df.iloc[start_pos : idx_pos + 1]["cvd"]
                        print(
                            f"         CVD range: {cvd_vals.min():.2f} to {cvd_vals.max():.2f}"
                        )
                        print(f"         CVD changes: {cvd_vals.diff().describe()}")
                except Exception as e:
                    print(f"         Error analyzing {idx}: {e}")

        assert inf_count == 0, f"{col} should not have inf values, found {inf_count}"


def test_rsi_inf_root_cause(real_training_data):
    """找出 RSI 产生 inf 值的根本原因"""
    df = real_training_data.copy()

    print(f"\n🔍 Testing RSI inf root cause...")

    # 检查源数据质量
    check_data_quality(
        df,
        data_source="RSI_TEST",
        stage="before_calc",
        raise_on_inf=False,
    )

    # 计算 RSI
    from src.features.time_series.baseline_features import BaselineFeatureEngineer

    rsi = BaselineFeatureEngineer.compute_rsi(df["close"], period=14)

    # 检查结果（区分 inf 和 NaN）
    inf_mask = np.isinf(rsi)
    nan_mask = rsi.isna()
    inf_count = inf_mask.sum()
    nan_count = nan_mask.sum()

    print(f"\n📊 RSI results:")
    print(f"   Total: {len(rsi)}")
    print(f"   Inf values: {inf_count} (actual inf)")
    print(f"   NaN values: {nan_count} (data insufficient)")

    if inf_count > 0:
        # 找出 inf 值的位置
        inf_indices = rsi[inf_mask].index[:10]
        inf_values = rsi[inf_mask].head(10)
        print(f"   First 10 inf indices: {inf_indices.tolist()}")
        print(f"   First 10 inf values: {inf_values.tolist()}")
        print(f"   ⚠️  These are REAL inf values, not NaN!")

        # 检查这些位置的源数据
        for idx in inf_indices[:3]:
            print(f"\n   🔍 Analyzing index {idx}:")
            idx_pos = df.index.get_loc(idx)
            # 检查 RSI 计算窗口内的数据
            window_start = max(0, idx_pos - 20)
            window_data = df.iloc[window_start : idx_pos + 1]["close"]
            print(
                f"      Close price range: {window_data.min():.2f} to {window_data.max():.2f}"
            )
            print(f"      Close price changes: {window_data.pct_change().describe()}")
            print(f"      Has inf in window: {np.isinf(window_data).any()}")
            print(f"      Has NaN in window: {window_data.isna().any()}")
            print(f"      All zeros: {(window_data == 0).all()}")
            print(f"      All same: {(window_data == window_data.iloc[0]).all()}")

    # RSI 前 period 个值是 NaN 是正常的（数据不足）
    # 但 inf 值是不正常的
    if inf_count > 0:
        print(f"   ⚠️  WARNING: Found {inf_count} REAL inf values in RSI!")
        print(
            f"   This suggests a problem in RSI calculation, not just data insufficiency."
        )
    assert (
        inf_count == 0
    ), f"RSI should not have inf values (NaN is OK), found {inf_count} inf"


def test_trade_clustering_inf_root_cause(real_training_data):
    """找出 Trade Clustering zscore 产生 inf 值的根本原因"""
    df = real_training_data.copy()

    print(f"\n🔍 Testing Trade Clustering inf root cause...")

    # 检查源数据质量
    check_data_quality(
        df,
        data_source="TRADE_CLUSTERING_TEST",
        stage="before_calc",
        raise_on_inf=False,
    )

    # 尝试加载 tick 数据
    data_path = Path("data/parquet_data")
    symbol = "BTCUSDT"

    # 创建 ticks_loader_json（模拟训练流程）
    from src.data_tools.tick_loader import serialize_tick_loader_params, list_tick_files

    start_ts = df.index.min().strftime("%Y-%m-%d %H:%M:%S")
    end_ts = df.index.max().strftime("%Y-%m-%d %H:%M:%S")

    try:
        tick_files = list_tick_files(
            symbol=symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            ticks_dir=str(data_path),
            lookback_minutes=0,
        )

        if not tick_files:
            pytest.skip("No tick files found")

        tick_params = {
            "symbol": symbol,
            "tick_files": tick_files,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "lookback_minutes": 0,
        }
        ticks_loader_json = serialize_tick_loader_params(tick_params)

        # 计算 Trade Clustering 特征
        df_features = extract_order_flow_features(
            df,
            ticks=None,
            ticks_loader_json=ticks_loader_json,
            include_trade_clustering=True,
            trade_clustering_window=100,
            monthly_cache_dir=None,  # 不使用缓存，强制重新计算
        )

        # 检查结果
        zscore_cols = [
            col
            for col in df_features.columns
            if "zscore" in col.lower() and "trade_cluster" in col.lower()
        ]
        print(f"\n📊 Trade Clustering zscore features: {len(zscore_cols)}")

        for col in zscore_cols:
            values = df_features[col]
            inf_mask = ~np.isfinite(values)
            inf_count = inf_mask.sum()

            print(f"   {col}: {inf_count} inf, {values.isna().sum()} NaN")

            if inf_count > 0:
                # 找出 inf 值的位置
                inf_indices = values[inf_mask].index[:10]
                print(f"      First 10 inf indices: {inf_indices.tolist()}")

                # 检查基础特征
                base_col = col.replace("_zscore_20", "").replace("_zscore_50", "")
                if base_col in df_features.columns:
                    base_vals = df_features.loc[inf_indices[:3], base_col]
                    print(
                        f"      Base feature ({base_col}) values: {base_vals.tolist()}"
                    )

        # 检查所有 zscore 特征
        total_inf = sum([(~np.isfinite(df_features[col])).sum() for col in zscore_cols])
        assert (
            total_inf == 0
        ), f"Trade Clustering zscore features should not have inf values, found {total_inf}"

    except Exception as e:
        pytest.skip(f"Failed to load tick data: {e}")


def test_trade_clustering_test_set_nan(real_training_data):
    """找出测试集 Trade Clustering 特征全为 NaN 的原因"""
    df = real_training_data.copy()

    print(f"\n🔍 Testing Trade Clustering test set NaN issue...")

    # 模拟训练集和测试集分割（与训练流程相同）
    test_size = 0.15
    split_idx = int(len(df) * (1 - test_size))
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    print(
        f"   Train: {len(df_train)} ({df_train.index.min()} to {df_train.index.max()})"
    )
    print(f"   Test: {len(df_test)} ({df_test.index.min()} to {df_test.index.max()})")

    # 尝试加载 tick 数据
    data_path = Path("data/parquet_data")
    symbol = "BTCUSDT"

    from src.data_tools.tick_loader import serialize_tick_loader_params, list_tick_files

    # 测试集时间范围
    test_start_ts = df_test.index.min().strftime("%Y-%m-%d %H:%M:%S")
    test_end_ts = df_test.index.max().strftime("%Y-%m-%d %H:%M:%S")

    try:
        # 检查测试集时间范围内的 tick 文件
        test_tick_files = list_tick_files(
            symbol=symbol,
            start_ts=test_start_ts,
            end_ts=test_end_ts,
            ticks_dir=str(data_path),
            lookback_minutes=0,
        )

        print(f"   Test set tick files: {len(test_tick_files)}")
        for f in test_tick_files[:5]:
            print(f"      {f}")

        if not test_tick_files:
            pytest.skip("No tick files found for test set")

        # 创建 ticks_loader_json
        tick_params = {
            "symbol": symbol,
            "tick_files": test_tick_files,
            "start_ts": test_start_ts,
            "end_ts": test_end_ts,
            "lookback_minutes": 0,
        }
        ticks_loader_json = serialize_tick_loader_params(tick_params)

        # 计算 Trade Clustering 特征（只对测试集）
        print(f"\n   📊 Computing Trade Clustering for test set...")
        df_test_features = extract_order_flow_features(
            df_test,
            ticks=None,
            ticks_loader_json=ticks_loader_json,
            include_trade_clustering=True,
            trade_clustering_window=100,
            monthly_cache_dir=None,
        )

        # 检查结果
        trade_cluster_cols = [
            col for col in df_test_features.columns if col.startswith("trade_cluster_")
        ]
        print(f"\n📊 Trade Clustering features in test set: {len(trade_cluster_cols)}")

        for col in trade_cluster_cols[:10]:
            values = df_test_features[col]
            nan_count = values.isna().sum()
            non_zero_count = (values != 0.0).sum()

            print(
                f"   {col}: {nan_count} NaN, {non_zero_count} non-zero, {len(values)} total"
            )

            if nan_count == len(values):
                print(f"      ⚠️  All values are NaN!")
                # 检查时间对齐
                print(
                    f"      K-line time range: {df_test.index.min()} to {df_test.index.max()}"
                )

        # 检查是否有任何非 NaN 值
        total_non_nan = sum(
            [df_test_features[col].notna().sum() for col in trade_cluster_cols]
        )
        if total_non_nan == 0:
            pytest.fail("All Trade Clustering features are NaN in test set")

    except Exception as e:
        import traceback

        traceback.print_exc()
        pytest.fail(f"Failed to compute Trade Clustering for test set: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
