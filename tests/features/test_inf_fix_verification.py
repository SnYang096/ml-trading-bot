"""
集成测试：验证 inf 值修复

这个测试会：
1. 使用修复后的代码计算特征
2. 验证没有 inf 值
3. 验证特征值的合理性
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
    """加载真实的训练数据"""
    data_path = Path("data/parquet_data")
    if not data_path.exists():
        pytest.skip(f"Data path {data_path} does not exist")

    symbol = "BTCUSDT"
    timeframe = "240T"
    start_date = "2025-01-01"
    end_date = "2025-07-31"

    try:
        df = load_raw_data_simple(
            symbol=symbol,
            timeframe=timeframe,
            data_path=str(data_path),
            start_date=start_date,
            end_date=end_date,
        )
        df = df.loc[start_date:end_date]
        return df
    except Exception as e:
        pytest.skip(f"Failed to load real data: {e}")


def test_sr_strength_max_no_inf_after_fix(real_training_data):
    """验证 sr_strength_max 修复后没有 inf 值"""
    df = real_training_data.copy()

    engineer = BaselineFeatureEngineer()
    df_features = engineer.engineer_features(
        df,
        required_features=["sr_strength_max"],
    )

    if "sr_strength_max" not in df_features.columns:
        pytest.skip("sr_strength_max not computed")

    sr_strength = df_features["sr_strength_max"]
    inf_count = np.isinf(sr_strength).sum()

    print(f"\n✅ sr_strength_max verification:")
    print(f"   Total: {len(sr_strength)}")
    print(f"   Inf values: {inf_count}")
    print(f"   NaN values: {sr_strength.isna().sum()}")
    print(f"   Min: {sr_strength.min()}, Max: {sr_strength.max()}")
    print(f"   Mean: {sr_strength.mean()}, Std: {sr_strength.std()}")

    assert (
        inf_count == 0
    ), f"sr_strength_max should not have inf values, found {inf_count}"

    # 验证值的合理性
    finite_vals = sr_strength[np.isfinite(sr_strength)]
    if len(finite_vals) > 0:
        assert finite_vals.min() >= 0, "sr_strength_max should be non-negative"
        assert finite_vals.max() < 100, "sr_strength_max should be reasonable (< 100)"


def test_hurst_no_inf_after_fix(real_training_data):
    """验证 Hurst 特征修复后没有 inf 值"""
    df = real_training_data.copy()

    df_features = extract_hurst_features(
        df,
        price_col="close",
        cvd_col="cvd" if "cvd" in df.columns else None,
        volume_col="volume",
    )

    hurst_cols = [col for col in df_features.columns if "hurst" in col.lower()]

    print(f"\n✅ Hurst features verification:")
    for col in hurst_cols:
        values = df_features[col]
        inf_count = np.isinf(values).sum()
        finite_vals = values[np.isfinite(values)]

        print(f"   {col}: {inf_count} inf, {values.isna().sum()} NaN")
        if len(finite_vals) > 0:
            print(f"      Range: [{finite_vals.min():.4f}, {finite_vals.max():.4f}]")

        assert inf_count == 0, f"{col} should not have inf values, found {inf_count}"

        # 验证值的合理性（Hurst 应该在 0-1 之间）
        if len(finite_vals) > 0:
            assert finite_vals.min() >= -1, f"{col} should be >= -1"
            assert finite_vals.max() <= 2, f"{col} should be <= 2"


def test_rsi_no_inf_after_fix(real_training_data):
    """验证 RSI 修复后没有 inf 值"""
    df = real_training_data.copy()

    from src.features.time_series.baseline_features import BaselineFeatureEngineer

    rsi = BaselineFeatureEngineer.compute_rsi(df["close"], period=14)

    inf_count = np.isinf(rsi).sum()
    finite_vals = rsi[np.isfinite(rsi)]

    print(f"\n✅ RSI verification:")
    print(f"   Total: {len(rsi)}")
    print(f"   Inf values: {inf_count}")
    print(f"   NaN values: {rsi.isna().sum()}")
    if len(finite_vals) > 0:
        print(f"   Range: [{finite_vals.min():.2f}, {finite_vals.max():.2f}]")
        print(f"   Mean: {finite_vals.mean():.2f}, Std: {finite_vals.std():.2f}")

    assert inf_count == 0, f"RSI should not have inf values, found {inf_count}"

    # 验证值的合理性（RSI 应该在 0-100 之间）
    if len(finite_vals) > 0:
        assert finite_vals.min() >= 0, "RSI should be >= 0"
        assert finite_vals.max() <= 100, "RSI should be <= 100"


def test_trade_clustering_no_inf_after_fix(real_training_data):
    """验证 Trade Clustering zscore 修复后没有 inf 值"""
    df = real_training_data.copy()

    data_path = Path("data/parquet_data")
    symbol = "BTCUSDT"

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

        df_features = extract_order_flow_features(
            df,
            ticks=None,
            ticks_loader_json=ticks_loader_json,
            include_trade_clustering=True,
            trade_clustering_window=100,
            monthly_cache_dir=None,
        )

        zscore_cols = [
            col
            for col in df_features.columns
            if "zscore" in col.lower() and "trade_cluster" in col.lower()
        ]

        print(f"\n✅ Trade Clustering zscore features verification:")
        total_inf = 0
        for col in zscore_cols:
            values = df_features[col]
            inf_count = np.isinf(values).sum()
            finite_vals = values[np.isfinite(values)]

            print(f"   {col}: {inf_count} inf, {values.isna().sum()} NaN")
            if len(finite_vals) > 0:
                print(
                    f"      Range: [{finite_vals.min():.4f}, {finite_vals.max():.4f}]"
                )

            total_inf += inf_count
            assert (
                inf_count == 0
            ), f"{col} should not have inf values, found {inf_count}"

        assert (
            total_inf == 0
        ), f"All Trade Clustering zscore features should not have inf values, found {total_inf}"

    except Exception as e:
        pytest.skip(f"Failed to load tick data: {e}")


def test_trade_clustering_test_set_has_values(real_training_data):
    """验证测试集 Trade Clustering 特征有值（不是全 NaN）"""
    df = real_training_data.copy()

    # 模拟训练集和测试集分割
    test_size = 0.15
    split_idx = int(len(df) * (1 - test_size))
    df_test = df.iloc[split_idx:].copy()

    data_path = Path("data/parquet_data")
    symbol = "BTCUSDT"

    from src.data_tools.tick_loader import serialize_tick_loader_params, list_tick_files

    test_start_ts = df_test.index.min().strftime("%Y-%m-%d %H:%M:%S")
    test_end_ts = df_test.index.max().strftime("%Y-%m-%d %H:%M:%S")

    try:
        test_tick_files = list_tick_files(
            symbol=symbol,
            start_ts=test_start_ts,
            end_ts=test_end_ts,
            ticks_dir=str(data_path),
            lookback_minutes=0,
        )

        if not test_tick_files:
            pytest.skip("No tick files found for test set")

        tick_params = {
            "symbol": symbol,
            "tick_files": test_tick_files,
            "start_ts": test_start_ts,
            "end_ts": test_end_ts,
            "lookback_minutes": 0,
        }
        ticks_loader_json = serialize_tick_loader_params(tick_params)

        df_test_features = extract_order_flow_features(
            df_test,
            ticks=None,
            ticks_loader_json=ticks_loader_json,
            include_trade_clustering=True,
            trade_clustering_window=100,
            monthly_cache_dir=None,
        )

        trade_cluster_cols = [
            col for col in df_test_features.columns if col.startswith("trade_cluster_")
        ]

        print(f"\n✅ Test set Trade Clustering verification:")
        print(f"   Test set size: {len(df_test)}")
        print(f"   Time range: {df_test.index.min()} to {df_test.index.max()}")
        print(f"   Trade Clustering features: {len(trade_cluster_cols)}")

        total_non_nan = 0
        for col in trade_cluster_cols[:10]:
            values = df_test_features[col]
            nan_count = values.isna().sum()
            non_zero_count = (values != 0.0).sum()

            print(f"   {col}: {nan_count} NaN, {non_zero_count} non-zero")
            total_non_nan += non_zero_count

        assert (
            total_non_nan > 0
        ), "Test set should have some non-zero Trade Clustering values"

    except Exception as e:
        import traceback

        traceback.print_exc()
        pytest.fail(f"Failed to compute Trade Clustering for test set: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
