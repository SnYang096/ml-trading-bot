import numpy as np
import pandas as pd
import pytest

import data_tools.data_loader as data_loader_module
from data_tools.baseline_feature_engineering import BaselineFeatureEngineer
from data_tools.comprehensive_feature_engineering import ComprehensiveFeatureEngineer
from data_tools.data_loader import MarketDataLoader
from data_tools.feature_engineering import FeatureEngineer
from data_tools.feature_engineering_talib import TalibFeatureEngineer
from time_series_model.pipeline.training.safe_multi_asset_preprocessing import (
    ensure_feature_consistency,
)


def _make_raw_ohlcv(
    periods: int = 120, freq: str = "15min", seed: int = 123
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=periods, freq=freq)
    close = np.linspace(100, 110, periods) + rng.normal(0, 0.3, size=periods)
    df = pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.1, size=periods),
            "high": close + rng.uniform(0.1, 0.3, size=periods),
            "low": close - rng.uniform(0.1, 0.3, size=periods),
            "close": close,
            "volume": rng.uniform(5_000, 10_000, size=periods),
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def test_baseline_feature_engineer_generates_expected_columns():
    raw = _make_raw_ohlcv()
    engineer = BaselineFeatureEngineer()
    features = engineer.engineer_features(raw, fit=True)

    assert len(features) == len(raw)
    assert {"close", "volume"}.issubset(features.columns)
    assert len(features.columns) > len(raw.columns)
    assert any("compression" in col for col in features.columns)
    assert any("sr_dist" in col for col in features.columns)
    assert any("atr" in col for col in features.columns)
    # All numeric features should be finite after engineering
    numeric = features.select_dtypes(include=[np.number])
    nan_ratio = numeric.isna().mean()
    assert nan_ratio.max() < 0.2


def test_multi_timeframe_feature_consistency():
    raw = _make_raw_ohlcv()
    engineer = BaselineFeatureEngineer()

    engineered_15m = engineer.engineer_features(raw, fit=True)
    engineered_1h = engineer.engineer_features(raw.resample("60min").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ), fit=True)

    common_cols = ensure_feature_consistency([engineered_15m, engineered_1h])
    assert len(common_cols) > 0
    # All returned column names should exist in both frames
    for frame in [engineered_15m, engineered_1h]:
        assert set(common_cols).issubset(frame.columns)


def test_comprehensive_feature_engineer_baseline_only():
    raw = _make_raw_ohlcv()
    cfe = ComprehensiveFeatureEngineer(feature_types="baseline")
    features = cfe.engineer_all_features(raw, fit=True)
    assert len(features) == len(raw)
    assert len(features.columns) > len(raw.columns)
    assert any("compression" in col for col in features.columns)
    assert any("sr_dist" in col for col in features.columns)


def test_market_data_loader_multi_timeframe(monkeypatch):
    # Patch TIMEFRAMES to a small deterministic set
    monkeypatch.setattr(data_loader_module, "TIMEFRAMES", ["15min", "60min"])
    loader = MarketDataLoader()

    # Create high-resolution raw data (1-minute) for resampling
    idx = pd.date_range("2024-01-01", periods=240, freq="1min")
    prices = 100 + np.cumsum(np.random.default_rng(321).normal(0, 0.05, len(idx)))
    loader.raw_data = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.1,
            "low": prices - 0.1,
            "close": prices,
            "volume": np.random.default_rng(321).integers(100, 500, len(idx)),
        },
        index=idx,
    )

    multi_tf = loader.get_multi_timeframe_data()
    assert set(multi_tf.keys()) == {"15min", "60min"}
    for tf, df in multi_tf.items():
        assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
        assert len(df) > 0
        # Index should be monotonically increasing without NaNs
        assert df.index.is_monotonic_increasing
        assert not df.isna().any().any()


def test_feature_engineer_multi_timeframe_with_mocked_talib(monkeypatch):
    def fake_add(self, data):
        df = data.copy()
        df["sma_stub"] = np.arange(len(df), dtype=float)
        return df

    monkeypatch.setattr(
        TalibFeatureEngineer, "add_technical_indicators", fake_add, raising=False
    )

    fe = FeatureEngineer()
    multi_data = {
        "15min": _make_raw_ohlcv(periods=30, freq="15min"),
        "60min": _make_raw_ohlcv(periods=30, freq="60min"),
    }

    engineered = fe.engineer_features(multi_data)
    assert set(engineered.keys()) == {"15min", "60min"}
    for df in engineered.values():
        assert "sma_stub" in df.columns
        assert np.array_equal(df["sma_stub"].values, np.arange(len(df), dtype=float))


def test_comprehensive_feature_engineer_default_uses_feature_engineer(monkeypatch):
    added_columns = []

    def fake_add(self, data):
        df = data.copy()
        df["fake_talib"] = 42.0
        added_columns.append(df.columns.tolist())
        return df

    monkeypatch.setattr(
        TalibFeatureEngineer, "add_technical_indicators", fake_add, raising=False
    )

    raw = _make_raw_ohlcv()
    cfe = ComprehensiveFeatureEngineer(feature_types="default")
    features = cfe.engineer_all_features(raw, fit=True)
    assert "fake_talib" in features.columns
    assert features["fake_talib"].eq(42.0).all()
    # Ensure FeatureEngineer was invoked at least once
    assert added_columns, "FeatureEngineer should call Talib engine under default mode"

