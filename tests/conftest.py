"""
Shared pytest configuration and fixtures.

This conftest.py automatically handles project root setup for all tests.
- Sets PROJECT_ROOT for path calculations
- Adds project root to sys.path (so imports work even if project is not installed)

Note: If you install the project via `pip install -e .`, sys.path setup is optional.
However, we still set it up here to ensure tests work in all environments.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to sys.path if not already there
# This ensures tests work even if project is not installed via pip
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_data():
    """创建样本数据用于测试"""
    np.random.seed(42)
    n_samples = 500
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="4h")

    # 生成价格数据
    price_base = 50000
    returns = np.random.randn(n_samples) * 0.01
    prices = price_base * (1 + returns).cumprod()

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_samples) * 0.001),
            "high": prices * (1 + np.abs(np.random.randn(n_samples)) * 0.002),
            "low": prices * (1 - np.abs(np.random.randn(n_samples)) * 0.002),
            "close": prices,
            "volume": np.random.uniform(1000, 10000, n_samples),
            "cvd": np.random.randn(n_samples).cumsum() * 1000,
            "taker_buy_ratio": np.random.uniform(0.3, 0.7, n_samples),
        },
        index=dates,
    )

    # 计算基础指标
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["rsi"] = 50 + np.random.randn(n_samples) * 10  # 简化的 RSI

    return df


@pytest.fixture
def feature_loader():
    """创建特征加载器 fixture"""
    from src.features.loader.strategy_feature_loader import StrategyFeatureLoader

    return StrategyFeatureLoader(
        feature_deps_path=str(PROJECT_ROOT / "config" / "feature_dependencies.yaml"),
        cache_dir=str(PROJECT_ROOT / "cache" / "features"),
        use_disk_cache=False,  # 测试时禁用缓存
        use_memory_cache=True,
        max_workers=2,
    )


@pytest.fixture
def strategy_config():
    """加载策略配置 fixture"""
    from src.time_series_model.strategy_config import StrategyConfigLoader

    strategy_dir = PROJECT_ROOT / "config" / "strategies" / "sr_reversal"
    config_loader = StrategyConfigLoader(strategy_dir)
    return config_loader.load()
