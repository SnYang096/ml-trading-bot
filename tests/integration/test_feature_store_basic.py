from __future__ import annotations

import pandas as pd

from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec
from pathlib import Path


def test_feature_store_write_read_range(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    spec = FeatureStoreSpec(layer="base_v1", symbol="TEST", timeframe="240T")

    idx = pd.date_range("2025-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 10.0,
            "_symbol": "TEST",
            "f1": range(10),
            "f2": range(10),
        },
        index=idx,
    )

    store.write_month(
        spec,
        "2025-01",
        df,
        base_columns=["open", "high", "low", "close", "volume", "_symbol"],
        feature_columns=["f1", "f2"],
        overwrite=True,
        metadata={"unit_test": True},
    )

    out = store.read_range(spec, pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-05"))
    assert not out.empty
    assert out.index.min() >= pd.Timestamp("2025-01-02")
    assert out.index.max() <= pd.Timestamp("2025-01-05")
    assert "f1" in out.columns
    assert "open" in out.columns
