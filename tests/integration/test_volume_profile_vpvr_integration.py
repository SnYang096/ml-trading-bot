from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on sys.path for direct execution

from src.features.loader.feature_computer import FeatureComputer  # noqa: E402


def test_volume_profile_vpvr_smoke_and_narrow() -> None:
    """
    Integration check:
    - Compute `volume_profile_vpvr` on tiny OHLCV.
    - Assert expected `vpvr_*` columns exist.
    - Assert no unexpected extra columns leak (narrow result).
    """
    idx = pd.date_range("2024-01-01", periods=160, freq="15min")
    close = pd.Series(
        np.linspace(100, 110, len(idx)) + np.sin(np.arange(len(idx)) / 7), index=idx
    )
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.2
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.2
    volume = pd.Series(np.linspace(1000, 1500, len(idx)), index=idx)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )

    import yaml

    cfg = yaml.safe_load(Path("config/feature_dependencies.yaml").read_text())
    features = cfg["features"]

    pfc = FeatureComputer(
        cache_dir=None,
        use_disk_cache=False,
        use_memory_cache=False,
        use_monthly_cache=False,
    )

    out = pfc.compute_features_parallel(
        df=df,
        features=features,
        requested_features=["volume_profile_vpvr"],
        fit=True,
    )

    vpvr_cols = features["volume_profile_vpvr"]["output_columns"]
    assert all(c in out.columns for c in vpvr_cols)

    base_cols = ["open", "high", "low", "close", "volume"]
    assert set(out.columns) <= set(base_cols + vpvr_cols)
