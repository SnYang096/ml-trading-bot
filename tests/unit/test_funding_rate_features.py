import numpy as np
import pandas as pd

from src.features.time_series.funding_rate_features import (
    compute_funding_rate_features_from_df,
    compute_funding_scene_semantic_scores_from_df,
)


def test_funding_rate_join_is_causal_merge_asof(tmp_path):
    # Create synthetic funding-rate parquet shards (same schema as downloader output)
    # Funding happens every 8 hours.
    sym = "BTCUSDT"
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00+00:00",
            "2024-01-01 08:00:00+00:00",
            "2024-01-01 16:00:00+00:00",
        ]
    )
    fr = pd.DataFrame(
        {"_symbol": sym, "funding_rate": [0.001, 0.002, -0.001]}, index=idx
    )
    p = tmp_path / f"{sym}_2024-01_funding_rate.parquet"
    fr.to_parquet(p)

    # Bar timestamps (4H)
    bar_idx = pd.date_range("2024-01-01 04:00:00+00:00", periods=6, freq="4H")
    df = pd.DataFrame({"close": 100.0, "_symbol": sym}, index=bar_idx)

    out = compute_funding_rate_features_from_df(df, funding_rate_dir=str(tmp_path))

    # No look-ahead: 04:00 uses 00:00; 08:00 uses 08:00; 12:00 uses 08:00; 16:00 uses 16:00; 20:00 uses 16:00
    got = out["funding_rate"].to_list()
    assert np.isclose(got[0], 0.001)
    assert np.isclose(got[1], 0.002)
    assert np.isclose(got[2], 0.002)
    assert np.isclose(got[3], -0.001)
    assert np.isclose(got[4], -0.001)


def test_funding_scene_scores_are_bounded():
    idx = pd.date_range("2024-01-01", periods=10, freq="4H", tz="UTC")
    df = pd.DataFrame(
        {
            "funding_rate_abs_zscore_50": np.linspace(0.0, 4.0, 10),
            "compression_score": np.linspace(0.0, 1.0, 10),
            "trend_r2_20": np.linspace(1.0, 0.0, 10),
        },
        index=idx,
    )
    out = compute_funding_scene_semantic_scores_from_df(df)
    for c in [
        "funding_compression_score",
        "funding_ignition_score",
        "funding_absorption_score",
        "funding_exhaustion_scene_score",
    ]:
        assert c in out.columns
        assert ((out[c] >= 0.0) & (out[c] <= 1.0)).all()
