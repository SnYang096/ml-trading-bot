import numpy as np
import pandas as pd

from src.features.time_series.utils_interaction_features import (
    compute_sr_strength_combined,
    compute_sr_strength_combined_from_series,
    compute_sr_distance_normalized,
    compute_sr_distance_normalized_from_series,
    compute_dist_to_zz_high,
    compute_dist_to_zz_high_from_series,
    compute_dist_to_zz_low,
    compute_dist_to_zz_low_from_series,
    compute_dist_to_zz_high_atr,
    compute_dist_to_zz_high_atr_from_series,
    compute_dist_to_zz_low_atr,
    compute_dist_to_zz_low_atr_from_series,
    compute_cvd_slope,
    compute_cvd_slope_from_series,
)


def test_sr_and_zz_derived_from_series_matches_df_versions():
    idx = pd.date_range("2024-01-01", periods=50, freq="5min")
    df = pd.DataFrame(
        {
            "sqs": np.linspace(0, 1, len(idx)),
            "dist_to_nearest_sr": np.linspace(0, 10, len(idx)),
            "atr": np.linspace(1, 2, len(idx)),
            "close": np.linspace(100, 110, len(idx)),
            "zz_high_value": np.linspace(105, 115, len(idx)),
            "zz_low_value": np.linspace(95, 105, len(idx)),
            "cvd": np.cumsum(np.random.default_rng(0).normal(0, 1, len(idx))),
        },
        index=idx,
    )

    assert np.allclose(
        compute_sr_strength_combined(df).values,
        compute_sr_strength_combined_from_series(sqs=df["sqs"])[
            "sr_strength_combined"
        ].values,
        equal_nan=True,
    )
    assert np.allclose(
        compute_sr_distance_normalized(df).values,
        compute_sr_distance_normalized_from_series(
            dist_to_nearest_sr=df["dist_to_nearest_sr"], atr=df["atr"]
        )["sr_distance_normalized"].values,
        equal_nan=True,
    )
    assert np.allclose(
        compute_dist_to_zz_high(df).values,
        compute_dist_to_zz_high_from_series(
            close=df["close"], zz_high_value=df["zz_high_value"]
        )["dist_to_zz_high"].values,
        equal_nan=True,
    )
    assert np.allclose(
        compute_dist_to_zz_low(df).values,
        compute_dist_to_zz_low_from_series(
            close=df["close"], zz_low_value=df["zz_low_value"]
        )["dist_to_zz_low"].values,
        equal_nan=True,
    )

    # atr-normalized versions use the dist columns
    dist_high = compute_dist_to_zz_high(df)
    dist_low = compute_dist_to_zz_low(df)
    df2 = df.copy()
    df2["dist_to_zz_high"] = dist_high
    df2["dist_to_zz_low"] = dist_low

    assert np.allclose(
        compute_dist_to_zz_high_atr(df2).values,
        compute_dist_to_zz_high_atr_from_series(
            dist_to_zz_high=dist_high, atr=df["atr"]
        )["dist_to_zz_high_atr"].values,
        equal_nan=True,
    )
    assert np.allclose(
        compute_dist_to_zz_low_atr(df2).values,
        compute_dist_to_zz_low_atr_from_series(dist_to_zz_low=dist_low, atr=df["atr"])[
            "dist_to_zz_low_atr"
        ].values,
        equal_nan=True,
    )

    # cvd slope
    assert np.allclose(
        compute_cvd_slope(df, window=5).values,
        compute_cvd_slope_from_series(cvd=df["cvd"], window=5)["cvd_slope_5"].values,
        equal_nan=True,
    )
