import numpy as np
import pandas as pd

from src.features.time_series.utils_dtw_features import (
    extract_dtw_features,
    extract_dtw_features_from_series,
)


def test_dtw_from_series_matches_df_entrypoint_small():
    idx = pd.date_range("2024-01-01", periods=80, freq="5min")
    close = pd.Series(
        np.linspace(100, 110, len(idx)) + np.sin(np.arange(len(idx)) / 5), index=idx
    )
    dist = pd.Series(np.linspace(0.5, 2.0, len(idx)), index=idx)
    atr = pd.Series(np.linspace(1.0, 1.2, len(idx)), index=idx)

    df = pd.DataFrame(
        {"close": close, "dist_to_nearest_sr": dist, "atr": atr}, index=idx
    )

    params = dict(
        window=[15, 20],
        template_filter=["hammer", "double_bottom"],
        compute_only_near_sr=False,
        sr_dist_col="dist_to_nearest_sr",
        sr_threshold=1.5,
        normalize_distance=True,
        warping_window=0.1,
        use_c=True,
    )

    df_out = extract_dtw_features(df, price_col="close", **params)
    s_out = extract_dtw_features_from_series(
        close=close, dist_to_nearest_sr=dist, atr=atr, **params
    )

    assert list(df_out.columns) == list(s_out.columns)
    # allow object columns (dtw_best_match_*) to compare exactly as strings
    for c in df_out.columns:
        assert df_out[c].equals(s_out[c])
