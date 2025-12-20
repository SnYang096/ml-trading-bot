import numpy as np
import pandas as pd

from src.features.time_series.baseline_features import (
    compute_acceleration_3,
    compute_acceleration_3_from_series,
    compute_volume_anomaly,
    compute_volume_anomaly_from_series,
    compute_trend_r2_20,
    compute_trend_r2_20_from_series,
    compute_trend_r2_50,
    compute_trend_r2_50_from_series,
    compute_slope_consistency_score,
    compute_slope_consistency_score_from_series,
    compute_volatility_reversal_score,
    compute_volatility_reversal_score_from_series,
    compute_atr,
    compute_atr_from_series,
    compute_atr_percentile,
    compute_atr_percentile_from_series,
    compute_trend_volatility_alignment,
    compute_trend_volatility_alignment_from_series,
    compute_compression_to_breakout_prob,
    compute_compression_to_breakout_prob_from_series,
    compute_roc_5_from_series,
)


def test_remaining_baseline_series_entrypoints_match_df_versions():
    idx = pd.date_range("2024-01-01", periods=600, freq="5min")
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.2, len(idx))), index=idx)
    high = close + np.abs(rng.normal(0.1, 0.05, len(idx)))
    low = close - np.abs(rng.normal(0.1, 0.05, len(idx)))
    volume = pd.Series(np.abs(rng.normal(1000, 200, len(idx))), index=idx)

    df = pd.DataFrame(
        {"close": close, "high": high, "low": low, "volume": volume}, index=idx
    )

    # acceleration_3
    df_a = compute_acceleration_3(df.copy(), feature_shift=0)
    s_a = compute_acceleration_3_from_series(close=close, feature_shift=0)[
        "acceleration_3"
    ]
    assert np.allclose(df_a["acceleration_3"].values, s_a.values, equal_nan=True)

    # volume_anomaly
    df_v = compute_volume_anomaly(df.copy())
    s_v = compute_volume_anomaly_from_series(volume=volume)["volume_anomaly"]
    assert np.allclose(df_v["volume_anomaly"].values, s_v.values, equal_nan=True)

    # trend r2
    df_r2_20 = compute_trend_r2_20(df.copy(), feature_shift=0)
    s_r2_20 = compute_trend_r2_20_from_series(close=close, feature_shift=0)[
        "trend_r2_20"
    ]
    assert np.allclose(df_r2_20["trend_r2_20"].values, s_r2_20.values, equal_nan=True)

    df_r2_50 = compute_trend_r2_50(df.copy(), feature_shift=0)
    s_r2_50 = compute_trend_r2_50_from_series(close=close, feature_shift=0)[
        "trend_r2_50"
    ]
    assert np.allclose(df_r2_50["trend_r2_50"].values, s_r2_50.values, equal_nan=True)

    # slope consistency
    df_sc = compute_slope_consistency_score(df.copy())
    s_sc = compute_slope_consistency_score_from_series(close=close)[
        "slope_consistency_score"
    ]
    assert np.allclose(
        df_sc["slope_consistency_score"].values, s_sc.values, equal_nan=True
    )

    # volatility reversal score
    df_vrs = compute_volatility_reversal_score(df.copy())
    s_vrs = compute_volatility_reversal_score_from_series(
        high=high, low=low, close=close
    )["volatility_reversal_score"]
    assert np.allclose(
        df_vrs["volatility_reversal_score"].values, s_vrs.values, equal_nan=True
    )

    # atr_percentile
    df_ap = compute_atr_percentile(df.copy(), window=288, shift=1)
    s_ap = compute_atr_percentile_from_series(
        high=high, low=low, close=close, window=288, shift=1
    )["atr_percentile"]
    assert np.allclose(df_ap["atr_percentile"].values, s_ap.values, equal_nan=True)

    # trend_volatility_alignment (depends on roc_5 + atr_percentile internally)
    df_tva = compute_trend_volatility_alignment(
        df.copy(), feature_shift=0, atr_percentile_window=288
    )
    s_tva = compute_trend_volatility_alignment_from_series(
        close=close, high=high, low=low, feature_shift=0, atr_percentile_window=288
    )["trend_volatility_alignment"]
    assert np.allclose(
        df_tva["trend_volatility_alignment"].values, s_tva.values, equal_nan=True
    )

    # compression_to_breakout_prob (simple product); use roc_5 series
    roc_5 = compute_roc_5_from_series(close=close)
    compression_duration = pd.Series(
        np.maximum(0, rng.normal(5, 2, len(idx))), index=idx
    )
    df_cb = pd.DataFrame(
        {"compression_duration": compression_duration, "roc_5": roc_5}, index=idx
    )
    df_cb2 = compute_compression_to_breakout_prob(df_cb.copy())
    s_cb = compute_compression_to_breakout_prob_from_series(
        compression_duration=compression_duration, roc_5=roc_5
    )["compression_to_breakout_prob"]
    if "compression_to_breakout_prob" in df_cb2.columns:
        assert np.allclose(
            df_cb2["compression_to_breakout_prob"].values, s_cb.values, equal_nan=True
        )
    else:
        # legacy returns df unchanged if missing deps; here we provided them, so it should exist
        raise AssertionError(
            "legacy compute_compression_to_breakout_prob did not create output column"
        )


def test_compute_atr_from_series():
    """
    测试修复：compute_atr_from_series 函数

    Bug修复：添加了 compute_atr_from_series 函数，返回 DataFrame 格式
    确保 narrow-IO 模式下能正确工作
    """
    idx = pd.date_range("2024-01-01", periods=100, freq="5min")
    rng = np.random.default_rng(42)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.2, len(idx))), index=idx)
    high = close + np.abs(rng.normal(0.1, 0.05, len(idx)))
    low = close - np.abs(rng.normal(0.1, 0.05, len(idx)))

    # 测试 compute_atr_from_series 返回 DataFrame
    result_df = compute_atr_from_series(high=high, low=low, close=close, period=14)

    # 验证返回类型
    assert isinstance(result_df, pd.DataFrame), "应该返回 DataFrame"
    assert "atr" in result_df.columns, "应该包含 'atr' 列"
    assert len(result_df) == len(idx), "长度应该匹配输入"

    # 验证与原始 compute_atr 的结果一致
    result_series = compute_atr(high, low, close, period=14)
    assert np.allclose(
        result_df["atr"].values, result_series.values, equal_nan=True, rtol=1e-10
    ), "DataFrame 版本应该与 Series 版本结果一致"

    # 验证数值合理性
    atr_values = result_df["atr"].dropna()
    assert len(atr_values) > 0, "应该有有效的 ATR 值"
    assert (atr_values >= 0).all(), "ATR 应该 >= 0"
