import numpy as np
import pandas as pd

from src.time_series_model.rl.router_diagnostics import diagnose_router_from_logs


def test_router_diagnostics_smoke_multi_symbol() -> None:
    n = 200
    ts = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")

    # AAA mostly TREND, BBB mostly MEAN
    mode_a = ["TREND"] * n
    mode_b = ["MEAN"] * n

    # returns: trend slightly better than mean
    rng = np.random.default_rng(0)
    ret_mean = 0.001 + rng.normal(0.0, 0.0005, size=n)
    ret_trend = 0.0015 + rng.normal(0.0, 0.0005, size=n)

    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": "AAA",
                    "timestamp": ts,
                    "mode": mode_a,
                    "ret_mean": ret_mean,
                    "ret_trend": ret_trend,
                }
            ),
            pd.DataFrame(
                {
                    "symbol": "BBB",
                    "timestamp": ts,
                    "mode": mode_b,
                    "ret_mean": ret_mean,
                    "ret_trend": ret_trend,
                }
            ),
        ],
        axis=0,
        ignore_index=True,
    )

    meta, metrics, per_symbol, rolling = diagnose_router_from_logs(df)
    assert "cfg" in meta
    assert metrics["symbols"] == 2
    assert len(per_symbol) == 2
    # Action distributions should differ
    js_vals = per_symbol["js_to_pooled"].to_numpy(dtype=float)
    assert float(np.max(js_vals)) > 0.01
    # Rolling drift should exist with defaults (n=200, window=300 -> may be empty if min_periods > n)
    # So we only assert schema
    if len(rolling):
        assert "roll_js_to_symbol_base" in rolling.columns
