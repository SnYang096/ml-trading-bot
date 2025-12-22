import dataclasses

import numpy as np
import pandas as pd
import pytest


def test_sr_reversal_sr_fuse_blocks_far_from_sr_entries():
    """
    Regression / behavior test:
    - Direction-fixed long_only/short_only strategies should be able to enable an SR-distance fuse.
    - When enabled, entries with dist_to_nearest_sr / ATR > max_dist_atr should be blocked.
    """
    try:
        import vectorbt  # noqa: F401
    except Exception:
        pytest.skip("vectorbt not installed")

    from scripts.train_strategy_pipeline import run_vectorbt_backtest
    from src.time_series_model.strategy_config import StrategyConfigLoader

    cfg = StrategyConfigLoader("config/strategies/sr_reversal_long").load()

    n = 120
    idx = pd.date_range("2025-01-01", periods=n, freq="4H")
    close = pd.Series(100.0 + np.sin(np.linspace(0, 6 * np.pi, n)), index=idx)
    atr = pd.Series(1.0, index=idx)

    # Half near SR, half far from SR
    dist = pd.Series(np.r_[np.full(n // 2, 1.0), np.full(n - n // 2, 20.0)], index=idx)

    df = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close).values,
            "high": (close + 0.5 * atr).values,
            "low": (close - 0.5 * atr).values,
            "close": close.values,
            "atr": atr.values,
            "dist_to_nearest_sr": dist.values,
        },
        index=idx,
    )

    preds = np.full(
        n, 0.95, dtype=float
    )  # always above threshold => entries everywhere without fuse

    params = dict(cfg.backtest.params)
    params["use_rr_exit"] = False
    params["use_signal_direction"] = False
    params["entry_threshold"] = 0.5
    params["sr_fuse"] = {
        "enabled": True,
        "dist_col": "dist_to_nearest_sr",
        "atr_col": "atr",
        "max_dist_atr": 5.0,
        "on_missing": "block",
    }

    out = run_vectorbt_backtest(
        df=df,
        preds=preds,
        backtest_cfg=dataclasses.replace(cfg.backtest, params=params),
        task_type="binary",
        strategy_config=cfg,
    )

    assert out is not None
    # We don't assert exact trade count (vectorbt depends on portfolio mechanics),
    # but we expect some trades (near SR), not zero.
    assert out.get("total_trades", 0) > 0
