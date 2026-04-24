"""Smoke tests for SRB staged 2b runtime (cross + EMA arm)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.time_series_model.live.srb_staged_entry_2b import SrbStagedEntry2bRuntime


def _tiny_uptrend_df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0.05, 0.3, size=n))
    high = close + rng.uniform(0.05, 0.2, size=n)
    low = close - rng.uniform(0.05, 0.2, size=n)
    open_ = np.r_[close[0], close[:-1]]
    atr = np.full(n, 0.5)
    ema_pos = np.linspace(-0.02, 0.05, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="120min", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1e6),
            "atr": atr,
            "ema_1200_position": ema_pos,
        },
        index=idx,
    )


def test_runtime_from_yaml_and_arm_consume() -> None:
    rt = SrbStagedEntry2bRuntime.from_execution_block(
        {
            "cross": {"confirm_k": 2, "cooldown_bars": 0},
            "post_2a_max_bars": 5,
            "ema_slope_bars": 1,
            "arm_pcm_bars": 3,
        }
    )
    df = _tiny_uptrend_df(50)
    ts = df.index[25]
    row = df.loc[ts].to_dict()
    for bi in range(1, 26):
        t0 = df.index[bi - 1]
        r0 = df.loc[t0].to_dict()
        rt.advance(
            symbol="TEST",
            df_srb=df,
            ts=pd.Timestamp(t0),
            bar_idx=bi,
            row=r0,
            has_srb_position=False,
        )
    rt.advance(
        symbol="TEST",
        df_srb=df,
        ts=pd.Timestamp(ts),
        bar_idx=26,
        row=row,
        has_srb_position=False,
    )
    # 未必每根都 arm；只测 API 不抛 + consume 幂等
    rt.consume_arm("TEST")
    rt.consume_arm("TEST")


def test_match_arm_rejects_without_arm() -> None:
    rt = SrbStagedEntry2bRuntime.from_execution_block({})
    assert rt.match_arm("X", "LONG", 1) is False
