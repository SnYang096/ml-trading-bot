from __future__ import annotations

import pandas as pd

from src.sim.multileg_account_sim import (
    apply_multileg_segment_gates,
    filter_trades_by_segment_blocks,
)


def _seg_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment_id": "C1",
                "symbol": "BTCUSDT",
                "start": "2025-01-01",
                "end": "2025-01-05",
            },
            {
                "segment_id": "C2",
                "symbol": "ETHUSDT",
                "start": "2025-01-02",
                "end": "2025-01-04",
            },
            {
                "segment_id": "C3",
                "symbol": "SOLUSDT",
                "start": "2025-01-03",
                "end": "2025-01-06",
            },
            {
                "segment_id": "C4",
                "symbol": "BNBUSDT",
                "start": "2025-01-03",
                "end": "2025-01-07",
            },
        ]
    )


def test_mutex_blocks_trend_when_chop_owns_symbol() -> None:
    chop = _seg_rows().iloc[:1]  # BTC chop
    trend = pd.DataFrame(
        [
            {
                "segment_id": "T1",
                "symbol": "BTCUSDT",
                "start": "2025-01-02",
                "end": "2025-01-04",
            }
        ]
    )
    stats = apply_multileg_segment_gates(chop, trend, max_concurrent_grid_symbols=0)
    assert "T1" in stats.blocked_trend_segment_ids
    assert stats.blocked_chop_segments == 0


def test_chop_concurrent_cap_blocks_fourth_symbol() -> None:
    chop = _seg_rows()
    stats = apply_multileg_segment_gates(
        chop, pd.DataFrame(), max_concurrent_grid_symbols=3
    )
    assert "C4" in stats.blocked_chop_segment_ids
    assert stats.peak_chop_symbols == 3


def test_filter_trades_drops_blocked_segment_legs() -> None:
    trades = pd.DataFrame(
        {
            "segment_id": ["C4", "C4", "C1"],
            "pnl_pct": [0.01, 0.01, 0.02],
            "entry_time": pd.to_datetime(["2025-01-01"] * 3, utc=True),
            "exit_time": pd.to_datetime(["2025-01-02"] * 3, utc=True),
        }
    )
    out = filter_trades_by_segment_blocks(trades, {"C4"})
    assert len(out) == 1
    assert out.iloc[0]["segment_id"] == "C1"
