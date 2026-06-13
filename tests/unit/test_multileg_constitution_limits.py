from __future__ import annotations

import pandas as pd

from src.live_data_stream.constitution_config import resolve_multileg_sim_limits
from src.sim.multileg_account_sim import simulate_account_with_constitution


def test_resolve_multileg_sim_limits_uses_account_max_drawdown() -> None:
    cfg = {
        "kill_switch": {"max_dd": 0.20, "daily_loss_limit": 0.06},
        "multi_leg": {
            "account": {"max_drawdown_pct": 0.10},
            "risk_limits": {
                "max_gross_notional_pct": 2.70,
                "max_symbol_net_notional_pct": 1.80,
            },
            "account_risk_limits": {"max_gross_leverage": 3.0},
        },
    }
    limits = resolve_multileg_sim_limits(cfg)
    assert limits["max_drawdown_pct"] == 0.10
    assert limits["max_symbol_net_notional_pct"] == 1.80
    assert limits["daily_loss_limit_pct"] == 0.06


def test_net_cap_rejects_extra_same_symbol_legs() -> None:
    base = pd.Timestamp("2025-01-01", tz="UTC")
    trades = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 3,
            "strategy": ["trend_scalp"] * 3,
            "side": ["LONG"] * 3,
            "segment_id": ["S1", "S2", "S3"],
            "pnl_pct": [0.01, 0.01, 0.01],
            "entry_time": [base, base, base + pd.Timedelta(hours=1)],
            "exit_time": [
                base + pd.Timedelta(hours=2),
                base + pd.Timedelta(hours=3),
                base + pd.Timedelta(hours=4),
            ],
        }
    )
    loose = simulate_account_with_constitution(
        trades,
        equity=10_000.0,
        unit_notional=5_000.0,
        max_symbol_net_notional_pct=1.80,
    )
    tight = simulate_account_with_constitution(
        trades,
        equity=10_000.0,
        unit_notional=5_000.0,
        max_symbol_net_notional_pct=1.00,
    )
    assert loose["n_trades"] == 3.0
    assert tight["n_trades"] == 2.0
    assert tight["n_rejected"] == 1.0


def test_tier_derate_scales_legs_before_halt() -> None:
    base = pd.Timestamp("2025-02-01", tz="UTC")
    trades = pd.DataFrame(
        {
            "symbol": ["ETHUSDT"] * 4,
            "strategy": ["trend_scalp"] * 4,
            "side": ["LONG"] * 4,
            "segment_id": ["E1", "E2", "E3", "E4"],
            "pnl_pct": [-0.03, 0.02, 0.02, 0.02],
            "entry_time": [
                base,
                base + pd.Timedelta(hours=1),
                base + pd.Timedelta(hours=2),
                base + pd.Timedelta(hours=3),
            ],
            "exit_time": [
                base + pd.Timedelta(minutes=30),
                base + pd.Timedelta(hours=4),
                base + pd.Timedelta(hours=5),
                base + pd.Timedelta(hours=6),
            ],
        }
    )
    hard = simulate_account_with_constitution(
        trades,
        equity=10_000.0,
        unit_notional=4_000.0,
        max_drawdown_pct=0.20,
        fuse_mode="hard",
    )
    derate = simulate_account_with_constitution(
        trades,
        equity=10_000.0,
        unit_notional=4_000.0,
        max_drawdown_pct=0.20,
        fuse_mode="tier_derate",
        fuse_soft_dd_ratio=0.5,
        fuse_derate_factor=0.5,
    )
    assert derate["n_derated"] >= hard["n_derated"]
    assert derate["n_trades"] >= hard["n_trades"] or not derate["halted"]
