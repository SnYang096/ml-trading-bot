from __future__ import annotations

from src.time_series_model.core.constitution.account_risk_guard import (
    BacktestAccountRiskTracker,
    evaluate_account_risk,
    snapshot_for_backtest,
)


def test_evaluate_account_risk_blocks_high_gross_leverage() -> None:
    snap = snapshot_for_backtest(equity_usdt=1000.0, gross_notional=2900.0)
    violations = evaluate_account_risk(
        limits={"enabled": True, "max_gross_leverage": 3.0},
        snapshot=snap,
        proposed_notional=200.0,
    )
    assert violations
    assert "projected_gross_leverage" in violations[0]


def test_evaluate_account_risk_allows_under_cap() -> None:
    snap = snapshot_for_backtest(equity_usdt=1000.0, gross_notional=1000.0)
    violations = evaluate_account_risk(
        limits={
            "enabled": True,
            "max_gross_leverage": 3.0,
            "max_projected_initial_margin_pct": 0.80,
            "min_projected_available_margin_pct": 0.10,
            "margin_stress_leverage": 5.0,
        },
        snapshot=snap,
        proposed_notional=500.0,
    )
    assert violations == []


def test_backtest_tracker_tracks_open_gross() -> None:
    tracker = BacktestAccountRiskTracker(
        limits={"enabled": True, "max_gross_leverage": 3.0},
        equity_usdt=1000.0,
    )
    ok, _ = tracker.allow_open(2000.0)
    assert ok
    tracker.on_open(2000.0)
    ok2, _ = tracker.allow_open(1500.0)
    assert not ok2
    assert tracker.rejected_count == 1
