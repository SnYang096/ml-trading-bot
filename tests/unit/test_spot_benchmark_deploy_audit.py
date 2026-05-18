"""Smoke tests for spot_accum BH benchmarks + deploy % series + funnel audit."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from scripts.event_backtest import (
    ClosedTrade,
    _bucket_spot_accum_funnel_row,
    _compute_deploy_quote_pct_series,
    _compute_spot_accum_accumulation_audit,
    _compute_spot_buy_hold_benchmarks,
)


def _dt(*args):
    return datetime(*args, tzinfo=timezone.utc)


def test_bucket_spot_accum_funnel_row_classifies_modes():
    assert (
        _bucket_spot_accum_funnel_row(
            {"accumulation_policy": "bull_exposure_stop_deploy"}
        )
        == "bull_exposure_stop"
    )
    assert _bucket_spot_accum_funnel_row({"prefilter": True}) == "prefilter_pass"
    assert (
        _bucket_spot_accum_funnel_row(
            {
                "prefilter": False,
                "alignment_used": True,
                "accumulation_transition_override": True,
            }
        )
        == "transition_override_path"
    )
    assert (
        _bucket_spot_accum_funnel_row(
            {
                "prefilter": False,
                "alignment_used": True,
                "prefilter_alignment_override": True,
                "accumulation_transition_override": False,
            }
        )
        == "prefilter_recent_alignment_only"
    )
    assert (
        _bucket_spot_accum_funnel_row({"prefilter": False, "alignment_used": False})
        == "prefilter_hard_deny"
    )


def test_deploy_quote_pct_series_respects_trade_interval():
    budgets = {"BTCUSDT": 500.0, "SOLUSDT": 500.0}
    spot_budget = {
        "symbol_budgets_usdt": budgets,
        "equity_usdt": 1000.0,
    }

    trades = [
        ClosedTrade(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=1.0,
            exit_price=1.2,
            entry_time=_dt(2024, 1, 1, 0, 0),
            exit_time=_dt(2024, 1, 3, 0, 0),
            atr_at_entry=1.0,
            pnl_r=0.0,
            pnl_usd=0.0,
            exit_reason="x",
            notional_usdt=100.0,
            qty_base=100.0,
            archetype="spot_accum",
        ),
    ]

    iso = ["2024-01-02T00:00:00+00:00", "2024-01-03T02:00:00+00:00"]

    seq = _compute_deploy_quote_pct_series(trades, [], spot_budget, iso)
    assert seq["status"] == "ok"
    # 100 USDT notionally open vs 1000 quote budget caps across symbols => 10% of sum budget.
    assert seq["total_deployed_quote_pct_of_sum_budget"][0] == pytest.approx(10.0)
    # after exit at 03:02 should be flat 0 deployed
    assert seq["total_deployed_quote_pct_of_sum_budget"][1] == pytest.approx(0.0)
    btc_share = seq["per_symbol_pct_of_symbol_budget"]["BTCUSDT"][0]
    assert btc_share == pytest.approx((100.0 / 500.0) * 100.0)


def test_accumulation_audit_counts_rows():
    rows = [
        {"strategy": "spot_accum", "accumulation_policy": "bull_exposure_stop_deploy"},
        {"strategy": "spot_accum", "prefilter": True},
        {"strategy": "spot_accum", "prefilter": False, "alignment_used": False},
    ]
    audit = _compute_spot_accum_accumulation_audit(rows)
    assert audit["status"] == "ok"
    assert audit["eval_rows_used"] == 3
    sh = audit["shares_eval_count"]
    assert sh["bull_exposure_stop"] == pytest.approx(1.0 / 3.0)


def test_spot_buy_hold_benchmarks_align_curves():
    t0 = _dt(2024, 6, 1, 12, 0)
    ix = pd.date_range(start=t0, periods=10, freq="min", tz="UTC")
    # linear ramp in close for predictable returns
    dfb = pd.DataFrame(
        {"close": [100 + i * 10 for i in range(10)], "open": 100}, index=ix
    )
    dfs = pd.DataFrame({"close": [20 + i for i in range(10)], "open": 20}, index=ix)

    budgets = {"BTCUSDT": 700.0, "SOLUSDT": 300.0}
    bm = _compute_spot_buy_hold_benchmarks(
        equity_ts_iso=[ts.isoformat() for ts in ix],
        bars_by_sym={"BTCUSDT": dfb.copy(), "SOLUSDT": dfs.copy()},
        spot_budget={
            "equity_usdt": 1000.0,
            "symbol_budgets_usdt": budgets,
        },
    )
    assert bm["status"] == "ok"
    assert bm["btc_hold_equity_usdt_curve"][-1] > bm["initial_cash_usdt"]
    assert bm["ew_hold_equity_usdt_curve"][-1] > bm["initial_cash_usdt"]
    assert len(bm["btc_hold_equity_usdt_curve"]) == len(ix)
