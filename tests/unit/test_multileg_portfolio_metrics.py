import pandas as pd
import pytest

from scripts.pipeline.multileg_portfolio_metrics import (
    build_portfolio_equity_curve,
    dual_add_summary_fields,
    portfolio_daily_returns_from_trades,
    portfolio_daily_sharpe_from_trades,
    portfolio_equity_from_trades,
    portfolio_metrics_from_trades,
    portfolio_pnl_from_trades,
)


def test_portfolio_return_is_equal_weight_mean() -> None:
    trades = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT"],
            "pnl_per_capital": [0.10, 0.05, 0.20],
        }
    )
    agg = portfolio_pnl_from_trades(trades)
    assert agg["n_symbols"] == 2
    assert agg["sum_pnl_per_capital_pooled"] == pytest.approx(0.35)
    assert agg["portfolio_pnl_per_capital"] == pytest.approx(0.175)
    assert agg["return_pct_pooled"] == pytest.approx(35.0)
    assert agg["return_pct_eq_mean"] == pytest.approx(17.5)


def test_single_symbol_no_division() -> None:
    trades = pd.DataFrame({"symbol": ["BTCUSDT"], "pnl_per_capital": [0.08]})
    agg = portfolio_pnl_from_trades(trades)
    assert agg["return_pct_eq_mean"] == agg["return_pct_pooled"] == 8.0


def test_timeline_return_matches_eq_mean_when_no_overlap() -> None:
    trades = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "pnl_per_capital": [0.10, 0.20],
            "exit_time": ["2024-01-01", "2024-01-02"],
        }
    )
    agg = portfolio_metrics_from_trades(trades)
    assert agg["return_pct"] == agg["return_pct_timeline"] == pytest.approx(15.0)
    assert agg["return_pct_eq_mean"] == pytest.approx(15.0)


def test_timeline_max_drawdown_differs_from_per_symbol_path() -> None:
    trades = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT"],
            "pnl_per_capital": [0.20, -0.15, 0.05],
            "exit_time": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
            ],
        }
    )
    agg = portfolio_metrics_from_trades(trades)
    curve = build_portfolio_equity_curve(trades)
    assert agg["return_pct"] == pytest.approx(5.0)
    assert agg["return_pct_eq_mean"] == pytest.approx(5.0)
    assert agg["max_drawdown_portfolio"] == pytest.approx(curve["drawdown"].min())
    assert agg["max_drawdown_portfolio"] < 0.0


def test_build_portfolio_equity_curve_weights_and_cumsum() -> None:
    trades = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "pnl_per_capital": [0.30, -0.12, 0.06],
            "exit_time": [
                "2024-03-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
                "2024-02-01T00:00:00Z",
            ],
        }
    )
    curve = build_portfolio_equity_curve(trades)
    assert list(curve["portfolio_pnl_per_capital"]) == pytest.approx(
        [-0.12 / 3, 0.06 / 3, 0.30 / 3]
    )
    assert list(curve["cum_pnl_per_capital"]) == pytest.approx([-0.04, -0.02, 0.08])
    assert curve["equity"].iloc[-1] == pytest.approx(1.08)
    assert curve["drawdown"].max() == pytest.approx(0.0)
    assert curve["cum_pnl_per_capital"].iloc[0] == pytest.approx(-0.04)


def test_portfolio_equity_from_trades_empty_and_missing_time() -> None:
    empty = portfolio_equity_from_trades(pd.DataFrame())
    assert empty["n_symbols"] == 0
    assert empty["return_pct_timeline"] == 0.0

    no_time = pd.DataFrame(
        {"symbol": ["BTCUSDT"], "pnl_per_capital": [0.10]},
    )
    timeline = portfolio_equity_from_trades(no_time)
    assert timeline["return_pct_timeline"] == 0.0
    diag = portfolio_pnl_from_trades(no_time)
    assert diag["return_pct_eq_mean"] == pytest.approx(10.0)


def test_portfolio_metrics_primary_fields_alias_timeline() -> None:
    trades = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "pnl_per_capital": [0.10, 0.20],
            "exit_time": ["2024-01-01", "2024-01-02"],
        }
    )
    agg = portfolio_metrics_from_trades(trades)
    assert agg["return_pct"] == agg["return_pct_timeline"]
    assert agg["sum_pnl_per_capital"] == agg["portfolio_pnl_per_capital_timeline"]
    assert agg["max_drawdown_portfolio"] == agg["max_drawdown_timeline"]


def test_dual_add_summary_fields_includes_timeline_columns() -> None:
    trades = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "pnl_per_capital": [0.10, -0.04],
            "pnl_pct": [0.10, -0.04],
            "exit_time": ["2024-01-01", "2024-01-02"],
        }
    )
    segments = pd.DataFrame(
        {
            "pnl_per_capital": [0.06, -0.04],
            "max_drawdown": [-0.01, -0.02],
        }
    )
    row = dual_add_summary_fields(trades, segments)
    assert row["trades"] == 2
    assert row["segments"] == 2
    assert row["return_pct"] == row["return_pct_timeline"]
    assert row["return_pct_pooled"] == pytest.approx(6.0)
    assert row["return_pct_eq_mean"] == pytest.approx(3.0)
    assert row["trade_win_rate"] == pytest.approx(0.5)
    assert row["segment_win_rate"] == pytest.approx(0.5)
    assert row["worst_segment"] == pytest.approx(-0.04)


def test_dual_add_summary_fields_empty_trades() -> None:
    row = dual_add_summary_fields(pd.DataFrame(), pd.DataFrame())
    assert row["trades"] == 0
    assert row["return_pct"] == 0.0
    assert row["return_pct_timeline"] == 0.0
    assert row["daily_sharpe"] == 0.0
    assert "trade_win_rate" not in row


def test_portfolio_daily_returns_weights_by_n_symbols() -> None:
    trades = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "pnl_per_capital": [0.20, 0.20],
            "exit_time": ["2024-01-01", "2024-01-01"],
        }
    )
    daily = portfolio_daily_returns_from_trades(trades)
    assert daily.sum() == pytest.approx(0.20)


def test_portfolio_daily_returns_vs_pooled_resample() -> None:
    trades = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT"],
            "pnl_per_capital": [0.10, 0.06, 0.04],
            "exit_time": ["2024-01-01", "2024-01-01", "2024-01-02"],
        }
    )
    pooled_daily = (
        trades.assign(exit_time=pd.to_datetime(trades["exit_time"], utc=True))
        .set_index("exit_time")["pnl_per_capital"]
        .resample("1D")
        .sum()
    )
    portfolio_daily = portfolio_daily_returns_from_trades(trades)
    assert portfolio_daily.loc["2024-01-01"] == pytest.approx(0.08)
    assert portfolio_daily.loc["2024-01-02"] == pytest.approx(0.02)
    assert portfolio_daily.sum() == pytest.approx(pooled_daily.sum() / 2.0)


def test_portfolio_daily_sharpe_exposed_on_metrics() -> None:
    trades = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"],
            "pnl_per_capital": [0.02, 0.03, -0.01, 0.01],
            "exit_time": ["2024-01-01", "2024-01-03", "2024-01-02", "2024-01-04"],
        }
    )
    agg = portfolio_metrics_from_trades(trades)
    assert agg["daily_sharpe"] == pytest.approx(
        portfolio_daily_sharpe_from_trades(trades)
    )
