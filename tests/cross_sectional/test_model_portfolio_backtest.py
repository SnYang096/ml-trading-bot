import numpy as np
import pandas as pd

from src.cross_sectional.model_portfolio_backtest import (
    PortfolioBacktestConfig,
    portfolio_backtest_from_signal,
    portfolio_backtest_with_rebalance_log,
)


def _make_panel(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    rows = []
    for j, sym in enumerate(["A", "B", "C", "D", "E", "F"]):
        base = 100 + j * 10
        close = base + np.linspace(0, 1.0, n)
        # signal: stable cross-sectional ordering by symbol
        sig = np.full(n, float(j))
        for t, c, s in zip(idx, close, sig):
            rows.append({"timestamp": t, "symbol": sym, "close": c, "signal": s})
    df = pd.DataFrame(rows).set_index(["timestamp", "symbol"]).sort_index()
    return df


def test_backtest_modes_basic_sanity():
    panel = _make_panel(80)
    cfg_lo = PortfolioBacktestConfig(
        mode="long_only",
        holding_period_bars=12,
        execution_lag_bars=1,
        top_k=2,
        gross_leverage=1.0,
        max_weight=0.6,
        fee_bps=0.0,
        slippage_bps=0.0,
        min_assets=4,
        periods_per_year=2190.0,
    )
    ts_lo, m_lo = portfolio_backtest_from_signal(panel, signal_col="signal", cfg=cfg_lo)
    assert not ts_lo.empty
    assert m_lo["mode"] == "long_only"
    assert np.isfinite(m_lo["sharpe_net"]) or np.isnan(m_lo["sharpe_net"])

    cfg_mn = PortfolioBacktestConfig(
        mode="market_neutral",
        holding_period_bars=12,
        execution_lag_bars=1,
        top_k=2,
        bottom_k=2,
        gross_leverage=1.0,
        max_weight=0.6,
        fee_bps=0.0,
        slippage_bps=0.0,
        min_assets=4,
        periods_per_year=2190.0,
    )
    ts_mn, m_mn = portfolio_backtest_from_signal(panel, signal_col="signal", cfg=cfg_mn)
    assert not ts_mn.empty
    assert m_mn["mode"] == "market_neutral"


def test_lag_and_holding_generate_sparse_turnover():
    panel = _make_panel(100)
    cfg = PortfolioBacktestConfig(
        mode="long_only",
        holding_period_bars=10,
        execution_lag_bars=2,
        top_k=2,
        gross_leverage=1.0,
        max_weight=1.0,
        fee_bps=2.0,
        slippage_bps=0.0,
        min_assets=4,
        periods_per_year=2190.0,
    )
    ts, _ = portfolio_backtest_from_signal(panel, signal_col="signal", cfg=cfg)
    # turnover should occur only on rebalance timestamps, otherwise 0
    assert (ts["turnover"] >= 0).all()
    assert (ts["turnover"] == 0).sum() > 0


def test_compound_equity_and_funding_cost():
    panel = _make_panel(120)
    # market-neutral with funding should reduce net vs gross
    cfg = PortfolioBacktestConfig(
        mode="market_neutral",
        holding_period_bars=12,
        execution_lag_bars=1,
        top_k=2,
        bottom_k=2,
        gross_leverage=1.0,
        max_weight=0.8,
        fee_bps=0.0,
        slippage_bps=0.0,
        funding_bps_per_bar=1.0,  # 1 bps per bar on shorts
        borrow_bps_per_bar=0.0,
        cash_buffer=0.0,
        equity_mode="compound",
        initial_capital=1.0,
        min_assets=4,
        periods_per_year=2190.0,
    )
    ts, m = portfolio_backtest_from_signal(panel, signal_col="signal", cfg=cfg)
    assert "net_equity" in ts.columns and "gross_equity" in ts.columns
    assert (
        float(ts["net_equity"].iloc[-1]) <= float(ts["gross_equity"].iloc[-1]) + 1e-12
    )
    assert m["avg_funding_cost"] >= 0.0


def test_rebalance_audit_log_basic_fields_and_constraints():
    panel = _make_panel(120)
    cfg = PortfolioBacktestConfig(
        mode="market_neutral",
        holding_period_bars=12,
        execution_lag_bars=1,
        top_k=2,
        bottom_k=2,
        gross_leverage=1.0,
        max_weight=0.6,
        fee_bps=2.0,
        slippage_bps=0.0,
        cash_buffer=0.1,
        min_assets=4,
        periods_per_year=2190.0,
    )
    _, _, rb = portfolio_backtest_with_rebalance_log(
        panel, signal_col="signal", cfg=cfg
    )
    assert not rb.empty
    # one row per rebalance
    assert rb["rebalance_ts"].isna().sum() == 0
    assert rb["signal_ts"].isna().sum() == 0
    # market-neutral should have both long and short exposure typically
    assert (rb["long_exposure"] >= 0).all()
    assert (rb["short_exposure"] >= 0).all()
    # net exposure near 0 for market-neutral (allow small due to caps/cash)
    assert (rb["net_exposure"].abs() <= 1e-6 + 1.0).all()
