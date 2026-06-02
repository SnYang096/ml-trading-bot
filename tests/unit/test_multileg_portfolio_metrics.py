import pandas as pd
import pytest

from scripts.pipeline.multileg_portfolio_metrics import portfolio_pnl_from_trades


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
    assert agg["return_pct"] == pytest.approx(17.5)


def test_single_symbol_no_division() -> None:
    trades = pd.DataFrame({"symbol": ["BTCUSDT"], "pnl_per_capital": [0.08]})
    agg = portfolio_pnl_from_trades(trades)
    assert agg["return_pct"] == agg["return_pct_pooled"] == 8.0
