import pandas as pd

from src.cross_sectional.factor_backtest import (
    LongShortBacktestConfig,
    long_short_backtest,
)


def _make_panel() -> pd.DataFrame:
    # 2 timestamps, 4 symbols
    ts1 = pd.Timestamp("2025-01-01T00:00:00Z")
    ts2 = pd.Timestamp("2025-01-01T04:00:00Z")
    symbols = ["A", "B", "C", "D"]

    # At ts1, factor ranks: D highest, A lowest
    # At ts2, flip rankings to induce turnover
    rows = []
    for ts, fvals, rets in [
        (ts1, [0.0, 1.0, 2.0, 3.0], [-0.01, 0.0, 0.01, 0.02]),
        (ts2, [3.0, 2.0, 1.0, 0.0], [0.02, 0.01, 0.0, -0.01]),
    ]:
        for sym, f, r in zip(symbols, fvals, rets):
            rows.append(
                {"timestamp": ts, "symbol": sym, "factor": f, "future_return_1": r}
            )
    df = pd.DataFrame(rows)
    return df.set_index(["timestamp", "symbol"]).sort_index()


def test_long_short_backtest_basic_metrics():
    panel = _make_panel()
    cfg = LongShortBacktestConfig(quantiles=2, fee_bps=0.0, min_assets=4)
    ts_df, metrics = long_short_backtest(
        panel, factor_col="factor", target_col="future_return_1", cfg=cfg
    )

    assert not ts_df.empty
    assert metrics["n_timestamps"] == 2.0
    # With 2-quantiles and 4 assets, long = top 2, short = bottom 2
    assert (ts_df["n_long"] == 2.0).all()
    assert (ts_df["n_short"] == 2.0).all()
    # Turnover must be > 0 on second step due to rank flip
    assert ts_df["turnover"].iloc[0] == 0.0
    assert ts_df["turnover"].iloc[1] > 0.0


def test_long_short_backtest_fee_applied():
    panel = _make_panel()
    cfg = LongShortBacktestConfig(quantiles=2, fee_bps=10.0, min_assets=4)  # 10 bps
    ts_df, _ = long_short_backtest(
        panel, factor_col="factor", target_col="future_return_1", cfg=cfg
    )
    assert not ts_df.empty
    # fee is proportional to turnover and non-negative
    assert (ts_df["fee"] >= 0).all()
    # net return should be <= gross return each step
    assert (ts_df["net_return"] <= ts_df["gross_return"] + 1e-12).all()
