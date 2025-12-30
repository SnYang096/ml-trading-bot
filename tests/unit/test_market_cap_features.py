import pandas as pd

from src.features.time_series.market_cap_features import (
    compute_market_cap_normalized_orderflow_from_df,
)


def test_market_cap_features_requires_symbol_column():
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0],
            "volume": [1.0, 2.0],
        },
        index=pd.to_datetime(["2025-01-01", "2025-01-02"]),
    )
    try:
        compute_market_cap_normalized_orderflow_from_df(
            df, market_cap_dir="data/market_cap", on_missing_market_cap="nan"
        )
        assert False, "Expected KeyError when symbol column is missing"
    except KeyError:
        pass


def test_market_cap_features_on_missing_can_nan(tmp_path):
    df = pd.DataFrame(
        {
            "_symbol": ["BTCUSDT", "BTCUSDT"],
            "close": [100.0, 101.0],
            "volume": [1.0, 2.0],
            "buy_qty": [3.0, 1.0],
            "sell_qty": [1.0, 2.0],
        },
        index=pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 04:00:00"]),
    )
    out = compute_market_cap_normalized_orderflow_from_df(
        df,
        market_cap_dir=str(tmp_path),  # empty
        on_missing_market_cap="nan",
        min_market_cap_usd=1.0,
    )
    assert "market_cap_usd" in out.columns
    assert out["market_cap_usd"].isna().all()
    # denom is NaN => normalized outputs NaN
    assert out["dollar_volume_over_mcap"].isna().all()


def test_market_cap_features_static_snapshot_fills_all_days(tmp_path):
    # Write a static (single-day) market cap snapshot
    mdir = tmp_path
    snap_day = pd.to_datetime("2025-01-10", utc=True)
    df_m = pd.DataFrame(
        {"market_cap_usd": [1000.0]}, index=pd.DatetimeIndex([snap_day], name="date")
    )
    df_m.to_parquet(mdir / "BTCUSDT.parquet")

    df = pd.DataFrame(
        {
            "_symbol": ["BTCUSDT"] * 3,
            "close": [100.0, 101.0, 102.0],
            "volume": [1.0, 2.0, 3.0],
            "buy_qty": [1.0, 1.0, 1.0],
            "sell_qty": [0.0, 0.0, 0.0],
        },
        # Dates BEFORE the snapshot day
        index=pd.to_datetime(
            ["2025-01-01 00:00:00", "2025-01-02 00:00:00", "2025-01-03 00:00:00"]
        ),
    )
    out = compute_market_cap_normalized_orderflow_from_df(
        df,
        market_cap_dir=str(mdir),
        on_missing_market_cap="raise",
        min_market_cap_usd=1.0,
    )
    assert out["market_cap_usd"].notna().all()
    assert float(out["market_cap_usd"].iloc[0]) == 1000.0
