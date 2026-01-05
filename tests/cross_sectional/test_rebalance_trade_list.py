import pandas as pd

from src.cross_sectional.rebalance_trade_list import (
    TradeListConfig,
    build_trade_list_from_rebalance_log,
)


def test_build_trade_list_from_rebalance_log_basic():
    # close prices for 2 timestamps and 2 symbols
    idx = pd.MultiIndex.from_product(
        [
            pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"], utc=True),
            ["AAA", "BBB"],
        ],
        names=["timestamp", "symbol"],
    )
    close = pd.Series([100.0, 110.0, 105.0, 100.0], index=idx, name="close")

    rb = pd.DataFrame(
        [
            {
                "rebalance_ts": "2025-01-01T00:00:00Z",
                "long_symbols_json": '["AAA"]',
                "short_symbols_json": '["BBB"]',
            },
            {
                "rebalance_ts": "2025-01-02T00:00:00Z",
                "long_symbols_json": '["AAA"]',
                "short_symbols_json": '["BBB"]',
            },
        ]
    )

    cfg = TradeListConfig(
        mode="market_neutral", gross_leverage=1.0, max_weight=1.0, cash_buffer=0.0
    )
    out = build_trade_list_from_rebalance_log(close=close, rb=rb, cfg=cfg)
    assert not out.empty
    assert set(out["symbol"]) == {"AAA", "BBB"}
    assert set(out["side"]) == {"LONG", "SHORT"}
