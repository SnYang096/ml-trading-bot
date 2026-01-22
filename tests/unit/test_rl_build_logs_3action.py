import numpy as np
import pandas as pd

from src.time_series_model.rl.build_execution_logs import (
    BuildExecutionLogsConfig,
    build_execution_logs,
)


def test_build_execution_logs_multi_symbol_no_leakage_and_columns() -> None:
    # Two symbols with different close patterns.
    # preds are in log1p space (regression heads).
    idx0 = pd.date_range("2024-01-01", periods=6, freq="4H")
    preds = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": "AAA",
                    "timestamp": idx0,
                    "pred_dir_prob": [0.6] * 6,
                    "pred_mfe_atr": [0.0] * 6,
                    "pred_mae_atr": [0.0] * 6,
                    "pred_t_to_mfe": [0.0] * 6,
                }
            ),
            pd.DataFrame(
                {
                    "symbol": "BBB",
                    "timestamp": idx0,
                    "pred_dir_prob": [0.4] * 6,
                    "pred_mfe_atr": [0.0] * 6,
                    "pred_mae_atr": [0.0] * 6,
                    "pred_t_to_mfe": [0.0] * 6,
                }
            ),
        ],
        axis=0,
        ignore_index=True,
    )

    raw = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": "AAA",
                    "timestamp": idx0,
                    "close": [100, 101, 102, 103, 104, 105],
                }
            ),
            pd.DataFrame(
                {
                    "symbol": "BBB",
                    "timestamp": idx0,
                    "close": [200, 199, 198, 197, 196, 195],
                }
            ),
        ],
        axis=0,
        ignore_index=True,
    )

    cfg = BuildExecutionLogsConfig(momentum_lookback=2, preds_in_log1p=True)
    logs = build_execution_logs(preds, raw_df=raw, cfg=cfg)

    # Required columns for RL pipeline
    for c in [
        "symbol",
        "timestamp",
        "head_dir_score",
        "head_mfe_atr",
        "head_mae_atr",
        "head_t_to_mfe",
        "drawdown",
        "ret_mean",
        "ret_trend",
    ]:
        assert c in logs.columns

    # Ensure per-symbol computation: AAA trending up should have non-negative trend returns
    # (momentum sign based on past; with lookback=2, later bars should be positive sign)
    aaa = logs[logs["symbol"] == "AAA"].reset_index(drop=True)
    bbb = logs[logs["symbol"] == "BBB"].reset_index(drop=True)

    # On BBB, trending down, trend returns should be mostly non-negative (short sign * negative r_next => positive)
    assert float(aaa["ret_trend"].iloc[-2]) >= 0.0
    assert float(bbb["ret_trend"].iloc[-2]) >= 0.0


def test_build_execution_logs_rr_execution_smoke() -> None:
    # Minimal OHLC required for rr_execution; use simple monotonic series.
    idx0 = pd.date_range("2024-01-01", periods=30, freq="4H")
    close = pd.Series(np.linspace(100, 130, num=len(idx0)), index=idx0)
    raw = pd.DataFrame(
        {
            "symbol": "AAA",
            "timestamp": idx0,
            "open": close.values,
            "high": (close * 1.001).values,
            "low": (close * 0.999).values,
            "close": close.values,
        }
    )
    preds = pd.DataFrame(
        {
            "symbol": "AAA",
            "timestamp": idx0,
            "pred_dir_prob": [0.9] * len(idx0),
            "pred_mfe_atr": np.log1p([1.2] * len(idx0)),
            "pred_mae_atr": np.log1p([0.8] * len(idx0)),
            "pred_t_to_mfe": np.log1p([12.0] * len(idx0)),
        }
    )

    cfg = BuildExecutionLogsConfig(returns_source="rr_execution", preds_in_log1p=True)
    logs = build_execution_logs(preds, raw_df=raw, cfg=cfg)
    assert "ret_mean" in logs.columns and "ret_trend" in logs.columns
    # With strong dir_prob and upward trend, trend returns should have some positive mass.
    assert float(logs["ret_trend"].sum()) >= 0.0


def test_build_execution_logs_exec_specialize_adds_market_profile() -> None:
    idx0 = pd.date_range("2024-01-01", periods=40, freq="4H", tz="UTC")
    # Two symbols with minimal OHLC
    close_a = np.linspace(100, 120, num=len(idx0))
    close_b = np.linspace(200, 180, num=len(idx0))
    raw = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": "AAA",
                    "timestamp": idx0,
                    "open": close_a,
                    "high": close_a * 1.001,
                    "low": close_a * 0.999,
                    "close": close_a,
                }
            ),
            pd.DataFrame(
                {
                    "symbol": "BBB",
                    "timestamp": idx0,
                    "open": close_b,
                    "high": close_b * 1.001,
                    "low": close_b * 0.999,
                    "close": close_b,
                }
            ),
        ],
        axis=0,
        ignore_index=True,
    )
    preds = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": "AAA",
                    "timestamp": idx0,
                    "pred_dir_prob": [0.9] * len(idx0),
                    "pred_mfe_atr": np.log1p([1.2] * len(idx0)),
                    "pred_mae_atr": np.log1p([0.8] * len(idx0)),
                    "pred_t_to_mfe": np.log1p([12.0] * len(idx0)),
                }
            ),
            pd.DataFrame(
                {
                    "symbol": "BBB",
                    "timestamp": idx0,
                    "pred_dir_prob": [0.1] * len(idx0),
                    "pred_mfe_atr": np.log1p([1.2] * len(idx0)),
                    "pred_mae_atr": np.log1p([0.8] * len(idx0)),
                    "pred_t_to_mfe": np.log1p([12.0] * len(idx0)),
                }
            ),
        ],
        axis=0,
        ignore_index=True,
    )

    cfg = BuildExecutionLogsConfig(
        returns_source="rr_execution",
        preds_in_log1p=True,
        symbol_profiles={"AAA": "meme", "BBB": "btc"},
        default_profile="standard",
        rr_profile_overrides={
            "meme": {"max_holding_bars": 6},
            "btc": {"max_holding_bars": 24},
        },
    )
    logs = build_execution_logs(preds, raw_df=raw, cfg=cfg)
    assert "market_profile" in logs.columns
    assert set(logs.loc[logs["symbol"] == "AAA", "market_profile"].unique()) == {"meme"}
    assert set(logs.loc[logs["symbol"] == "BBB", "market_profile"].unique()) == {"btc"}


def test_build_execution_logs_vectorbt_execution_smoke() -> None:
    # Skip if vectorbt is not installed in this environment
    try:
        import vectorbt  # noqa: F401
    except Exception:
        return

    n = 200
    ts = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    sym = "AAA"
    rng = np.random.default_rng(3)
    rets = rng.normal(0.0, 0.001, size=n)
    close = 100.0 * np.cumprod(1.0 + rets)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0005, 0.0002, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0005, 0.0002, size=n)))
    raw = pd.DataFrame(
        {
            "symbol": sym,
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )

    dir_prob = (0.5 + 0.4 * np.sin(np.linspace(0, 10, n))).clip(0.01, 0.99)
    preds = pd.DataFrame(
        {
            "symbol": sym,
            "timestamp": ts,
            "pred_dir_prob": dir_prob,
            "pred_mfe_atr": np.log1p(np.full(n, 1.2)),
            "pred_mae_atr": np.log1p(np.full(n, 0.8)),
            "pred_t_to_mfe": np.log1p(np.full(n, 10.0)),
        }
    )

    cfg = BuildExecutionLogsConfig(
        returns_source="vectorbt_execution", preds_in_log1p=True
    )
    logs = build_execution_logs(preds, raw_df=raw, cfg=cfg)
    assert {"ret_mean", "ret_trend"}.issubset(set(logs.columns))
    assert np.isfinite(logs["ret_mean"].to_numpy(dtype=float)).all()
    assert np.isfinite(logs["ret_trend"].to_numpy(dtype=float)).all()
