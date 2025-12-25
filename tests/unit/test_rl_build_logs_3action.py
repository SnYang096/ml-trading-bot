import numpy as np
import pandas as pd

from src.time_series_model.rl.build_logs_3action import (
    BuildLogs3ActionConfig,
    build_logs_3action,
)


def test_build_logs_3action_multi_symbol_no_leakage_and_columns() -> None:
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

    cfg = BuildLogs3ActionConfig(momentum_lookback=2, preds_in_log1p=True)
    logs = build_logs_3action(preds, raw_df=raw, cfg=cfg, mode_df=None)

    # Required columns for RL pipeline
    for c in [
        "symbol",
        "timestamp",
        "mode",
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


def test_build_logs_3action_rr_execution_smoke() -> None:
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

    cfg = BuildLogs3ActionConfig(returns_source="rr_execution", preds_in_log1p=True)
    logs = build_logs_3action(preds, raw_df=raw, cfg=cfg, mode_df=None)
    assert "ret_mean" in logs.columns and "ret_trend" in logs.columns
    # With strong dir_prob and upward trend, trend returns should have some positive mass.
    assert float(logs["ret_trend"].sum()) >= 0.0
