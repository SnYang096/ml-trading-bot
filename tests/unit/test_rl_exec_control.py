import numpy as np
import pandas as pd

from src.time_series_model.rl.exec_control import (
    ExecControlConfig,
    control_check_from_logs,
)
from src.time_series_model.rl.sim_env_3action import SimEnvConfig


def test_exec_control_check_smoke_no_kill() -> None:
    n = 200
    ts = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    df = pd.DataFrame(
        {
            "symbol": ["AAA"] * n,
            "timestamp": ts,
            "mode": ["NO_TRADE"] * n,
            "ret_mean": np.zeros(n),
            "ret_trend": np.zeros(n),
        }
    )
    cfg = ExecControlConfig(sim_cfg=SimEnvConfig(entry_delay=1, cost_per_turnover=0.0))
    metrics, per_symbol = control_check_from_logs(df, cfg=cfg)
    assert metrics["kill_switch"] is False
    assert len(per_symbol) == 1


def test_exec_control_kill_on_nan_ratio() -> None:
    n = 100
    ts = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    ret = np.zeros(n)
    ret[:10] = np.nan
    df = pd.DataFrame(
        {
            "symbol": ["AAA"] * n,
            "timestamp": ts,
            "mode": ["TREND"] * n,
            "ret_mean": np.zeros(n),
            "ret_trend": ret,
        }
    )
    cfg = ExecControlConfig(
        max_nan_ratio=0.05, sim_cfg=SimEnvConfig(entry_delay=0, cost_per_turnover=0.0)
    )
    metrics, _ = control_check_from_logs(df, cfg=cfg)
    assert metrics["data_bad"] is True
    assert metrics["kill_switch"] is True
