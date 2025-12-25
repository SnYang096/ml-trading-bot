import pandas as pd

from src.time_series_model.rl.bc_dataset import Router3Action
from src.time_series_model.rl.reward import RewardConfig
from src.time_series_model.rl.sim_env_3action import (
    SimEnvConfig,
    TradingSimEnv3Action,
    simulate_3action_episode,
)


def test_simulate_3action_episode_basic_no_cost() -> None:
    df = pd.DataFrame(
        {
            "ret_mean": [0.01, 0.01, 0.01, 0.01],
            "ret_trend": [0.02, 0.02, 0.02, 0.02],
        }
    )
    actions = [
        int(Router3Action.NO_TRADE),
        int(Router3Action.MEAN),
        int(Router3Action.TREND),
        int(Router3Action.TREND),
    ]
    out = simulate_3action_episode(
        df,
        actions=actions,
        cfg=SimEnvConfig(
            entry_delay=0, cost_per_turnover=0.0, slippage_bps=0.0, initial_equity=1.0
        ),
    )
    assert len(out) == len(df)
    # last equity should be > 1 due to positive returns
    assert out["equity"].iloc[-1] > 1.0


def test_env_step_drawdown_stop_forces_no_trade() -> None:
    # Construct a scenario where TREND loses heavily, triggers drawdown stop.
    df = pd.DataFrame(
        {
            "ret_mean": [0.0, 0.0, 0.0, 0.0, 0.0],
            "ret_trend": [-0.2, -0.2, 0.0, 0.0, 0.0],
        }
    )
    cfg = SimEnvConfig(
        entry_delay=0,
        cost_per_turnover=0.0,
        max_drawdown_stop=0.2,
        cooldown_steps=2,
        initial_equity=1.0,
    )
    env = TradingSimEnv3Action(df, cfg=cfg, reward_cfg=RewardConfig(clip_abs=None))
    env.reset()

    # step 0: take TREND, big loss -> dd hits 0.2, cooldown will trigger
    _, _, _, info0 = env.step(Router3Action.TREND)
    assert info0["equity"] < 1.0

    # subsequent steps should be forced NO_TRADE via cooldown, exposure should be 0
    _, _, _, info1 = env.step(Router3Action.TREND)
    assert info1["mode_eff"] == "NO_TRADE"
    _, _, _, info2 = env.step(Router3Action.MEAN)
    assert info2["mode_eff"] == "NO_TRADE"


def test_env_cost_turnover_applied() -> None:
    df = pd.DataFrame(
        {
            "ret_mean": [0.0, 0.0],
            "ret_trend": [0.0, 0.0],
        }
    )
    cfg = SimEnvConfig(entry_delay=0, cost_per_turnover=0.01, initial_equity=1.0)
    env = TradingSimEnv3Action(df, cfg=cfg, reward_cfg=RewardConfig(clip_abs=None))
    env.reset()

    # switch from 0 exposure to trend exposure -> turnover cost
    _, _, _, info0 = env.step(Router3Action.TREND)
    assert info0["cost"] > 0.0
