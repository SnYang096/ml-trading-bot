import pandas as pd

from src.time_series_model.rl.reward import RewardConfig
from src.time_series_model.rl.replay_buffer import (
    build_replay_transitions_from_router_logs,
)
from src.time_series_model.rl.router_logging import RouterEpisodeLogger, RouterStepLog
from src.time_series_model.rl.router_types import (
    RouterAction,
    RouterContext,
    RouterDecision,
    RouterHeads,
)


def _mk_step(
    *,
    ts: str,
    symbol: str,
    dir_score: float,
    mfe: float,
    mae: float,
    ttm: float,
    action_pause: bool,
    mult: float,
    pnl: float,
    drawdown: float,
    mode: str,
) -> RouterStepLog:
    ctx = RouterContext(timestamp=ts, symbol=symbol, timeframe="4H", regime_score=0.1)
    heads = RouterHeads(dir_score=dir_score, mfe_atr=mfe, mae_atr=mae, t_to_mfe=ttm)
    action = RouterAction(
        router_enabled={"sr": True, "trend": True},
        capital_multiplier={"sr": mult, "trend": mult},
        global_pause=action_pause,
    )
    decision = RouterDecision(
        router_name="sr", gated=True, score=0.1, position_size=0.2
    )
    return RouterStepLog(
        ctx=ctx,
        heads=heads,
        action=action,
        decision=decision,
        mode=mode,
        pnl=pnl,
        drawdown=drawdown,
    )


def test_build_replay_transitions_from_router_logs() -> None:
    logger = RouterEpisodeLogger()
    logger.append(
        _mk_step(
            ts="2025-01-01T00:00:00Z",
            symbol="BTC",
            dir_score=0.2,
            mfe=1.0,
            mae=0.7,
            ttm=1.1,
            action_pause=False,
            mult=1.0,
            pnl=0.01,
            drawdown=0.0,
            mode="MEAN",
        )
    )
    logger.append(
        _mk_step(
            ts="2025-01-01T04:00:00Z",
            symbol="BTC",
            dir_score=0.3,
            mfe=1.2,
            mae=0.6,
            ttm=1.0,
            action_pause=False,
            mult=0.5,
            pnl=-0.02,
            drawdown=0.1,
            mode="TREND",
        )
    )
    df = logger.to_frame()
    assert isinstance(df, pd.DataFrame)

    cfg = RewardConfig(action_change_weight=1.0, diversity_weight=0.0, clip_abs=None)
    transitions = build_replay_transitions_from_router_logs(df, reward_cfg=cfg)
    assert len(transitions) == 1

    t0 = transitions[0]
    assert t0.symbol == "BTC"
    assert t0.timestamp == "2025-01-01T00:00:00Z"
    assert "head_dir_score" in t0.state
    assert "head_dir_score" in t0.next_state
    assert isinstance(t0.action, dict)
    assert t0.action.get("mode") in {"MEAN", "TREND", "NO_TRADE"}
    assert isinstance(t0.reward, float)
    assert t0.done is True
