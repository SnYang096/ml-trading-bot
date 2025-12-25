import json

import pandas as pd

from src.time_series_model.rl.router_logging import RouterEpisodeLogger, RouterStepLog
from src.time_series_model.rl.router_types import (
    RouterAction,
    RouterContext,
    RouterDecision,
    RouterHeads,
)


def test_router_action_clip_and_json() -> None:
    a = RouterAction(
        router_enabled={"sr": True, "trend": False},
        capital_multiplier={"sr": 10.0, "trend": -1.0},
        global_pause=True,
    )
    c = a.clipped(min_mult=0.0, max_mult=2.0)
    assert c.capital_multiplier["sr"] == 2.0
    assert c.capital_multiplier["trend"] == 0.0
    payload = c.as_dict()
    assert payload["global_pause"] is True
    assert payload["router_enabled"]["sr"] is True
    assert isinstance(payload["capital_multiplier"]["sr"], float)


def test_router_episode_logger_to_frame_schema() -> None:
    logger = RouterEpisodeLogger()
    ctx = RouterContext(
        timestamp="2025-01-01T00:00:00Z", symbol="BTCUSDT", timeframe="4H"
    )
    heads = RouterHeads(
        dir_score=0.2, mfe_atr=1.5, mae_atr=0.8, t_to_mfe=3.0, persistence=0.7
    )
    action = RouterAction(
        router_enabled={"sr": True}, capital_multiplier={"sr": 1.2}, global_pause=False
    )
    decision = RouterDecision(
        router_name="sr", gated=True, score=0.42, position_size=0.3
    )
    rec = RouterStepLog(
        ctx=ctx,
        heads=heads,
        action=action,
        decision=decision,
        pnl=0.01,
        turnover=0.2,
        cost=0.001,
        equity=1.01,
        drawdown=0.0,
    )
    logger.append(rec)
    df = logger.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1

    # Required columns
    for col in [
        "timestamp",
        "symbol",
        "timeframe",
        "router_name",
        "gated",
        "score",
        "position_size",
        "action_json",
        "pnl",
        "turnover",
        "cost",
        "equity",
        "drawdown",
        "head_dir_score",
        "head_mfe_atr",
        "head_mae_atr",
        "head_t_to_mfe",
    ]:
        assert col in df.columns

    # action_json should be valid JSON
    parsed = json.loads(df.loc[0, "action_json"])
    assert parsed["router_enabled"]["sr"] is True
    assert parsed["capital_multiplier"]["sr"] == 1.2
