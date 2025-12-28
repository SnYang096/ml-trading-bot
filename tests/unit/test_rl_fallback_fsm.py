from src.time_series_model.rl.fallback_fsm import (
    FallbackFSM,
    GateConfig,
    GateInputs,
    RouterControlState,
)


def test_candidate_promotes_to_active_after_streak() -> None:
    fsm = FallbackFSM(cfg=GateConfig(promote_min_days=3, cooldown_days=5))
    fsm.state = RouterControlState.RL_CANDIDATE

    good = GateInputs(
        max_dd_rule=0.2,
        max_dd_rl=0.15,
        switch_rate_rule=0.1,
        switch_rate_rl=0.05,
        pnl_dd_rule=1.0,
        pnl_dd_rl=1.0,
    )
    fsm.step(good)
    fsm.step(good)
    out = fsm.step(good)
    assert out["state"] == RouterControlState.RL_ACTIVE.value
    assert out["transition_reason"] == "promoted"


def test_active_suspends_on_hard_dd() -> None:
    fsm = FallbackFSM(cfg=GateConfig(dd_ratio_max=1.2, cooldown_days=2))
    fsm.state = RouterControlState.RL_ACTIVE

    bad = GateInputs(
        max_dd_rule=0.1,
        max_dd_rl=0.2,  # > 0.12
        switch_rate_rule=0.1,
        switch_rate_rl=0.1,
        pnl_dd_rule=1.0,
        pnl_dd_rl=1.0,
    )
    out = fsm.step(bad)
    assert out["state"] == RouterControlState.RL_SUSPENDED.value
    assert out["transition_reason"] == "suspend_hard"
    assert out["cooldown_left_days"] == 2


def test_active_suspends_on_hard_sharpe_drop() -> None:
    fsm = FallbackFSM(cfg=GateConfig(sharpe_ratio_min=0.9, cooldown_days=2))
    fsm.state = RouterControlState.RL_ACTIVE

    inp = GateInputs(
        max_dd_rule=0.2,
        max_dd_rl=0.2,
        switch_rate_rule=0.1,
        switch_rate_rl=0.1,
        sharpe_rule=2.0,
        sharpe_rl=1.0,  # < 1.8 triggers hard_sharpe
    )
    out = fsm.step(inp)
    assert out["state"] == RouterControlState.RL_SUSPENDED.value
    assert bool(out["gates"]["hard_sharpe"]) is True


def test_suspended_returns_to_candidate_after_cooldown() -> None:
    fsm = FallbackFSM(cfg=GateConfig(cooldown_days=2, promote_min_days=2))
    fsm.state = RouterControlState.RL_SUSPENDED
    fsm.cooldown_left_days = 2

    good = GateInputs(
        max_dd_rule=0.2, max_dd_rl=0.15, switch_rate_rule=0.1, switch_rate_rl=0.05
    )
    out1 = fsm.step(good)
    assert out1["state"] == RouterControlState.RL_SUSPENDED.value
    out2 = fsm.step(good)
    assert out2["state"] == RouterControlState.RL_CANDIDATE.value
    assert out2["transition_reason"] == "cooldown_done"
