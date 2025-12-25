from src.time_series_model.rl.reward import (
    RewardConfig,
    compute_action_change_penalty,
    compute_diversity_penalty,
    compute_drawdown_penalty,
    compute_router_reward_from_step,
)


def test_drawdown_penalty_only_triggers_above_limit() -> None:
    p0 = compute_drawdown_penalty(
        dd_ratio=0.9, dd_limit=1.0, dd_weight=10.0, dd_power=2.0
    )
    assert p0 == 0.0
    p1 = compute_drawdown_penalty(
        dd_ratio=1.2, dd_limit=1.0, dd_weight=10.0, dd_power=2.0
    )
    assert p1 > 0.0


def test_action_change_penalty_l1() -> None:
    a0 = {"weights": {"sr": 0.0, "trend": 1.0}}
    a1 = {"weights": {"sr": 0.5, "trend": 0.5}}
    pen = compute_action_change_penalty(prev_action=a0, next_action=a1, weight=2.0)
    # L1 change = |0.5-0| + |0.5-1| = 1.0
    assert abs(pen - 2.0) < 1e-12


def test_diversity_penalty_higher_when_collapsed() -> None:
    uniform = {"sr": 1.0, "trend": 1.0, "breakout": 1.0}
    collapsed = {"sr": 1.0, "trend": 0.0, "breakout": 0.0}
    p_u = compute_diversity_penalty(weights=uniform, weight=1.0)
    p_c = compute_diversity_penalty(weights=collapsed, weight=1.0)
    assert p_c > p_u


def test_router_reward_combines_terms() -> None:
    cfg = RewardConfig(
        pnl_weight=1.0,
        cost_weight=1.0,
        turnover_weight=0.0,
        drawdown_weight=2.0,
        dd_limit=1.0,
        dd_power=2.0,
        action_change_weight=1.0,
        diversity_weight=1.0,
        clip_abs=None,
    )
    r = compute_router_reward_from_step(
        pnl=0.01,
        cost=0.002,
        turnover=0.0,
        dd_ratio=1.2,
        realized_vol=0.0,
        action_prev={"weights": {"sr": 0.0, "trend": 1.0}},
        action_next={"weights": {"sr": 1.0, "trend": 0.0}},
        cfg=cfg,
    )
    # base = pnl - cost = 0.008; dd_pen > 0; action_pen > 0; div_pen > 0
    assert r < 0.008
