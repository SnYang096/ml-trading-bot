from src.time_series_model.rl.router_policy import apply_action_to_decision
from src.time_series_model.rl.router_types import RouterAction, RouterDecision


def test_apply_action_global_pause_vetoes() -> None:
    base = RouterDecision(router_name="sr", gated=True, score=0.5, position_size=0.4)
    action = RouterAction(
        router_enabled={"sr": True}, capital_multiplier={"sr": 1.0}, global_pause=True
    )
    applied = apply_action_to_decision(base=base, action=action)
    assert applied.decision.gated is False
    assert applied.decision.position_size == 0.0
    assert applied.reason == "global_pause"


def test_apply_action_router_disabled_vetoes() -> None:
    base = RouterDecision(router_name="sr", gated=True, score=0.5, position_size=0.4)
    action = RouterAction(
        router_enabled={"sr": False}, capital_multiplier={"sr": 1.0}, global_pause=False
    )
    applied = apply_action_to_decision(base=base, action=action)
    assert applied.decision.gated is False
    assert applied.decision.position_size == 0.0
    assert applied.reason == "router_disabled"


def test_apply_action_capital_multiplier_scales_position_only() -> None:
    base = RouterDecision(router_name="sr", gated=True, score=0.5, position_size=0.4)
    action = RouterAction(
        router_enabled={"sr": True}, capital_multiplier={"sr": 1.5}, global_pause=False
    )
    applied = apply_action_to_decision(base=base, action=action)
    assert applied.decision.gated is True
    assert abs(applied.decision.position_size - 0.6) < 1e-12
    assert applied.decision.score == 0.5
    assert applied.reason and "capital_multiplier" in applied.reason
