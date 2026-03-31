from typing import List

from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.portfolio.live_pcm import LivePCM


class _StaticStrategy:
    def __init__(self, intents: List[TradeIntent]):
        self._intents = intents

    def decide(self, *, features, symbol, bars=None):
        return list(self._intents)


def _intent(archetype: str = "bpc-long-120T") -> TradeIntent:
    return TradeIntent(
        action="LONG",
        symbol="BTCUSDT",
        archetype=archetype,
        confidence=0.8,
        size_multiplier=1.0,
        execution_profile={
            "rr_constraints": {"stop_loss_r": 1.0, "max_stop_pct": 0.02}
        },
    )


def _policy() -> dict:
    return {
        "enabled": True,
        "risk_budget_mode": "percent",
        "disable_slot_gate": True,
        "max_total_risk_pct": 0.05,
        "max_family_risk_pct": 0.03,
        "max_symbol_risk_pct": 0.015,
        "breakeven_release_enabled": True,
        "breakeven_residual_risk_pct": 0.005,
        "expansion": {
            "enabled": True,
            "target_families": ["bpc"],
            "trigger_released_slots": 1,
            "step_released_slots": 1,
            "step_multiplier": 0.5,
            "max_family_multiplier": 3.0,
            "max_total_multiplier": 2.0,
            "max_symbol_multiplier": 1.6,
        },
        "shrink": {
            "enabled": True,
            "by_drawdown": [{"drawdown_gte": 0.03, "cap_multiplier": 0.75}],
        },
        "stress": {
            "enabled": True,
            "shock_pct": 0.5,
            "max_stress_loss_pct": 0.1,
        },
        "deleveraging": {
            "enabled": True,
            "freeze_new_entries_ratio": 0.9,
            "tiers": [{"trigger_ratio": 1.0, "reduce_to_ratio": 0.6}],
        },
    }


def test_profit_expansion_activates_after_released_slot():
    pcm = LivePCM(max_slots=8)
    pol = _policy()
    pcm._constitution["risk_budget_policy"] = pol

    base = pcm._dynamic_caps(family="bpc", policy=pol)
    assert base["family_cap"] == 0.03

    pcm._slot_risk_frac["BTCUSDT:bpc-long-120T"] = 0.005
    expanded = pcm._dynamic_caps(family="bpc", policy=pol)
    assert expanded["family_cap"] > base["family_cap"]


def test_drawdown_shrink_contracts_caps():
    pcm = LivePCM(max_slots=8)
    pol = _policy()
    pcm._constitution["risk_budget_policy"] = pol
    pcm._latest_features = {"drawdown": 0.04}

    caps = pcm._dynamic_caps(family="me", policy=pol)
    assert caps["total_cap"] == 0.05 * 0.75
    assert caps["family_cap"] == 0.03 * 0.75


def test_stress_guard_blocks_risk_increasing_intent():
    pcm = LivePCM(max_slots=8)
    pol = _policy()
    pol["max_total_risk_pct"] = 0.2
    pol["max_family_risk_pct"] = 0.2
    pol["max_symbol_risk_pct"] = 0.2
    pcm._constitution["risk_budget_policy"] = pol
    pcm.register("bpc-long-120T", _StaticStrategy([_intent()]))

    out = pcm.decide(features={"close": 100.0, "atr": 2.0}, symbol="BTCUSDT")
    assert out == []
    assert pcm.get_stats()["risk_budget_runtime"]["reject_counts"]["stress_cap"] >= 1


def test_tiered_deleveraging_emits_weakest_first():
    pcm = LivePCM(max_slots=8)
    pol = _policy()
    pcm._constitution["risk_budget_policy"] = pol
    pcm._slot_risk_frac = {
        "BTCUSDT:bpc-a": 0.010,
        "BTCUSDT:bpc-b": 0.030,
        "BTCUSDT:bpc-c": 0.020,
    }
    pcm._slot_loss_r = {
        "BTCUSDT:bpc-a": 1.5,
        "BTCUSDT:bpc-b": 0.2,
        "BTCUSDT:bpc-c": 2.0,
    }

    evictions, freeze = pcm._plan_tiered_deleveraging(symbol="BTCUSDT", policy=pol)
    assert freeze is True
    assert evictions[:2] == [("BTCUSDT", "bpc-c"), ("BTCUSDT", "bpc-a")]
