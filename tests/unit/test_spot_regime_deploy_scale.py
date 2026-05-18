from scripts.event_backtest import _spot_regime_unit_multiplier


def test_regime_unit_tiers():
    cfg = {
        "enabled": True,
        "tiers": [
            {"max_score_exclusive": 1.0, "unit_multiplier": 2.0},
            {"max_score_exclusive": 2.0, "unit_multiplier": 1.5},
            {"max_score_exclusive": 999.0, "unit_multiplier": 1.0},
        ],
    }
    assert _spot_regime_unit_multiplier(0.5, cfg) == 2.0
    assert _spot_regime_unit_multiplier(1.2, cfg) == 1.5
    assert _spot_regime_unit_multiplier(1.9, cfg) == 1.5
    assert _spot_regime_unit_multiplier(2.5, cfg) == 1.0
