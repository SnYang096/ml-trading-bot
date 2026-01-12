import pytest


@pytest.mark.unit
def test_no_rsi_rule_based_fallback_in_nautilus_strategy_with_features():
    p = "/workspaces/ml_trading_bot/src/time_series_model/live/nautilus_strategy_with_features.py"
    txt = open(p, "r", encoding="utf-8").read()
    # This repo's "strategy subtraction" constitution forbids indicator-mean entry logic.
    assert "Simple RSI-based signal" not in txt
    assert 'if "rsi" in features_df.columns' not in txt
    assert "rsi < 30" not in txt
    assert "rsi > 70" not in txt
