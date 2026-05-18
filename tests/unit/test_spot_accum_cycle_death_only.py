from datetime import datetime, timezone

from src.time_series_model.live.position_logic import enforce_position


def _pos():
    return {
        "symbol": "BTCUSDT",
        "side": "LONG",
        "entry_price": 100.0,
        "entry_time": datetime(2022, 1, 1, tzinfo=timezone.utc),
        "atr_at_entry": 1.0,
        "bar_minutes": 120,
        "structural_exit": "weekly_macro_cycle",
        "regime_lifecycle_exit": {
            "bull_min_score": 4.0,
            "cycle_exit_requires_bull": True,
            "arm_cycle_exit_min_peak": 4.0,
            "allow_regime_risk_off": False,
        },
        "_regime_saw_bull": False,
        "_regime_peak_score": 0.0,
    }


def test_weekly_cycle_death_ignored_before_bull_exposure():
    pos = _pos()
    reason, _ = enforce_position(
        pos,
        price_high=100.0,
        price_low=100.0,
        price_close=100.0,
        now=datetime(2022, 6, 1, tzinfo=timezone.utc),
        macro_cycle_exit_signal=1.0,
        macro_regime_score=3.0,
    )
    assert reason is None


def test_weekly_cycle_death_allowed_after_bull_exposure():
    pos = _pos()
    enforce_position(
        pos,
        price_high=100.0,
        price_low=100.0,
        price_close=100.0,
        now=datetime(2022, 6, 1, tzinfo=timezone.utc),
        macro_cycle_exit_signal=0.0,
        macro_regime_score=4.0,
    )
    reason, _ = enforce_position(
        pos,
        price_high=100.0,
        price_low=100.0,
        price_close=100.0,
        now=datetime(2022, 7, 1, tzinfo=timezone.utc),
        macro_cycle_exit_signal=1.0,
        macro_regime_score=3.0,
    )
    assert reason == "structural_exit_weekly_macro_cycle"


def test_weekly_cycle_death_requires_min_days_after_bull_when_configured():
    pos = _pos()
    pos["regime_lifecycle_exit"]["cycle_exit_min_days_after_bull"] = 180
    enforce_position(
        pos,
        price_high=100.0,
        price_low=100.0,
        price_close=100.0,
        now=datetime(2022, 6, 1, tzinfo=timezone.utc),
        macro_cycle_exit_signal=0.0,
        macro_regime_score=5.0,
    )
    reason, _ = enforce_position(
        pos,
        price_high=100.0,
        price_low=100.0,
        price_close=100.0,
        now=datetime(2022, 7, 1, tzinfo=timezone.utc),
        macro_cycle_exit_signal=1.0,
        macro_regime_score=3.0,
    )
    assert reason is None
    reason, _ = enforce_position(
        pos,
        price_high=100.0,
        price_low=100.0,
        price_close=100.0,
        now=datetime(2023, 1, 1, tzinfo=timezone.utc),
        macro_cycle_exit_signal=1.0,
        macro_regime_score=3.0,
    )
    assert reason == "structural_exit_weekly_macro_cycle"
