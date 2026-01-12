import pytest

from src.time_series_model.portfolio.slot_sizing import (
    compute_slot_size_from_risk,
    estimate_stop_return_frac,
    risk_only_down,
)


@pytest.mark.unit
def test_estimate_stop_return_frac():
    # price=100, atr=2 => atr_pct=0.02; stop_atr=1.5 => 0.03
    assert estimate_stop_return_frac(
        price=100.0, atr=2.0, stop_atr=1.5
    ) == pytest.approx(0.03)


@pytest.mark.unit
def test_compute_slot_size_risk_limited_and_leverage_capped():
    # equity=10k, risk=1%, risk_usd=100
    # price=100, atr=2, stop_atr=1 => stop_ret=0.02
    # risk-limited notional=100/0.02=5000 => qty=50
    out = compute_slot_size_from_risk(
        equity_usd=10_000,
        risk_frac=0.01,
        price=100.0,
        atr=2.0,
        stop_atr=1.0,
        max_leverage=10.0,
    )
    assert out.stop_return_frac == pytest.approx(0.02)
    assert out.notional_usd == pytest.approx(5000.0)
    assert out.qty == pytest.approx(50.0)

    # Now cap leverage hard: max_leverage=0.2 => notional cap=2000 => qty=20
    out2 = compute_slot_size_from_risk(
        equity_usd=10_000,
        risk_frac=0.01,
        price=100.0,
        atr=2.0,
        stop_atr=1.0,
        max_leverage=0.2,
    )
    assert out2.notional_usd == pytest.approx(2000.0)
    assert out2.qty == pytest.approx(20.0)


@pytest.mark.unit
def test_compute_slot_size_bad_inputs_return_zero():
    out = compute_slot_size_from_risk(
        equity_usd=10_000,
        risk_frac=0.01,
        price=0.0,
        atr=2.0,
        stop_atr=1.0,
    )
    assert out.qty == 0.0
    assert out.notional_usd == 0.0


@pytest.mark.unit
def test_risk_only_down():
    assert risk_only_down(
        prev_risk_frac=None, proposed_risk_frac=0.02
    ) == pytest.approx(0.02)
    assert risk_only_down(
        prev_risk_frac=0.015, proposed_risk_frac=0.02
    ) == pytest.approx(0.015)
    assert risk_only_down(
        prev_risk_frac=0.015, proposed_risk_frac=0.01
    ) == pytest.approx(0.01)
