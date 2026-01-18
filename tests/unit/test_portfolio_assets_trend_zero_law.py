import pytest

from src.time_series_model.portfolio.portfolio_assets import (
    aggregate_from_symbol_modes,
    compute_portfolio_asset_weights,
    load_portfolio_assets_config,
    trend_zero_law_status,
)


def _write_cfg(tmp_path, *, rules):
    p = tmp_path / "pa.yaml"
    p.write_text(
        """
name: pa_test
assets:
  GLOBAL_TREND: {min_weight: 0.0, max_weight: 0.4}
  GLOBAL_MEAN: {min_weight: 0.2, max_weight: 0.35}
  GLOBAL_CASH: {min_weight: 0.1, max_weight: 1.0}
  HIGH_BETA_OVERLAY: {min_weight: 0.0, max_weight: 0.1}
  DEFENSIVE_MEAN: {min_weight: 0.0, max_weight: 0.25}
router_to_weights:
  global_trend: {p_trend_min: 0.6, regime_entropy_max: 0.4, max_weight: 0.4, crowding_penalty: true}
  global_mean: {base_floor: 0.2, max_weight: 0.35}
  global_cash: {min_weight: 0.1}
  defensive_mean: {regime_entropy_min: 0.5, max_weight: 0.25}
  high_beta_overlay: {p_trend_min: 0.75, crowding_max: 0.3, confidence_min: 0.7, max_weight: 0.1}
trend_zero_law:
  rules:
"""
        + "\n".join([f"  - {r}" for r in rules]),
        encoding="utf-8",
    )
    return str(p)


@pytest.mark.unit
def test_trend_zero_law_triggered_by_entropy(tmp_path):
    cfg_path = _write_cfg(tmp_path, rules=["{name: ent, regime_entropy_gt: 0.1}"])
    cfg = load_portfolio_assets_config(cfg_path)
    decisions = [
        {"symbol": "BTCUSDT", "mode": "TREND"},
        {"symbol": "ETHUSDT", "mode": "MEAN"},
        {"symbol": "SOLUSDT", "mode": "NO_TRADE"},
    ]
    sig = aggregate_from_symbol_modes(decisions=decisions, key_symbols=["BTCUSDT"])
    st = trend_zero_law_status(
        cfg=cfg, sig=sig, gate_veto=False, portfolio_drawdown=0.0
    )
    assert st["triggered"] is True
    assert "ent" in st["reasons"]

    w = compute_portfolio_asset_weights(
        cfg=cfg, sig=sig, gate_veto=False, portfolio_drawdown=0.0
    )
    assert pytest.approx(sum(w.values()), abs=1e-9) == 1.0
    assert w.get("GLOBAL_TREND", 0.0) == 0.0


@pytest.mark.unit
def test_trend_zero_law_triggered_by_key_symbols(tmp_path):
    cfg_path = _write_cfg(
        tmp_path, rules=["{name: keys, require_key_symbols: [BTCUSDT, ETHUSDT]}"]
    )
    cfg = load_portfolio_assets_config(cfg_path)
    decisions = [
        {"symbol": "BTCUSDT", "mode": "TREND"},
        {"symbol": "ETHUSDT", "mode": "MEAN"},  # not TREND => triggers
    ]
    sig = aggregate_from_symbol_modes(
        decisions=decisions, key_symbols=["BTCUSDT", "ETHUSDT"]
    )
    st = trend_zero_law_status(
        cfg=cfg, sig=sig, gate_veto=False, portfolio_drawdown=0.0
    )
    assert st["triggered"] is True
    assert "keys" in st["reasons"]
