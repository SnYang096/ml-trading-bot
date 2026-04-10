"""
FER 配置管线集成测试 — 覆盖 config/strategies/fer/archetypes/ 当前 YAML，
不依赖行情文件；用于回归（prefilter / direction / gate / entry）。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STRATEGIES_ROOT = _REPO_ROOT / "config" / "strategies"


def _fer() -> GenericLiveStrategy:
    return GenericLiveStrategy(
        strategy_name="fer",
        strategies_root=str(_STRATEGIES_ROOT),
        primary_timeframe="120T",
        bar_minutes=120,
    )


def _baseline_features() -> dict:
    """满足当前 prefilter + gate + direction(仅 signed failure) + entry(做多分支) 的一组正值。"""
    return {
        "timestamp": "2024-08-15T12:00:00+00:00",
        "sr_strength_max": 0.6,
        "fer_impulse_failure_score": 0.25,
        "fer_aggressor_absorption": 0.50,
        "fer_ols_width_norm": 0.50,
        "fer_sr_failed_breakout_direction_signed": 0.0,
        "fer_impulse_failure_direction_signed": 1.0,
        # gate：fer_volume_price_divergence 在 (0.425, 0.675) 内会 deny；取低端外
        "fer_volume_price_divergence": 0.20,
        "fer_trapped_shorts_score": 0.15,
        "fer_range_pos_20": 0.35,
        "bars_since_local_low": 0.15,
        "close": 60000.0,
        "atr": 100.0,
    }


@pytest.fixture(scope="module")
def fer_strategy() -> GenericLiveStrategy:
    return _fer()


def test_prefilter_rejects_when_impulse_failure_below_threshold(fer_strategy):
    f = deepcopy(_baseline_features())
    f["fer_impulse_failure_score"] = 0.05
    out = fer_strategy.decide(features=f, symbol="BTCUSDT")
    assert out == []
    assert fer_strategy._last_funnel.get("prefilter") is False
    assert "fer_impulse_failure_score" in str(
        fer_strategy._last_funnel.get("prefilter_reason", "")
    )


def test_prefilter_rejects_when_absorption_below_threshold(fer_strategy):
    f = deepcopy(_baseline_features())
    f["fer_aggressor_absorption"] = 0.10
    out = fer_strategy.decide(features=f, symbol="BTCUSDT")
    assert out == []
    assert fer_strategy._last_funnel.get("prefilter") is False
    assert "fer_aggressor_absorption" in str(
        fer_strategy._last_funnel.get("prefilter_reason", "")
    )


def test_prefilter_rejects_when_ols_width_below_threshold(fer_strategy):
    f = deepcopy(_baseline_features())
    f["fer_ols_width_norm"] = 0.10
    out = fer_strategy.decide(features=f, symbol="BTCUSDT")
    assert out == []
    assert fer_strategy._last_funnel.get("prefilter") is False
    assert "fer_ols_width_norm" in str(
        fer_strategy._last_funnel.get("prefilter_reason", "")
    )


def test_no_direction_when_signed_failure_missing(fer_strategy):
    """direction 仅认 SR / impulse signed；二者皆为 0 时不应给方向成交。"""
    f = deepcopy(_baseline_features())
    f["fer_impulse_failure_direction_signed"] = 0.0
    f["fer_sr_failed_breakout_direction_signed"] = 0.0
    out = fer_strategy.decide(features=f, symbol="BTCUSDT")
    assert out == []
    assert fer_strategy._last_funnel.get("prefilter") is True
    assert fer_strategy._last_funnel.get("direction") is False


def test_gate_vetoes_fer_volume_price_divergence_mid_band(fer_strategy):
    f = deepcopy(_baseline_features())
    f["fer_volume_price_divergence"] = 0.55
    out = fer_strategy.decide(features=f, symbol="BTCUSDT")
    assert out == []
    assert fer_strategy._last_funnel.get("gate") is False
    reasons = fer_strategy._last_funnel.get("gate_reasons") or []
    assert len(reasons) >= 1


def test_entry_filter_blocks_when_long_branch_fails_range(fer_strategy):
    f = deepcopy(_baseline_features())
    f["fer_range_pos_20"] = 0.90
    out = fer_strategy.decide(features=f, symbol="BTCUSDT")
    assert out == []
    assert fer_strategy._last_funnel.get("entry_filter") is False


def test_baseline_can_emit_signal(fer_strategy):
    f = _baseline_features()
    out = fer_strategy.decide(features=f, symbol="BTCUSDT")
    assert len(out) == 1
    assert fer_strategy._last_funnel.get("prefilter") is True
    assert fer_strategy._last_funnel.get("direction") is True
    assert fer_strategy._last_funnel.get("gate") is True
    assert fer_strategy._last_funnel.get("entry_filter") is True


def test_config_files_exist():
    arch = _STRATEGIES_ROOT / "fer" / "archetypes"
    for name in ("prefilter.yaml", "gate.yaml", "entry_filters.yaml", "direction.yaml"):
        p = arch / name
        assert p.is_file(), f"missing {p}"
