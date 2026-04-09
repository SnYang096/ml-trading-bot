"""
FER 配置管线集成测试 — 解释「为何长时间不开仓」的典型机制。

覆盖生产目录 config/strategies/fer/archetypes/ 下当前 YAML，
不依赖行情文件；用于回归与文档化配置冲突。
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
    """过 prefilter + gate(当前两条 hard) + entry_filter + direction 的一组保守正值。"""
    return {
        "timestamp": "2024-08-15T12:00:00+00:00",
        "dist_to_nearest_sr": 0.4,
        "sr_strength_max": 0.6,
        "fer_sr_failed_breakout_direction_signed": 0.0,
        "fer_sr_failed_breakout_score_pct": 0.0,
        "fer_impulse_failure_direction": 0.0,
        "fer_impulse_failure_direction_signed": 0.0,
        # 规则2 CVD：略正 → negate_sign → SHORT
        "cvd_change_5_normalized": 0.02,
        "roc_20": 0.01,
        "fer_volume_price_divergence": 0.2,
        "fer_trapped_shorts_score": 0.1,
        "vol_mom_10": 0.0,
        "close": 60000.0,
        "atr": 100.0,
    }


@pytest.fixture(scope="module")
def fer_strategy() -> GenericLiveStrategy:
    return _fer()


def test_prefilter_rejects_when_price_too_far_from_sr(fer_strategy):
    f = deepcopy(_baseline_features())
    f["dist_to_nearest_sr"] = 2.5
    f["sr_strength_max"] = 0.7
    out = fer_strategy.decide(features=f, symbol="BTCUSDT")
    assert out == []
    assert fer_strategy._last_funnel.get("prefilter") is False
    assert "dist_to_nearest_sr" in str(
        fer_strategy._last_funnel.get("prefilter_reason", "")
    )


def test_prefilter_rejects_when_sr_strength_below_threshold(fer_strategy):
    f = deepcopy(_baseline_features())
    f["sr_strength_max"] = 0.5
    out = fer_strategy.decide(features=f, symbol="BTCUSDT")
    assert out == []
    assert fer_strategy._last_funnel.get("prefilter") is False
    assert "sr_strength_max" in str(
        fer_strategy._last_funnel.get("prefilter_reason", "")
    )


def test_strongly_negative_cvd_long_passes_gate_when_cvd_hard_gate_disabled(
    fer_strategy,
):
    """方向规则2：CVD 负 → LONG；CVD hard gate 已 disabled，不得再整单 veto。"""
    f = deepcopy(_baseline_features())
    f["fer_impulse_failure_direction_signed"] = 0.0
    f["fer_impulse_failure_direction"] = 0.0
    f["cvd_change_5_normalized"] = -0.08
    f["roc_20"] = 0.0
    out = fer_strategy.decide(features=f, symbol="BTCUSDT")
    assert len(out) == 1
    assert fer_strategy._last_funnel.get("prefilter") is True
    assert fer_strategy._last_funnel.get("direction_value") == 1
    assert fer_strategy._last_funnel.get("gate") is True


def test_gate_vetoes_fer_volume_price_divergence_mid_band(fer_strategy):
    f = deepcopy(_baseline_features())
    f["fer_volume_price_divergence"] = 0.55
    out = fer_strategy.decide(features=f, symbol="BTCUSDT")
    assert out == []
    assert fer_strategy._last_funnel.get("gate") is False
    reasons = fer_strategy._last_funnel.get("gate_reasons") or []
    assert len(reasons) >= 1


def test_entry_filter_blocks_when_both_or_legs_fail(fer_strategy):
    f = deepcopy(_baseline_features())
    f["fer_trapped_shorts_score"] = 0.0
    f["vol_mom_10"] = -0.5
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
