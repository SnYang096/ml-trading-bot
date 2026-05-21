"""Unit tests for RegimeConfig + StrategyArchetype regime integration."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.time_series_model.archetype.loader import (
    PrefilterConfig,
    RegimeConfig,
    load_strategy_archetype,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")


# ---------------------------------------------------------------------------
# RegimeConfig.from_yaml roundtrip
# ---------------------------------------------------------------------------


def test_regime_config_defaults_when_file_missing(tmp_path: Path):
    cfg = RegimeConfig.from_yaml(tmp_path / "nonexistent.yaml")
    assert cfg.rules == []
    assert cfg.allowed_regimes == ["bull", "bear", "neutral"]
    assert cfg.allowed_sides == ["long", "short"]
    assert cfg.is_empty is True


def test_regime_config_loads_rules_and_masks(tmp_path: Path):
    regime_yaml = tmp_path / "regime.yaml"
    _write(
        regime_yaml,
        """
        allowed_regimes: [bear, neutral]
        allowed_sides: [short]
        rules:
          - feature: tpc_semantic_chop
            operator: "<="
            value: 0.4
        """,
    )
    cfg = RegimeConfig.from_yaml(regime_yaml)
    assert cfg.allowed_regimes == ["bear", "neutral"]
    assert cfg.allowed_sides == ["short"]
    assert len(cfg.rules) == 1
    assert cfg.rules[0]["feature"] == "tpc_semantic_chop"
    assert cfg.is_empty is False


# ---------------------------------------------------------------------------
# RegimeConfig.evaluate semantics
# ---------------------------------------------------------------------------


def test_regime_evaluate_passes_when_no_rules():
    cfg = RegimeConfig()
    passed, reason = cfg.evaluate({"x": 0.5})
    assert passed is True
    assert reason is None


def test_regime_evaluate_reject_prefixed_with_regime():
    cfg = RegimeConfig(
        rules=[{"feature": "tpc_semantic_chop", "operator": "<=", "value": 0.4}]
    )
    passed, reason = cfg.evaluate({"tpc_semantic_chop": 0.6})
    assert passed is False
    assert reason is not None
    assert reason.startswith("regime_")
    # 必须不漏出 prefilter 前缀
    assert "prefilter_" not in reason


def test_regime_evaluate_passes_when_threshold_satisfied():
    cfg = RegimeConfig(
        rules=[{"feature": "tpc_semantic_chop", "operator": "<=", "value": 0.4}]
    )
    passed, reason = cfg.evaluate({"tpc_semantic_chop": 0.2})
    assert passed is True
    assert reason is None


def test_regime_any_of_rule_evaluates_like_prefilter():
    cfg = RegimeConfig(
        rules=[
            {
                "any_of": [
                    {"feature": "box_pos_120", "operator": "<=", "value": 0.15},
                    {"feature": "box_pos_120", "operator": ">=", "value": 0.85},
                ]
            }
        ]
    )
    # box mid → reject
    passed, reason = cfg.evaluate({"box_pos_120": 0.5})
    assert passed is False
    assert reason is not None and reason.startswith("regime_")

    # box edge → pass
    passed, _ = cfg.evaluate({"box_pos_120": 0.1})
    assert passed is True


# ---------------------------------------------------------------------------
# allowed_sides masking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "allowed,direction,expected",
    [
        (["long", "short"], 1, True),
        (["long", "short"], -1, True),
        (["long", "short"], 0, True),
        (["long"], 1, True),
        (["long"], -1, False),
        (["short"], 1, False),
        (["short"], -1, True),
        ([], 1, False),
        ([], -1, False),
    ],
)
def test_allowed_sides_masking(allowed, direction, expected):
    cfg = RegimeConfig(allowed_sides=list(allowed))
    assert cfg.allows_side(direction) is expected


# ---------------------------------------------------------------------------
# RegimeConfig is a PrefilterConfig subclass — substitutable
# ---------------------------------------------------------------------------


def test_regime_config_is_prefilter_subclass():
    cfg = RegimeConfig(rules=[{"feature": "x", "operator": ">=", "value": 0.0}])
    assert isinstance(cfg, PrefilterConfig)


# ---------------------------------------------------------------------------
# load_strategy_archetype wires regime.yaml when present
# ---------------------------------------------------------------------------


def _minimal_archetype_dir(root: Path, *, with_regime: bool) -> Path:
    pkg = root / "demo_strat"
    arch = pkg / "archetypes"
    arch.mkdir(parents=True)
    _write(arch / "gate.yaml", "hard_gates: []\n")
    _write(arch / "evidence.yaml", "evidence: []\n")
    _write(arch / "execution.yaml", "execution_constraints: {}\n")
    _write(
        arch / "prefilter.yaml",
        """
        rules:
          - feature: pf_feat
            operator: ">="
            value: 0.0
        """,
    )
    if with_regime:
        _write(
            arch / "regime.yaml",
            """
            allowed_sides: [long]
            rules:
              - feature: tpc_semantic_chop
                operator: "<="
                value: 0.4
            """,
        )
    return pkg


def test_load_strategy_archetype_without_regime_file(tmp_path: Path):
    _minimal_archetype_dir(tmp_path, with_regime=False)
    arch = load_strategy_archetype("demo_strat", strategies_root=str(tmp_path))
    assert isinstance(arch.regime, RegimeConfig)
    assert arch.regime.is_empty is True
    assert arch.regime.allows_side(1) is True
    assert arch.regime.allows_side(-1) is True


def test_load_strategy_archetype_with_regime_file(tmp_path: Path):
    _minimal_archetype_dir(tmp_path, with_regime=True)
    arch = load_strategy_archetype("demo_strat", strategies_root=str(tmp_path))
    assert arch.regime.allowed_sides == ["long"]
    assert arch.regime.allows_side(1) is True
    assert arch.regime.allows_side(-1) is False
    # Regime evaluator independent from prefilter
    passed, reason = arch.regime.evaluate({"tpc_semantic_chop": 0.6})
    assert passed is False
    assert reason is not None and reason.startswith("regime_")
