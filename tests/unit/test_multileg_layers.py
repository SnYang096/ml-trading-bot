from scripts.pipeline.multileg_layers import (
    candidate_for_enabled_layers,
    resolve_multileg_layer_settings,
    score_candidate_with_constraints,
)


def test_resolve_multileg_layer_settings_respects_strategy_flags() -> None:
    settings = resolve_multileg_layer_settings(
        strategy_type="grid",
        strategy_cfg={
            "has_prefilter": True,
            "has_gate": False,
            "has_entry_filter": False,
        },
        threshold_cfg={
            "prefilter": {"optimize": True},
            "gate": {"optimize": True},
            "entry_filter": {"optimize": True},
            "execution_opt": {"enabled": True},
        },
        default_prefilter_optimize=False,
        default_gate_optimize=False,
        default_entry_filter_optimize=False,
        default_execution_optimize=False,
    )
    assert settings.prefilter_optimize is True
    assert settings.gate_optimize is False
    assert settings.entry_filter_optimize is False
    assert settings.execution_optimize is True


def test_candidate_for_enabled_layers_filters_keys() -> None:
    settings = resolve_multileg_layer_settings(
        strategy_type="grid",
        strategy_cfg={"has_prefilter": True},
        threshold_cfg={
            "prefilter": {"optimize": True},
            "execution_opt": {"enabled": False},
        },
        default_prefilter_optimize=True,
        default_gate_optimize=False,
        default_entry_filter_optimize=False,
        default_execution_optimize=False,
    )
    tuned = candidate_for_enabled_layers(
        strategy_type="grid",
        candidate={"entry_chop_min": 0.4, "atr_mult": 0.7, "min_pct": 0.003},
        settings=settings,
    )
    assert "entry_chop_min" in tuned
    assert "atr_mult" not in tuned
    assert "min_pct" not in tuned


def test_score_candidate_with_constraints_penalizes_rule_violations() -> None:
    base = score_candidate_with_constraints(
        metrics={
            "total_r": 0.2,
            "worst_segment": -0.01,
            "forced_rate": 0.01,
            "n_trades": 200,
        },
        kpi_backtest={"min_trades": 100, "max_forced_exit_rate": 0.35},
    )
    bad = score_candidate_with_constraints(
        metrics={
            "total_r": 0.4,
            "worst_segment": -0.01,
            "forced_rate": 0.5,
            "n_trades": 20,
        },
        kpi_backtest={"min_trades": 100, "max_forced_exit_rate": 0.35},
    )
    assert bad < base
