"""Parity tests: src.research kernels vs optimize_gate_unified re-exports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import optimize_gate_unified as legacy
from src.research.stat_kernels.gate_lift import (
    compute_lift_for_threshold,
    scan_thresholds_for_lift,
)
from src.research.stat_kernels.plateau import find_stable_lift_plateau
from src.research.stat_kernels.robustness import (
    UnifiedOptimizationConfig,
    compute_robustness_score,
)
from src.research.stat_kernels.stratify import compute_stratification


def _synthetic_gate_df(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    feat = rng.normal(size=n)
    is_good = (feat > 0).astype(int)
    # flip some labels for mixed pass rates
    flip = rng.random(n) < 0.25
    is_good = np.where(flip, 1 - is_good, is_good)
    return pd.DataFrame(
        {
            "pulse_z": feat,
            "is_good": is_good,
            "forward_rr": rng.normal(0.2, 1.0, size=n),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        }
    )


def test_lift_parity_with_legacy_module():
    df = _synthetic_gate_df()
    for op, th in [("lt", 0.0), ("gt", 0.5), ("le", -0.2)]:
        new = compute_lift_for_threshold(df, "pulse_z", op, th)
        old = legacy.compute_lift_for_threshold(df, "pulse_z", op, th)
        for key in ("lift", "pass_rate_good", "pass_rate_bad", "pass_rate_all"):
            if np.isnan(new[key]) and np.isnan(old[key]):
                continue
            assert new[key] == pytest.approx(old[key], rel=1e-9, abs=1e-9)


def test_scan_thresholds_midpoint_parity():
    df = _synthetic_gate_df(n=500)
    cfg = UnifiedOptimizationConfig(
        min_samples_good=20, min_samples_bad=20, min_lift=0.5
    )
    scan = scan_thresholds_for_lift(df, "pulse_z", "lt", (0.0, 1.0), 0.1)
    legacy_scan = legacy.scan_thresholds_for_lift(df, "pulse_z", "lt", (0.0, 1.0), 0.1)
    assert len(scan) == len(legacy_scan)
    new_plateau = find_stable_lift_plateau(scan, cfg, actual_step=0.1)
    old_plateau = legacy.find_stable_lift_plateau(legacy_scan, cfg, actual_step=0.1)
    if new_plateau is None:
        assert old_plateau is None
    else:
        assert old_plateau is not None
        assert new_plateau["recommended_threshold"] == pytest.approx(
            old_plateau["recommended_threshold"], rel=1e-9
        )
        assert new_plateau["plateau_mid"] == pytest.approx(
            old_plateau["plateau_mid"], rel=1e-9
        )


def test_robustness_parity():
    df = _synthetic_gate_df(n=600)
    cfg = UnifiedOptimizationConfig(min_samples_good=20, min_samples_bad=20)
    new = compute_robustness_score(df, "pulse_z", "lt", 0.0, config=cfg)
    old = legacy.compute_robustness_score(df, "pulse_z", "lt", 0.0, config=cfg)
    assert new.overall_score == pytest.approx(old.overall_score, rel=1e-9)
    assert new.param_stability == pytest.approx(old.param_stability, rel=1e-9)


def test_stratification_kernel():
    df = _synthetic_gate_df(n=200)
    row = compute_stratification(
        df, "pulse_z", 0.0, "high", "forward_rr", "is_good", min_samples=20
    )
    assert row is not None
    assert row["n_signal"] + row["n_rest"] <= len(df)
    assert 0.0 <= row["bad_rate_signal"] <= 1.0


def test_rr_simulate_import():
    from src.research.execution_kernel.rr_simulate import simulate_rr_execution

    assert simulate_rr_execution is not None
