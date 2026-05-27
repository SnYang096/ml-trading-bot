"""Parity: snotio_calc kernel wired into entry plateau script."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.optimize_entry_filter_plateau import (
    _find_plateau,
    _width_to_confidence,
)
from src.research.stat_kernels.snotio_calc import (
    compute_snotio,
    find_snotio_plateau,
    width_to_confidence,
)


def _synthetic_scan(n: int = 12) -> list[dict]:
    rows = []
    for i in range(n):
        th = round(0.1 * i, 2)
        rows.append(
            {
                "threshold": th,
                "snotio": 0.15 + 0.01 * (i % 5),
                "trades": 80 + i * 3,
                "too_few": i < 1,
            }
        )
    return rows


def test_entry_script_uses_shared_kernel():
    assert _find_plateau is find_snotio_plateau
    assert _width_to_confidence is width_to_confidence


def test_width_to_confidence_tiers():
    assert width_to_confidence(0.35) == "HIGH"
    assert width_to_confidence(0.2) == "MEDIUM"
    assert width_to_confidence(0.05) == "LOW"


def test_compute_snotio_mean():
    assert compute_snotio(pd.Series([1.0, 2.0, 3.0])) == pytest.approx(2.0)
    assert compute_snotio(pd.Series([])) == 0.0


def test_find_snotio_plateau_stable_window():
    results = _synthetic_scan()
    for operator in (">=", "<="):
        out = find_snotio_plateau(results, operator=operator, window=5)
        if out.get("is_plateau"):
            assert "recommended" in out
            assert out["confidence"] in ("HIGH", "MEDIUM", "LOW")


def test_find_snotio_plateau_fallback_best_single():
    results = [
        {"threshold": 0.1, "snotio": 0.05, "trades": 10, "too_few": False},
        {"threshold": 0.2, "snotio": 0.25, "trades": 12, "too_few": False},
        {"threshold": 0.3, "snotio": 0.04, "trades": 11, "too_few": False},
        {"threshold": 0.4, "snotio": 0.22, "trades": 13, "too_few": False},
        {"threshold": 0.5, "snotio": 0.03, "trades": 14, "too_few": False},
    ]
    out = find_snotio_plateau(results, window=5)
    assert out["is_plateau"] is False
    assert "best_single" in out
    assert out["best_single"]["threshold"] == 0.2
