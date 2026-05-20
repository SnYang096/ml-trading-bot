from __future__ import annotations

import pytest

from src.time_series_model.live.metrics_exporter import (
    METRICS,
    Metrics,
    _PROM_AVAILABLE,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("trend", "trend"),
        ("", "trend"),
        ("hedge", "hedge"),
        ("multi_leg", "hedge"),
        ("multi-leg", "hedge"),
        ("spot", "spot"),
        ("spot_accum", "spot"),
        ("spot-accum", "spot"),
        ("SPOT", "spot"),
    ],
)
def test_normalize_scope(raw: str, expected: str) -> None:
    assert Metrics._normalize_scope(raw) == expected


@pytest.mark.skipif(not _PROM_AVAILABLE, reason="prometheus_client not installed")
def test_update_reconciliation_metrics_uses_spot_scope_label() -> None:
    from prometheus_client import REGISTRY

    METRICS.update_reconciliation_metrics(
        scope="spot_accum",
        strategy="accum",
        symbol="ALL",
        ok=False,
        issue_counts={"api_error": 1},
        ts_seconds=1_700_000_000.0,
    )
    samples = []
    for metric in REGISTRY.collect():
        if metric.name != "mlbot_reconciliation_ok":
            continue
        for sample in metric.samples:
            if sample.labels.get("scope") == "spot":
                samples.append(sample)
    assert samples, "expected mlbot_reconciliation_ok with scope=spot"
    assert any(s.value == 0.0 for s in samples)
