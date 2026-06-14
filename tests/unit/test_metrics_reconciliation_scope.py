from __future__ import annotations

import pytest

from src.order_management.execution_truth_sync import (
    RECONCILIATION_ISSUE_BUCKETS,
    reconciliation_ok_from_issues,
)
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


def test_reconciliation_issue_buckets_include_open_reconcile_updated() -> None:
    assert "open_reconcile_updated" in RECONCILIATION_ISSUE_BUCKETS


def test_reconciliation_ok_ignores_self_healing_open_reconcile_updated() -> None:
    assert reconciliation_ok_from_issues(
        {
            "open_reconcile_updated": 3,
            "stale_local_order": 0,
            "api_error": 0,
        }
    )
    assert not reconciliation_ok_from_issues(
        {
            "open_reconcile_updated": 2,
            "stale_local_order": 1,
        }
    )


@pytest.mark.skipif(not _PROM_AVAILABLE, reason="prometheus_client not installed")
def test_update_reconciliation_metrics_writes_open_reconcile_updated_gauge() -> None:
    from prometheus_client import REGISTRY

    METRICS.update_reconciliation_metrics(
        scope="trend",
        strategy="all",
        symbol="ALL",
        ok=True,
        issue_counts={"open_reconcile_updated": 3},
        ts_seconds=1_700_000_001.0,
    )
    samples = []
    for metric in REGISTRY.collect():
        if metric.name != "mlbot_reconciliation_issue_count":
            continue
        for sample in metric.samples:
            if (
                sample.labels.get("scope") == "trend"
                and sample.labels.get("issue") == "open_reconcile_updated"
            ):
                samples.append(sample)
    assert samples, "expected open_reconcile_updated gauge for trend scope"
    assert any(s.value == 3.0 for s in samples)


@pytest.mark.skipif(not _PROM_AVAILABLE, reason="prometheus_client not installed")
def test_update_reconciliation_metrics_writes_all_issue_buckets() -> None:
    from prometheus_client import REGISTRY

    METRICS.update_reconciliation_metrics(
        scope="hedge",
        strategy="all",
        symbol="ALL",
        ok=True,
        issue_counts={"stale_local_order": 2},
        ts_seconds=1_700_000_002.0,
    )
    issues_seen: set[str] = set()
    for metric in REGISTRY.collect():
        if metric.name != "mlbot_reconciliation_issue_count":
            continue
        for sample in metric.samples:
            if (
                sample.labels.get("scope") == "hedge"
                and sample.labels.get("strategy") == "all"
                and sample.labels.get("symbol") == "ALL"
            ):
                issues_seen.add(str(sample.labels.get("issue")))
    assert issues_seen >= set(RECONCILIATION_ISSUE_BUCKETS)
