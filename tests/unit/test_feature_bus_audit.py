"""Tests for feature-bus publish audit."""

from __future__ import annotations

import logging

import pytest

from src.live_data_stream.feature_bus_audit import (
    FeatureBusAuditError,
    audit_published_features,
)
from tests.unit.test_feature_health_report import _make_ifc


class TestFeatureBusAudit:
    def test_audit_logs_critical_to_audit_logger(self, monkeypatch, caplog):
        monkeypatch.setenv("MLBOT_FEATURE_BUS_AUDIT", "1")
        monkeypatch.delenv("MLBOT_FEATURE_BUS_AUDIT_STRICT", raising=False)
        ifc = _make_ifc(["atr", "oi_zscore", "close"])
        features = {"close": 1.0}
        audit_log = logging.getLogger("mlbot.feature_bus.audit")
        with caplog.at_level(logging.ERROR, logger=audit_log.name):
            report = audit_published_features(
                features=features,
                symbol="BTCUSDT",
                timeframe="120T",
                feature_computer=ifc,
                update_prometheus=False,
            )
        assert "oi_zscore" in report["critical_nan"]
        assert any("feature_publish_audit" in r.message for r in caplog.records)

    def test_strict_raises_on_critical_nan(self, monkeypatch):
        monkeypatch.setenv("MLBOT_FEATURE_BUS_AUDIT", "1")
        monkeypatch.setenv("MLBOT_FEATURE_BUS_AUDIT_STRICT", "1")
        ifc = _make_ifc(["oi_zscore"])
        with pytest.raises(FeatureBusAuditError, match="critical"):
            audit_published_features(
                features={},
                symbol="BTCUSDT",
                timeframe="120T",
                feature_computer=ifc,
                update_prometheus=False,
            )

    def test_nan_value_counts_as_missing(self):
        ifc = _make_ifc(["close", "oi_zscore"])
        missing = ifc._missing_or_nan_features(
            {"close": 1.0, "oi_zscore": float("nan")}
        )
        assert missing == ["oi_zscore"]
