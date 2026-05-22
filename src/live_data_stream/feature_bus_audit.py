"""Feature-bus publish audit: log NaN/missing batches and optionally fail hard."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from src.time_series_model.live.incremental_feature_computer import (
        IncrementalFeatureComputer,
    )

_audit = logging.getLogger("mlbot.feature_bus.audit")
_logger = logging.getLogger(__name__)


class FeatureBusAuditError(RuntimeError):
    """Raised when published features violate bus audit thresholds."""


def _truthy(name: str, default: str = "1") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw not in ("0", "false", "off", "no")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def bus_audit_enabled() -> bool:
    return _truthy("MLBOT_FEATURE_BUS_AUDIT", "1")


def audit_published_features(
    *,
    features: Dict[str, Any],
    symbol: str,
    timeframe: str,
    feature_computer: IncrementalFeatureComputer,
    update_prometheus: bool = True,
    context: str = "publish",
) -> Dict[str, Any]:
    """Run health report, write audit line, ERROR/raise on large NaN batches."""
    if not bus_audit_enabled():
        return {}

    report = feature_computer.report_feature_health(
        features,
        symbol=symbol,
        timeframe=timeframe,
        update_prometheus=update_prometheus,
    )
    report["context"] = context

    payload = {
        "event": "feature_publish_audit",
        "context": context,
        "symbol": symbol,
        "timeframe": timeframe,
        "expected": report.get("expected"),
        "total": report.get("total"),
        "missing_count": report.get("missing_count"),
        "nan_ratio": report.get("nan_ratio"),
        "critical_nan": report.get("critical_nan"),
        "missing_sample": report.get("missing_names"),
    }
    critical = list(report.get("critical_nan") or [])
    nan_ratio = float(report.get("nan_ratio") or 0.0)
    error_ratio = _float_env("MLBOT_FEATURE_BUS_AUDIT_NAN_ERROR_RATIO", 0.2)
    fatal_ratio = _float_env("MLBOT_FEATURE_BUS_AUDIT_NAN_FATAL_RATIO", 0.5)

    if critical or nan_ratio >= fatal_ratio:
        _audit.error("%s", json.dumps(payload, ensure_ascii=False, default=str))
    elif nan_ratio >= error_ratio:
        _audit.warning("%s", json.dumps(payload, ensure_ascii=False, default=str))
    else:
        _audit.info("%s", json.dumps(payload, ensure_ascii=False, default=str))

    _enforce_thresholds(report, symbol=symbol, timeframe=timeframe, context=context)
    return report


def should_skip_feature_bus_publish(report: Dict[str, Any]) -> bool:
    """Block writing poisoned snapshots to the shared bus (default on)."""
    if not _truthy("MLBOT_FEATURE_BUS_SKIP_PUBLISH_ON_BAD_HEALTH", "1"):
        return False
    critical = list(report.get("critical_nan") or [])
    nan_ratio = float(report.get("nan_ratio") or 0.0)
    fatal_ratio = _float_env("MLBOT_FEATURE_BUS_AUDIT_NAN_FATAL_RATIO", 0.5)
    return bool(critical) or nan_ratio >= fatal_ratio


def _enforce_thresholds(
    report: Dict[str, Any],
    *,
    symbol: str,
    timeframe: str,
    context: str,
) -> None:
    if not _truthy("MLBOT_FEATURE_BUS_AUDIT_STRICT", "0"):
        return

    critical = list(report.get("critical_nan") or [])
    nan_ratio = float(report.get("nan_ratio") or 0.0)
    fatal_ratio = _float_env("MLBOT_FEATURE_BUS_AUDIT_NAN_FATAL_RATIO", 0.5)
    fail_critical = _truthy("MLBOT_FEATURE_BUS_AUDIT_FAIL_ON_CRITICAL", "1")

    if fail_critical and critical:
        msg = (
            f"feature-bus audit STRICT: critical NaN {critical} "
            f"symbol={symbol} tf={timeframe} context={context}"
        )
        _logger.error(msg)
        raise FeatureBusAuditError(msg)

    if nan_ratio >= fatal_ratio:
        msg = (
            f"feature-bus audit STRICT: nan_ratio={nan_ratio:.1%} >= {fatal_ratio:.1%} "
            f"symbol={symbol} tf={timeframe} context={context} "
            f"missing={report.get('missing_count')}/{report.get('expected')}"
        )
        _logger.error(msg)
        raise FeatureBusAuditError(msg)
