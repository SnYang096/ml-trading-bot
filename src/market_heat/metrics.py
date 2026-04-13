"""Prometheus gauge definitions for market heat scores.

Follows the same pattern as src/time_series_model/live/metrics_exporter.py:
prometheus_client is an optional dependency; falls back to no-ops if unavailable.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_PROM_AVAILABLE = False
try:
    from prometheus_client import Gauge

    _PROM_AVAILABLE = True
except ImportError:
    pass


class _NoopGauge:
    def set(self, *a, **kw):
        pass

    def labels(self, *a, **kw):
        return self


_NOOP = _NoopGauge()


class HeatMetrics:
    """All heat-related Prometheus gauges."""

    def __init__(self) -> None:
        if not _PROM_AVAILABLE:
            self.symbol_score = _NOOP
            self.symbol_state = _NOOP
            self.symbol_ema_slope = _NOOP
            self.symbol_ema_distance = _NOOP
            self.sector_score = _NOOP
            self.market_score = _NOOP
            return

        self.symbol_score = Gauge(
            "mlbot_heat_score",
            "Symbol heat score (0.0 ~ 1.0)",
            ["symbol", "sector"],
        )
        self.symbol_state = Gauge(
            "mlbot_heat_state",
            "Symbol heat state: 0=COLD, 1=WARM, 2=HOT",
            ["symbol", "sector"],
        )
        self.symbol_ema_slope = Gauge(
            "mlbot_heat_ema_slope",
            "Weekly EMA50 slope (4-week normalized change)",
            ["symbol", "sector"],
        )
        self.symbol_ema_distance = Gauge(
            "mlbot_heat_ema_distance",
            "Price distance from weekly EMA50 (fraction)",
            ["symbol", "sector"],
        )
        self.sector_score = Gauge(
            "mlbot_heat_sector_score",
            "Aggregated sector heat score (0.0 ~ 1.0)",
            ["sector"],
        )
        self.market_score = Gauge(
            "mlbot_heat_market_score",
            "Overall market heat score (0.0 ~ 1.0)",
            ["market"],
        )


HEAT_METRICS = HeatMetrics()

_STATE_MAP = {"COLD": 0, "WARM": 1, "HOT": 2}


def export_heat_to_prometheus(market_heat, registry) -> None:
    """Push latest heat scores into Prometheus gauges.

    Args:
        market_heat: MarketHeat instance from sector_aggregator.aggregate().
        registry: SectorRegistry for symbol->sector lookup.
    """
    if not _PROM_AVAILABLE:
        return

    for sym, hr in market_heat.symbol_heats.items():
        sector = registry.sector_for(sym) or "unknown"
        HEAT_METRICS.symbol_score.labels(symbol=sym, sector=sector).set(hr.score)
        HEAT_METRICS.symbol_state.labels(symbol=sym, sector=sector).set(
            _STATE_MAP.get(hr.state, 0)
        )
        HEAT_METRICS.symbol_ema_slope.labels(symbol=sym, sector=sector).set(
            hr.ema_slope
        )
        HEAT_METRICS.symbol_ema_distance.labels(symbol=sym, sector=sector).set(
            hr.ema_distance
        )

    for name, sh in market_heat.sector_heats.items():
        HEAT_METRICS.sector_score.labels(sector=name).set(sh.score)

    HEAT_METRICS.market_score.labels(market="crypto").set(market_heat.score)
    logger.info(
        "Exported heat metrics: market=%.3f, %d sectors, %d symbols",
        market_heat.score,
        len(market_heat.sector_heats),
        len(market_heat.symbol_heats),
    )
