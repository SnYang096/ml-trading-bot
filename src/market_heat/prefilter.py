"""Strategy prefilter based on market heat scores.

Provides a simple API for the live trading loop to check whether a symbol
is in a favorable heat environment before entering the strategy pipeline.

The prefilter maintains an in-memory cache of heat scores, refreshed
periodically (default: every 6 hours). Weekly EMA moves slowly, so
high-frequency updates are unnecessary.

Integration:
    In run_live.py or the strategy orchestrator, call before decide():

        from src.market_heat.prefilter import HeatPrefilter

        heat_pf = HeatPrefilter()  # loads config from meta.yaml heat_filter section
        ...
        if not heat_pf.is_tradeable(symbol):
            logger.info("Heat prefilter: %s blocked (sector cold)", symbol)
            continue
        intents = strategy.decide(features=features, symbol=symbol)

Config (in strategy meta.yaml or standalone):
    heat_filter:
      enabled: true
      min_symbol_heat: 0.3
      min_sector_heat: 0.2
      min_market_heat: 0.2
      refresh_interval_hours: 6
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .sector_aggregator import MarketHeat

logger = logging.getLogger(__name__)


@dataclass
class HeatFilterConfig:
    enabled: bool = True
    min_symbol_heat: float = 0.3
    min_sector_heat: float = 0.2
    min_market_heat: float = 0.2
    refresh_interval_hours: float = 6.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HeatFilterConfig":
        return cls(
            enabled=bool(d.get("enabled", True)),
            min_symbol_heat=float(d.get("min_symbol_heat", 0.3)),
            min_sector_heat=float(d.get("min_sector_heat", 0.2)),
            min_market_heat=float(d.get("min_market_heat", 0.2)),
            refresh_interval_hours=float(d.get("refresh_interval_hours", 6.0)),
        )


class HeatPrefilter:
    """Singleton-style prefilter that caches heat scores in memory.

    Thread-safe for the typical single-threaded async live loop.
    """

    def __init__(self, config: Optional[HeatFilterConfig] = None):
        self.config = config or HeatFilterConfig()
        self._market_heat: Optional[MarketHeat] = None
        self._registry = None
        self._last_refresh: float = 0.0

    def refresh(self, force: bool = False) -> bool:
        """Refresh heat scores if stale or forced.

        Returns True if refresh happened.
        """
        if not self.config.enabled:
            return False

        now = time.time()
        age_h = (now - self._last_refresh) / 3600
        if (
            not force
            and self._market_heat is not None
            and age_h < self.config.refresh_interval_hours
        ):
            return False

        try:
            from .data_fetcher import fetch_weekly_ohlcv
            from .heat_calculator import compute_heat_batch
            from .sector_aggregator import aggregate
            from .sector_registry import load_sector_registry

            if self._registry is None:
                self._registry = load_sector_registry()

            ohlcv = fetch_weekly_ohlcv(
                self._registry.all_symbols,
                cache_max_age_hours=self.config.refresh_interval_hours,
            )
            symbol_heats = compute_heat_batch(ohlcv)
            self._market_heat = aggregate(symbol_heats, self._registry)
            self._last_refresh = now

            logger.info(
                "Heat prefilter refreshed: market=%s (%.3f), %d symbols",
                self._market_heat.state,
                self._market_heat.score,
                len(symbol_heats),
            )
            return True

        except Exception:
            logger.exception("Heat prefilter refresh failed")
            return False

    def is_tradeable(self, symbol: str) -> bool:
        """Check if a symbol passes heat thresholds.

        Returns True (allow trading) if:
          - heat_filter is disabled
          - no heat data available (fail-open)
          - symbol heat >= min_symbol_heat
            AND sector heat >= min_sector_heat
            AND market heat >= min_market_heat
        """
        if not self.config.enabled:
            return True

        self.refresh()

        if self._market_heat is None:
            return True

        # Market-level check
        if self._market_heat.score < self.config.min_market_heat:
            logger.debug(
                "Heat prefilter: market score %.3f < %.3f, blocking %s",
                self._market_heat.score,
                self.config.min_market_heat,
                symbol,
            )
            return False

        # Symbol-level check
        base = symbol.replace("USDT", "").replace("/USDT:USDT", "")
        hr = self._market_heat.symbol_heats.get(base)
        if hr is not None and hr.score < self.config.min_symbol_heat:
            logger.debug(
                "Heat prefilter: %s score %.3f < %.3f",
                symbol,
                hr.score,
                self.config.min_symbol_heat,
            )
            return False

        # Sector-level check
        if self._registry is not None:
            sector_name = self._registry.sector_for(symbol)
            if sector_name and sector_name in self._market_heat.sector_heats:
                sh = self._market_heat.sector_heats[sector_name]
                if sh.score < self.config.min_sector_heat:
                    logger.debug(
                        "Heat prefilter: sector %s score %.3f < %.3f for %s",
                        sector_name,
                        sh.score,
                        self.config.min_sector_heat,
                        symbol,
                    )
                    return False

        return True

    def get_symbol_heat(self, symbol: str) -> Optional[float]:
        """Get the heat score for a symbol, or None if unavailable."""
        if self._market_heat is None:
            return None
        base = symbol.replace("USDT", "").replace("/USDT:USDT", "")
        hr = self._market_heat.symbol_heats.get(base)
        return hr.score if hr else None

    def get_sector_heat(self, symbol: str) -> Optional[float]:
        """Get the sector heat score for a symbol's sector, or None."""
        if self._market_heat is None or self._registry is None:
            return None
        sector_name = self._registry.sector_for(symbol)
        if sector_name and sector_name in self._market_heat.sector_heats:
            return self._market_heat.sector_heats[sector_name].score
        return None

    def get_market_heat(self) -> Optional[float]:
        """Get the overall market heat score, or None."""
        if self._market_heat is None:
            return None
        return self._market_heat.score

    @property
    def market_state(self) -> Optional[str]:
        if self._market_heat is None:
            return None
        return self._market_heat.state

    def summary(self) -> Dict[str, Any]:
        """Return a dict summary for logging / diagnostics."""
        if self._market_heat is None:
            return {"status": "not_loaded"}
        return {
            "market_score": self._market_heat.score,
            "market_state": self._market_heat.state,
            "sectors": {
                name: {"score": sh.score, "state": sh.state}
                for name, sh in self._market_heat.sector_heats.items()
            },
            "n_symbols": len(self._market_heat.symbol_heats),
            "last_refresh_age_min": round((time.time() - self._last_refresh) / 60, 1),
        }
