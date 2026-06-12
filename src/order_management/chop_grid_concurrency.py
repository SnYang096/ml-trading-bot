"""Cap concurrent multi-leg symbols and enforce strategy-switch cooldown."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BAR_MINUTES = 120  # 2h bar duration for cooldown unit


class MultiLegConcurrencyGate:
    """Shared gate across per-symbol engine instances (chop_grid + trend_scalp).

    Added 2026-06-12: strategy-switch cooldown. After a symbol's engine
    deactivates and another strategy takes over, the previous strategy cannot
    re-activate until ``cooldown_bars × 120 min`` have elapsed.
    """

    def __init__(
        self,
        max_symbols: int,
        *,
        cooldown_bars: int = 0,
    ) -> None:
        self.max_symbols = max(1, int(max_symbols))
        self._cooldown_seconds = max(0, int(cooldown_bars)) * BAR_MINUTES * 60
        self._engines: List[Tuple[str, str, Any]] = []  # (symbol, strategy, engine)
        # (symbol, strategy) → last deactivation wall-clock
        self._last_deactivated: Dict[Tuple[str, str], float] = {}

    def register(
        self, symbol: str, engine: Any, *, strategy: str = "chop_grid"
    ) -> None:
        sym = str(symbol or "").upper().strip()
        if not sym:
            return
        engine._concurrency_gate = self
        self._engines.append((sym, str(strategy).strip().lower(), engine))

    def _engines_for_symbol(
        self, symbol: str, strategy: Optional[str] = None
    ) -> List[Any]:
        sym = str(symbol or "").upper().strip()
        out = []
        for s, st, eng in self._engines:
            if s == sym and (strategy is None or st == str(strategy).strip().lower()):
                out.append(eng)
        return out

    def _any_engine_active(self, symbol: str, strategy: Optional[str] = None) -> bool:
        for eng in self._engines_for_symbol(symbol, strategy=strategy):
            if self._engine_holds_slot(eng):
                return True
        return False

    def _active_strategy(self, symbol: str) -> Optional[str]:
        sym = str(symbol or "").upper().strip()
        for s, st, eng in self._engines:
            if s == sym and self._engine_holds_slot(eng):
                return st
        return None

    def _all_active_symbols(self) -> set[str]:
        self._purge_ghost_segments()
        out: set[str] = set()
        for sym, _st, eng in self._engines:
            if self._engine_holds_slot(eng):
                out.add(sym)
        return out

    def _purge_ghost_segments(self) -> None:
        """Clear stale ``active`` flags before slot accounting."""
        for _sym, _st, engine in self._engines:
            clear = getattr(engine, "clear_stale_active_if_ghost", None)
            if callable(clear):
                try:
                    clear()
                except Exception:
                    logger.warning(
                        "multi-leg concurrency: clear_stale_active_if_ghost raised",
                        exc_info=True,
                    )

    @staticmethod
    def _engine_holds_slot(engine: Any) -> bool:
        """Count a symbol against the cap only if it really occupies a slot."""
        holds = getattr(engine, "holds_real_grid_slot", None)
        if callable(holds):
            try:
                return bool(holds())
            except Exception:
                logger.warning(
                    "multi-leg concurrency: holds_real_grid_slot raised; "
                    "falling back to state.active",
                    exc_info=True,
                )
        state = getattr(engine, "state", None)
        return state is not None and bool(getattr(state, "active", False))

    def notify_deactivation(self, symbol: str, strategy: str) -> None:
        """Record when a strategy engine goes inactive so cooldown can be checked."""
        key = (str(symbol).upper().strip(), str(strategy).strip().lower())
        self._last_deactivated[key] = time.monotonic()

    def _cooldown_remaining(self, symbol: str, strategy: str) -> float:
        """Seconds until this strategy is allowed to re-activate on the symbol, or 0."""
        if self._cooldown_seconds <= 0:
            return 0.0
        sym = str(symbol).upper().strip()
        strat = str(strategy).strip().lower()
        active = self._active_strategy(sym)
        if active is not None and active != strat:
            # Another strategy currently holds the symbol. Check if this one
            # was recently deactivated (cooldown applies to the one giving way).
            key = (sym, strat)
            last = self._last_deactivated.get(key)
            if last is not None:
                elapsed = time.monotonic() - last
                if elapsed < self._cooldown_seconds:
                    return self._cooldown_seconds - elapsed
        return 0.0

    def allow_new_segment(self, symbol: str, *, strategy: str = "chop_grid") -> bool:
        sym = str(symbol or "").upper().strip()
        strat = str(strategy).strip().lower()

        # Cooldown check: if this strategy was recently active and then
        # deactivated (other strategy took over), enforce minimum gap.
        cooldown = self._cooldown_remaining(sym, strat)
        if cooldown > 0:
            bars_left = int(cooldown / (BAR_MINUTES * 60)) + 1
            logger.info(
                "multi-leg cooldown: block %s/%s activation (%.0fs ≈ %d bars remaining)",
                sym,
                strat,
                cooldown,
                bars_left,
            )
            return False

        # Concurrent symbol cap (shared across both strategies).
        active = self._all_active_symbols()
        if sym in active:
            return True
        if len(active) >= self.max_symbols:
            logger.info(
                "multi-leg concurrent symbol cap: reject %s/%s " "(active=%s cap=%d)",
                sym,
                strat,
                sorted(active),
                self.max_symbols,
            )
            return False
        return True
