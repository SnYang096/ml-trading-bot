"""Cap how many symbols may run an active chop_grid segment at once."""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


class ChopGridConcurrencyGate:
    """Shared gate across per-symbol ``ChopGridLiveEngine`` instances."""

    def __init__(self, max_symbols: int) -> None:
        self.max_symbols = max(1, int(max_symbols))
        self._engines: List[Tuple[str, Any]] = []

    def register(self, symbol: str, engine: Any) -> None:
        sym = str(symbol or "").upper().strip()
        if not sym:
            return
        engine._concurrency_gate = self  # noqa: SLF001 — live engine hook
        self._engines.append((sym, engine))

    @staticmethod
    def _engine_holds_slot(engine: Any) -> bool:
        """Count a symbol against the cap only if it really occupies a slot.

        Ghost-active engines (active=True but no pending/inventory/exchange
        activity) are excluded so they cannot permanently block other symbols.
        """
        holds = getattr(engine, "holds_real_grid_slot", None)
        if callable(holds):
            try:
                return bool(holds())
            except Exception:  # pragma: no cover - defensive
                logger.warning(
                    "chop_grid concurrency: holds_real_grid_slot raised; "
                    "falling back to state.active",
                    exc_info=True,
                )
        state = getattr(engine, "state", None)
        return state is not None and bool(getattr(state, "active", False))

    def _purge_ghost_segments(self) -> None:
        """Clear stale ``active`` flags before slot accounting (any symbol's bar)."""
        for _sym, engine in self._engines:
            clear = getattr(engine, "clear_stale_active_if_ghost", None)
            if callable(clear):
                try:
                    clear()
                except Exception:  # pragma: no cover - defensive
                    logger.warning(
                        "chop_grid concurrency: clear_stale_active_if_ghost raised",
                        exc_info=True,
                    )

    def _active_symbols(self) -> set[str]:
        self._purge_ghost_segments()
        out: set[str] = set()
        for sym, engine in self._engines:
            if self._engine_holds_slot(engine):
                out.add(sym)
        return out

    def allow_new_segment(self, symbol: str) -> bool:
        sym = str(symbol or "").upper().strip()
        active = self._active_symbols()
        if sym in active:
            return True
        if len(active) >= self.max_symbols:
            logger.info(
                "chop_grid concurrent symbol cap: reject new segment on %s "
                "(active=%s cap=%d)",
                sym,
                sorted(active),
                self.max_symbols,
            )
            return False
        return True
