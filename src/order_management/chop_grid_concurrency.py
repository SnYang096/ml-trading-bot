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

    def _active_symbols(self) -> set[str]:
        out: set[str] = set()
        for sym, engine in self._engines:
            state = getattr(engine, "state", None)
            if state is not None and bool(getattr(state, "active", False)):
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
