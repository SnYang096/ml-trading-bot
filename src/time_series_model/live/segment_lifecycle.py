"""Shared segment lifecycle helpers for multi-leg live engines."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class SegmentState(str, Enum):
    IDLE = "idle"
    ENTERING = "entering"
    ACTIVE = "active"
    CLOSING = "closing"
    # Reserved for audit trails; transitions today go CLOSING → IDLE via _deactivate().
    CLOSED = "closed"


def segment_occupies_slot(segment_state: str) -> bool:
    return segment_state in {
        SegmentState.ENTERING.value,
        SegmentState.ACTIVE.value,
        SegmentState.CLOSING.value,
    }


def segment_allows_new_entry(segment_state: str) -> bool:
    return segment_state == SegmentState.IDLE.value


def migrate_segment_state_from_legacy(
    *,
    active: bool,
    segment_state_raw: str | None,
) -> str:
    if segment_state_raw:
        return str(segment_state_raw)
    if not active:
        return SegmentState.IDLE.value
    # active=True without persisted segment_state: treat as ACTIVE; ghost segments
    # (active with empty local state) are cleared by the concurrency gate.
    return SegmentState.ACTIVE.value


class _SegmentLifecycleState(Protocol):
    symbol: str
    active: bool
    segment_state: str
    pending_orders: list[Any]
    inventory: list[Any]
    current_regime: str


class SegmentLifecycleMixin:
    """Common deactivate / ghost / slot logic for chop_grid and trend_scalp."""

    _engine_name: str = ""
    _exchange_open_orders: bool = False

    state: _SegmentLifecycleState

    def save_state(self) -> None:  # noqa: E704
        ...

    def _log_stale_active_reset(self) -> None:
        raise NotImplementedError

    def _reconcile_legacy_active_flag(self) -> None:
        if self.state.active and self.state.segment_state == SegmentState.IDLE.value:
            self.state.segment_state = SegmentState.ACTIVE.value

    def _sync_active_from_segment_state(self) -> None:
        self.state.active = segment_occupies_slot(self.state.segment_state)

    def _needs_late_fill_cleanup(self) -> bool:
        """True when segment is winding down (trend: skip promote/protection on late fills)."""
        return (
            not self.state.active
            or self.state.segment_state == SegmentState.CLOSING.value
        )

    def _deactivate(self, reason: str) -> None:
        logger.info(
            "%s deactivate: symbol=%s reason=%s",
            self._engine_name,
            self.state.symbol,
            reason,
        )
        self.state.segment_state = SegmentState.IDLE.value
        self.state.active = False
        if hasattr(self.state, "current_regime"):
            self.state.current_regime = "idle"
        self.save_state()
        gate = getattr(self, "_concurrency_gate", None)
        if gate is not None:
            gate.notify_deactivation(self.state.symbol, self._engine_name)

    def _enter_segment(self) -> None:
        self.state.segment_state = SegmentState.ENTERING.value
        self.state.active = True

    def _promote_to_active(self) -> None:
        if (
            self.state.segment_state == SegmentState.ENTERING.value
            and self.state.inventory
        ):
            self.state.segment_state = SegmentState.ACTIVE.value
            self.state.active = True

    def _begin_closing(self, reason: str) -> None:
        logger.info(
            "%s begin_closing: symbol=%s reason=%s",
            self._engine_name,
            self.state.symbol,
            reason,
        )
        self.state.segment_state = SegmentState.CLOSING.value
        self.state.active = True

    def _maybe_deactivate_if_fully_closed(self) -> None:
        if self._exchange_has_open_activity():
            return
        if (
            self.state.active
            and not self.state.inventory
            and not self.state.pending_orders
        ):
            self._deactivate("fully_closed")

    def is_stale_active_ghost(self) -> bool:
        if self._exchange_has_open_activity():
            return False
        return bool(
            self.state.active
            and not self.state.pending_orders
            and not self.state.inventory
        )

    def clear_stale_active_if_ghost(self) -> bool:
        if not self.is_stale_active_ghost():
            return False
        self._log_stale_active_reset()
        self._deactivate("ghost_cleared")
        return True

    def holds_real_grid_slot(self) -> bool:
        if not segment_occupies_slot(
            getattr(self.state, "segment_state", SegmentState.IDLE.value)
        ):
            return False
        if not bool(getattr(self.state, "active", False)):
            return False
        return bool(self.state.pending_orders or self.state.inventory)

    def _exchange_has_open_activity(self) -> bool:
        return bool(self._exchange_open_orders)
