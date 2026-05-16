"""Periodic REST backfill for multi-leg order rows."""

from __future__ import annotations

import asyncio
import logging
import os
from functools import partial
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_SECONDS = 60.0


def multi_leg_backfill_interval_seconds() -> float:
    """Return configured interval, or ``0`` when disabled."""
    unset = "MLBOT_MULTI_LEG_ORDER_BACKFILL_INTERVAL_SECONDS" not in os.environ
    raw = (
        os.getenv("MLBOT_MULTI_LEG_ORDER_BACKFILL_INTERVAL_SECONDS", "").strip().lower()
    )
    if raw in {"0", "false", "no", "off", "disable", "disabled"}:
        return 0.0
    if raw == "":
        return _DEFAULT_INTERVAL_SECONDS if unset else 0.0
    try:
        val = float(raw)
        return val if val > 0 else 0.0
    except ValueError:
        logger.warning(
            "MLBOT_MULTI_LEG_ORDER_BACKFILL_INTERVAL_SECONDS invalid %r -> default %.0f",
            raw,
            _DEFAULT_INTERVAL_SECONDS,
        )
        return _DEFAULT_INTERVAL_SECONDS


def multi_leg_backfill_enabled(api: Any, storage: Any) -> bool:
    return (
        api is not None
        and storage is not None
        and callable(getattr(storage, "get_recent_orders_for_backfill", None))
        and callable(getattr(storage, "apply_execution_report", None))
        and callable(getattr(api, "get_order", None))
    )


def normalize_rest_order_status(status: Any) -> str:
    """Map REST/ccxt order statuses into storage/user-stream vocabulary."""
    raw = str(status or "").strip().lower()
    if raw in {"closed", "filled"}:
        return "filled"
    if raw in {"canceled", "cancelled"}:
        return "canceled"
    if raw in {"expired", "rejected", "open", "new", "pending", "partially_filled"}:
        return raw
    return raw or "unknown"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("%s invalid %r -> default %d", name, raw, default)
        return int(default)


def run_multi_leg_backfill_once(
    *,
    api: Any,
    storage: Any,
    lookback_hours: int,
    limit: int,
) -> int:
    """Run one REST backfill pass; return updated row count."""
    updated_rows = 0
    candidates = storage.get_recent_orders_for_backfill(
        lookback_hours=max(1, int(lookback_hours)),
        limit=max(1, int(limit)),
    )
    for row in candidates:
        ex_id = str(row.get("exchange_order_id") or "").strip()
        symbol = str(row.get("symbol") or "").strip().upper()
        if not ex_id or not symbol:
            continue
        try:
            snap = api.get_order(ex_id, symbol)
            if not snap:
                continue
            event_time = snap.get("update_time") or snap.get("timestamp")
            payload = {
                "run_id": row.get("run_id"),
                "strategy": row.get("strategy"),
                "symbol": symbol,
                "order_id": snap.get("order_id") or ex_id,
                "client_order_id": snap.get("client_order_id")
                or row.get("client_order_id"),
                "status": normalize_rest_order_status(snap.get("status")),
                "filled_qty": snap.get("filled"),
                "avg_price": snap.get("average_price"),
                "event_time": event_time,
                "trade_time": event_time,
                "reject_reason": snap.get("reject_reason"),
                "error_message": snap.get("error_message"),
                "raw": snap,
            }
            changed = int(storage.apply_execution_report(payload) or 0)
            updated_rows += max(0, changed)
        except Exception:
            logger.debug(
                "multi-leg REST backfill skipped row local=%s exchange=%s symbol=%s",
                row.get("local_order_id"),
                ex_id,
                symbol,
                exc_info=True,
            )
    return updated_rows


async def periodic_multi_leg_order_backfill(
    *,
    api: Any,
    storage: Any,
    startup_delay_seconds: float = 20.0,
) -> None:
    """Periodic task to refresh multi-leg order status from REST order snapshots."""
    interval = multi_leg_backfill_interval_seconds()
    if interval <= 0:
        return
    if not multi_leg_backfill_enabled(api, storage):
        logger.info(
            "multi-leg REST backfill: disabled (missing api/storage capabilities)"
        )
        return

    lookback_hours = _env_int("MLBOT_MULTI_LEG_ORDER_BACKFILL_LOOKBACK_HOURS", 168)
    limit = _env_int("MLBOT_MULTI_LEG_ORDER_BACKFILL_LIMIT", 200)
    logger.info(
        "multi-leg REST backfill: every %.0fs lookback_hours=%d limit=%d",
        interval,
        max(1, lookback_hours),
        max(1, limit),
    )
    if startup_delay_seconds > 0:
        await asyncio.sleep(startup_delay_seconds)

    loop = asyncio.get_running_loop()
    while True:
        try:
            updated = await loop.run_in_executor(
                None,
                partial(
                    run_multi_leg_backfill_once,
                    api=api,
                    storage=storage,
                    lookback_hours=max(1, lookback_hours),
                    limit=max(1, limit),
                ),
            )
            if updated > 0:
                logger.info("multi-leg REST backfill: updated %d row(s)", updated)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.warning("multi-leg REST backfill iteration failed", exc_info=True)
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break
