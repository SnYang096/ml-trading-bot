"""Periodic REST backfill for multi-leg order rows."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from functools import partial
from typing import Any

from src.time_series_model.live.metrics_exporter import METRICS

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


def _parse_utc_ts(raw: Any) -> float | None:
    txt = str(raw or "").strip()
    if not txt:
        return None
    try:
        return datetime.fromisoformat(txt.replace("Z", "+00:00")).timestamp()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return (
                datetime.strptime(txt, fmt).replace(tzinfo=timezone.utc).timestamp()
            )
        except ValueError:
            continue
    return None


def _fetch_open_order_rows(api: Any, symbol: str) -> list[dict[str, Any]] | None:
    """Return open orders for symbol, or None when the snapshot fetch failed."""
    fetch = getattr(api, "get_open_orders_for_sl_cleanup", None)
    try:
        if callable(fetch):
            return list(fetch(symbol) or [])
        return list(api.get_open_orders(symbol) or [])
    except Exception:
        return None


def _resolve_rest_order_snapshot(
    api: Any,
    row: dict[str, Any],
    *,
    symbol: str,
    exchange_order_id: str,
) -> tuple[str, dict[str, Any] | None]:
    """
    Return (kind, snapshot).

    kind:
      - ``ok``: snapshot dict to apply
      - ``missing``: REST + client id both empty (candidate for stale expire)
      - ``error``: API failure; do not mark expired
    """
    client_order_id = str(row.get("client_order_id") or "").strip()
    try:
        try:
            snap = api.get_order(
                exchange_order_id,
                symbol,
                client_order_id=client_order_id or None,
            )
        except TypeError:
            snap = api.get_order(exchange_order_id, symbol)
    except Exception:
        logger.debug(
            "multi-leg REST backfill: get_order failed symbol=%s exchange=%s",
            symbol,
            exchange_order_id,
            exc_info=True,
        )
        return "error", None
    if snap:
        return "ok", snap
    if client_order_id:
        fetch_cid = getattr(api, "get_order_by_client_id", None)
        if callable(fetch_cid):
            try:
                snap = fetch_cid(client_order_id, symbol)
            except Exception:
                logger.debug(
                    "multi-leg REST backfill: get_order_by_client_id failed "
                    "symbol=%s client=%s",
                    symbol,
                    client_order_id,
                    exc_info=True,
                )
                return "error", None
            if snap:
                return "ok", snap
    return "missing", None


def _is_stale_missing_exchange_order(
    row: dict[str, Any],
    *,
    now_ts: float,
    grace_seconds: float,
    open_exchange_ids: set[str],
) -> bool:
    status = str(row.get("status") or "").strip().lower()
    if status not in {"submitted", "open", "pending", "partially_filled", "unknown", "new"}:
        return False
    ex_id = str(row.get("exchange_order_id") or "").strip()
    if not ex_id or ex_id in open_exchange_ids:
        return False
    
    # 白名单机制：保护 chop_grid 的网格入场单不被错误标为 expired
    strategy = str(row.get("strategy") or "").lower()
    client_id = str(row.get("client_order_id") or "").lower()
    if strategy == "chop_grid" and client_id.startswith("cg_"):
        # 检查是否为入场单 (L1/L2/S1/S2 等，不含 _tp/_sl)
        import re
        if re.search(r"_(L|S)\d+$", client_id, re.I):
            return False

    age_ref = row.get("updated_at") or row.get("created_at")
    age_ts = _parse_utc_ts(age_ref)
    if age_ts is None:
        return False
    return now_ts - age_ts >= max(0.0, grace_seconds)


def run_multi_leg_backfill_once(
    *,
    api: Any,
    storage: Any,
    lookback_hours: int,
    limit: int,
) -> int:
    """Run one REST backfill pass; return updated row count."""
    updated_rows = 0
    stale_marked = 0
    api_error_count = 0
    candidates = storage.get_recent_orders_for_backfill(
        lookback_hours=max(1, int(lookback_hours)),
        limit=max(1, int(limit)),
    )
    now_ts = time.time()
    stale_grace_seconds = float(
        max(
            0,
            _env_int("MLBOT_MULTI_LEG_STALE_OPEN_GRACE_SECONDS", 6 * 3600),
        )
    )
    symbols = sorted(
        {
            str(row.get("symbol") or "").strip().upper()
            for row in candidates
            if str(row.get("symbol") or "").strip()
        }
    )
    # None = open snapshot unavailable (API error); do not treat as empty exchange.
    open_exchange_ids_by_symbol: dict[str, set[str] | None] = {}
    open_rows_by_symbol: dict[str, list[dict[str, Any]] | None] = {}
    for symbol in symbols:
        rows = _fetch_open_order_rows(api, symbol)
        if rows is None:
            api_error_count += 1
            logger.debug(
                "multi-leg REST backfill: open snapshot failed symbol=%s",
                symbol,
                exc_info=True,
            )
            open_exchange_ids_by_symbol[symbol] = None
            open_rows_by_symbol[symbol] = None
            continue
        open_rows_by_symbol[symbol] = rows
        open_exchange_ids_by_symbol[symbol] = {
            str(o.get("order_id") or "").strip()
            for o in rows
            if str(o.get("order_id") or "").strip()
        }

    for row in candidates:
        ex_id = str(row.get("exchange_order_id") or "").strip()
        symbol = str(row.get("symbol") or "").strip().upper()
        if not ex_id or not symbol:
            continue
        try:
            kind, snap = _resolve_rest_order_snapshot(
                api, row, symbol=symbol, exchange_order_id=ex_id
            )
            if kind == "error":
                api_error_count += 1
                continue
            if not snap:
                open_rows = open_rows_by_symbol.get(symbol)
                local_status = str(row.get("status") or "").strip().lower()
                if open_rows and local_status not in {
                    "expired",
                    "canceled",
                    "rejected",
                    "filled",
                }:
                    for open_row in open_rows:
                        if str(open_row.get("order_id") or "").strip() != ex_id:
                            continue
                        payload = {
                            "run_id": row.get("run_id"),
                            "strategy": row.get("strategy"),
                            "symbol": symbol,
                            "order_id": ex_id,
                            "client_order_id": open_row.get("client_order_id")
                            or row.get("client_order_id"),
                            "status": normalize_rest_order_status(
                                open_row.get("status")
                            ),
                            "filled_qty": open_row.get("filled"),
                            "avg_price": open_row.get("average_price"),
                            "event_time": open_row.get("update_time")
                            or open_row.get("timestamp"),
                            "trade_time": open_row.get("update_time")
                            or open_row.get("timestamp"),
                            "raw": open_row,
                        }
                        changed = int(storage.apply_execution_report(payload) or 0)
                        updated_rows += max(0, changed)
                        break
                    continue
                open_exchange_ids = open_exchange_ids_by_symbol.get(symbol)
                if open_exchange_ids is None:
                    continue
                if _is_stale_missing_exchange_order(
                    row,
                    now_ts=now_ts,
                    grace_seconds=stale_grace_seconds,
                    open_exchange_ids=open_exchange_ids,
                ):
                    payload = {
                        "run_id": row.get("run_id"),
                        "strategy": row.get("strategy"),
                        "symbol": symbol,
                        "order_id": ex_id,
                        "client_order_id": row.get("client_order_id"),
                        "status": "expired",
                        "event_time": datetime.now(timezone.utc).isoformat(),
                        "reject_reason": "exchange_order_missing",
                        "error_message": "exchange_order_missing",
                        "raw": {
                            "source": "periodic_multi_leg_order_backfill",
                            "reason": "exchange_order_missing",
                            "exchange_order_id": ex_id,
                            "symbol": symbol,
                        },
                    }
                    changed = int(storage.apply_execution_report(payload) or 0)
                    if changed > 0:
                        stale_marked += changed
                        updated_rows += changed
                        logger.warning(
                            "multi-leg REST backfill: mark stale local open as expired "
                            "symbol=%s exchange_order_id=%s strategy=%s",
                            symbol,
                            ex_id,
                            row.get("strategy"),
                        )
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
            api_error_count += 1
            logger.debug(
                "multi-leg REST backfill skipped row local=%s exchange=%s symbol=%s",
                row.get("local_order_id"),
                ex_id,
                symbol,
                exc_info=True,
            )
    try:
        METRICS.update_reconciliation_metrics(
            scope="hedge",
            strategy="all",
            symbol="ALL",
            ok=(stale_marked == 0 and api_error_count == 0),
            issue_counts={
                "stale_local_order": stale_marked,
                "api_error": api_error_count,
            },
            ts_seconds=now_ts,
        )
    except Exception:
        logger.debug("multi-leg backfill reconciliation metrics skipped", exc_info=True)
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
