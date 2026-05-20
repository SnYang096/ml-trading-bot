"""
Trend / live 「终态订单」REST 回填（独立于 User Stream）。

``OrderFlowListener`` 仅在收到终端 WS 时尝试 ``sync_order_status``；已在库中但未带均价的单子不会再次触发 WS。
此处按可配置间隔调用 ``OrderManager.reconcile_recent_terminal_orders``，与磁盘 ``orders`` 表候选集对齐。

控制环境变量::

    MLBOT_TERMINAL_ORDER_BACKFILL_INTERVAL_SECONDS — 间隔秒数，<=0 或 ``off`` / ``false`` / ``no`` / ``disable`` / ``disabled`` 表示关闭。
    MLBOT_TERMINAL_ORDER_BACKFILL_LOOKBACK_HOURS — 候选 ``created_at`` 回溯窗口（传给 storage 查询）。
    MLBOT_TERMINAL_ORDER_BACKFILL_LIMIT — 单次最多回填条数。
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import partial
from typing import Any

from src.time_series_model.live.metrics_exporter import METRICS

logger = logging.getLogger(__name__)


_DEFAULT_INTERVAL = 60.0


def terminal_order_backfill_enabled_interval_seconds() -> float:
    """Return interval seconds, or ``0`` if periodic backfill is disabled."""
    unset = "MLBOT_TERMINAL_ORDER_BACKFILL_INTERVAL_SECONDS" not in os.environ
    raw = (
        (os.getenv("MLBOT_TERMINAL_ORDER_BACKFILL_INTERVAL_SECONDS") or "")
        .strip()
        .lower()
    )
    if raw in {"0", "false", "no", "off", "disable", "disabled"}:
        return 0.0
    if raw == "":
        return _DEFAULT_INTERVAL if unset else 0.0
    try:
        v = float(raw)
        return v if v > 0 else 0.0
    except ValueError:
        logger.warning(
            "MLBOT_TERMINAL_ORDER_BACKFILL_INTERVAL_SECONDS invalid %r → default %s",
            raw,
            _DEFAULT_INTERVAL,
        )
        return _DEFAULT_INTERVAL


def terminal_order_backfill_should_run(order_manager: Any | None) -> bool:
    if order_manager is None:
        return False
    if getattr(order_manager, "shadow", False):
        return False
    if getattr(order_manager, "binance_api", None) is None:
        return False
    fn = getattr(order_manager, "reconcile_recent_terminal_orders", None)
    return callable(fn)


def terminal_order_backfill_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("%s invalid %r -> default %d", name, raw, default)
        return int(default)


async def periodic_terminal_order_backfill(
    order_manager: Any,
    *,
    startup_delay_seconds: float = 20.0,
) -> None:
    """Coroutine: sleep-loop calling ``reconcile_recent_terminal_orders`` via executor."""

    interval = terminal_order_backfill_enabled_interval_seconds()
    if interval <= 0:
        return

    if not terminal_order_backfill_should_run(order_manager):
        logger.info(
            "Terminal order REST backfill: disabled (no non-shadow OrderManager with API)"
        )
        return

    lookback_hours = terminal_order_backfill_env_int(
        "MLBOT_TERMINAL_ORDER_BACKFILL_LOOKBACK_HOURS", 168
    )
    limit = terminal_order_backfill_env_int("MLBOT_TERMINAL_ORDER_BACKFILL_LIMIT", 200)
    logger.info(
        "Terminal order REST backfill: every %.0fs lookback_hours=%s limit=%s",
        interval,
        lookback_hours,
        limit,
    )

    if startup_delay_seconds > 0:
        try:
            await asyncio.sleep(startup_delay_seconds)
        except asyncio.CancelledError:
            raise

    loop = asyncio.get_running_loop()
    while True:
        try:
            updated = await loop.run_in_executor(
                None,
                partial(
                    order_manager.reconcile_recent_terminal_orders,
                    lookback_hours=max(1, lookback_hours),
                    limit=max(1, limit),
                ),
            )
            stats = getattr(order_manager, "_last_terminal_backfill_stats", {}) or {}
            stale_marked = int(stats.get("stale_marked", 0) or 0)
            api_error = int(stats.get("api_error", 0) or 0)
            try:
                METRICS.update_reconciliation_metrics(
                    scope="trend",
                    strategy="all",
                    symbol="ALL",
                    ok=(stale_marked == 0 and api_error == 0),
                    issue_counts={
                        "stale_local_order": stale_marked,
                        "api_error": api_error,
                    },
                )
            except Exception:
                logger.debug("terminal backfill reconciliation metrics skipped", exc_info=True)
            if updated:
                logger.info(
                    "Terminal order REST backfill: updated %s row(s)",
                    len(updated),
                )
        except asyncio.CancelledError:
            break
        except Exception:
            logger.warning(
                "Terminal order REST backfill iteration failed", exc_info=True
            )

        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break
