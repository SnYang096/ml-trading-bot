#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yaml

from src.config.strategy_layout import resolve_strategy_package_under_root
from src.live_data_stream.classic_feature_bus_provider import ClassicFeatureBusProvider
from src.live_data_stream.constitution_config import (
    load_constitution_dict,
    resolve_constitution_yaml,
    spot_account_equity_anchor_usdt,
    spot_strategies_from_constitution,
)
from src.order_management.spot_binance_api import SpotBinanceAPI
from src.order_management.spot_live_recovery import (
    OPEN_BUY_STATUSES,
    apply_buy_fill_to_position,
    apply_sell_fill_to_position,
    clear_pending_buy,
    effective_symbol_deployed,
    has_blocking_pending_buy,
    merge_rebuilt_deploy_into_positions,
    mark_pending_fill_recorded,
    new_position_shell,
    normalize_spot_symbol,
    parse_ccxt_fill,
    pending_buy_count_for_day,
    pending_buy_age_hours,
    pending_buy_quote_for_day,
    pending_fill_delta,
    rebuild_positions_from_filled_orders,
    set_pending_buy,
    sync_position_qty_from_balance,
    iso_now,
)
from src.order_management.spot_order_manager import SpotOrderManager
from src.time_series_model.live.metrics_exporter import METRICS, start_metrics_server
from src.time_series_model.live.decision_chain_debug import (
    chain_debug_enabled,
    collect_spot_new_buy_report,
    log_spot_new_buy_eligibility,
    log_spot_no_intent,
    spot_eligibility_log_enabled,
)
from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy
from src.time_series_model.live.non_trend_funnel import (
    FifteenMinFlusher,
    default_live_monitor_db_path,
    funnel_for_spot_decision,
)
from src.time_series_model.live.stats_collector import StatsCollector
from src.time_series_model.live.spot_accum_simple import (
    apply_partial_sell_to_position,
    deploy_decay_multiplier,
    deploy_schedule_allows_new_buy,
    deploy_schedule_policy,
    maybe_spot_simple_partial_sell,
    pending_buy_max_age_hours as schedule_pending_buy_max_age_hours,
    resolve_min_profit_multiple,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_spot_accum_live")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _is_insufficient_funds(exc: BaseException) -> bool:
    try:
        from ccxt.base.errors import InsufficientFunds
    except ImportError:
        InsufficientFunds = ()  # type: ignore[misc,assignment]
    if isinstance(exc, InsufficientFunds):
        return True
    msg = str(exc).lower()
    return "insufficient balance" in msg or "insufficient funds" in msg


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _tf_to_minutes(tf: str) -> int:
    t = str(tf or "").strip().upper()
    if t.endswith("T"):
        return max(1, int(t[:-1]))
    if t.endswith("H"):
        return max(1, int(float(t[:-1]) * 60))
    return 120


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in str(raw or "").split(",") if s.strip()]


@dataclass
class SpotBudgetConfig:
    equity_anchor_usdt: float
    target_deploy_pct: float
    max_gross_notional_pct: float
    max_daily_deploy_pct: float
    min_order_interval_minutes: int
    max_new_entries_per_day: int
    symbol_budgets_usdt: Dict[str, float]
    symbol_units_usdt: Dict[str, float]
    entry_order_type: str
    entry_limit_offset_bps: float
    deploy_decay_cfg: Dict[str, Any]
    deploy_schedule_cfg: Dict[str, Any]
    profit_take_ladder_cfg: Dict[str, Any]


class SpotAccumLedger:
    """Persist spot positions + daily deploy counters for spot_accum_simple."""

    def __init__(self, db_path: str) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS state_kv (
                    k TEXT PRIMARY KEY,
                    v TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_counters (
                    day_key TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    buy_entries INTEGER NOT NULL DEFAULT 0,
                    deploy_usdt REAL NOT NULL DEFAULT 0.0,
                    PRIMARY KEY (day_key, symbol)
                )
                """
            )
            conn.commit()

    def load_positions(self) -> Dict[str, Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT v FROM state_kv WHERE k='positions'").fetchone()
            if not row:
                return {}
            try:
                raw = json.loads(row["v"])
            except Exception:
                return {}
            return raw if isinstance(raw, dict) else {}

    def save_positions(self, positions: Dict[str, Dict[str, Any]]) -> None:
        payload = json.dumps(positions, ensure_ascii=True, separators=(",", ":"))
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO state_kv(k,v) VALUES('positions', ?)",
                (payload,),
            )
            conn.commit()

    def add_buy(self, day_key: str, symbol: str, deploy_usdt: float) -> None:
        sym = symbol.upper()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO daily_counters(day_key, symbol, buy_entries, deploy_usdt)
                VALUES(?, ?, 1, ?)
                ON CONFLICT(day_key, symbol)
                DO UPDATE SET
                    buy_entries = buy_entries + 1,
                    deploy_usdt = deploy_usdt + excluded.deploy_usdt
                """,
                (day_key, sym, float(deploy_usdt)),
            )
            conn.commit()

    def buy_entries_today(self, day_key: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(buy_entries),0) AS n FROM daily_counters WHERE day_key=?",
                (day_key,),
            ).fetchone()
            return int(row["n"] if row else 0)

    def deploy_today_usdt(self, day_key: str) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(deploy_usdt),0.0) AS v FROM daily_counters WHERE day_key=?",
                (day_key,),
            ).fetchone()
            return float(row["v"] if row else 0.0)


def _load_spot_budget_config(
    *,
    constitution: Dict[str, Any],
    strategy_name: str,
    strategy: GenericLiveStrategy,
) -> SpotBudgetConfig:
    spot = (
        constitution.get("spot") if isinstance(constitution.get("spot"), dict) else {}
    )
    account = spot.get("account") if isinstance(spot.get("account"), dict) else {}
    accumulation = (
        spot.get("accumulation") if isinstance(spot.get("accumulation"), dict) else {}
    )
    risk_limits = (
        spot.get("risk_limits") if isinstance(spot.get("risk_limits"), dict) else {}
    )
    sl = (
        spot.get("strategy_limits")
        if isinstance(spot.get("strategy_limits"), dict)
        else {}
    )
    strategy_limits = (
        sl.get(strategy_name) if isinstance(sl.get(strategy_name), dict) else {}
    )

    symbol_budgets_raw = accumulation.get("symbol_budgets_usdt")
    symbol_budgets: Dict[str, float] = {}
    if isinstance(symbol_budgets_raw, dict):
        for k, v in symbol_budgets_raw.items():
            try:
                fv = float(v or 0.0)
            except (TypeError, ValueError):
                continue
            if fv > 0:
                symbol_budgets[str(k).upper()] = fv

    symbol_units_raw = accumulation.get("symbol_unit_notional_usdt")
    symbol_units: Dict[str, float] = {}
    if isinstance(symbol_units_raw, dict):
        for k, v in symbol_units_raw.items():
            try:
                fv = float(v or 0.0)
            except (TypeError, ValueError):
                continue
            if fv > 0:
                symbol_units[str(k).upper()] = fv

    exec_raw = strategy.archetype.execution.raw if strategy.archetype else {}
    if not isinstance(exec_raw, dict):
        exec_raw = {}
    exec_cons = exec_raw.get("execution_constraints")
    if not isinstance(exec_cons, dict):
        exec_cons = {}
    entry_order = exec_cons.get("entry_order")
    if not isinstance(entry_order, dict):
        entry_order = {}
    ladder = (
        (exec_raw.get("stop_loss") or {}).get("profit_take_ladder")
        if isinstance(exec_raw.get("stop_loss"), dict)
        else {}
    )
    if not isinstance(ladder, dict):
        ladder = {}
    decay_cfg = exec_raw.get("deploy_decay")
    if not isinstance(decay_cfg, dict):
        decay_cfg = {}
    schedule_cfg = deploy_schedule_policy(exec_raw)

    min_interval = max(
        _env_int("MLBOT_SPOT_MIN_ORDER_INTERVAL_MINUTES", 0),
        int(accumulation.get("min_order_interval_minutes", 0) or 0),
        int(exec_cons.get("min_order_interval_minutes", 0) or 0),
    )
    max_new = int(strategy_limits.get("max_new_entries_per_day", 1) or 1)

    return SpotBudgetConfig(
        equity_anchor_usdt=spot_account_equity_anchor_usdt(account),
        target_deploy_pct=float(accumulation.get("target_deploy_pct", 1.0) or 1.0),
        max_gross_notional_pct=float(
            risk_limits.get("max_gross_notional_pct", 1.0) or 1.0
        ),
        max_daily_deploy_pct=float(risk_limits.get("max_daily_deploy_pct", 1.0) or 1.0),
        min_order_interval_minutes=max(0, min_interval),
        max_new_entries_per_day=max(1, max_new),
        symbol_budgets_usdt=symbol_budgets,
        symbol_units_usdt=symbol_units,
        entry_order_type=str(entry_order.get("type", "market")).strip().lower(),
        entry_limit_offset_bps=float(entry_order.get("limit_offset_bps", 0.0) or 0.0),
        deploy_decay_cfg=decay_cfg,
        deploy_schedule_cfg=schedule_cfg,
        profit_take_ladder_cfg=ladder,
    )


def _utc_day_key(ts: Any) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.strftime("%Y-%m-%d")


def _spot_available_usdt_for_buy(
    om: SpotOrderManager,
    positions: Dict[str, Dict[str, Any]],
) -> Optional[float]:
    """Free USDT on exchange minus quote reserved by open limit buys."""
    if om.api is None or om.shadow:
        return None
    try:
        free = om.api.get_free_balances()
        usdt = float(free.get("USDT", 0.0) or 0.0)
        reserved = 0.0
        for pos in positions.values():
            if not has_blocking_pending_buy(pos):
                continue
            pending = pos.get("_pending_buy") if isinstance(pos, dict) else None
            if not isinstance(pending, dict):
                continue
            reserved += float(pending.get("quote_reserved", 0.0) or 0.0)
        return max(0.0, usdt - reserved)
    except Exception:
        logger.warning("spot: failed to read free USDT balance", exc_info=True)
        return None


def _symbol_base_asset(symbol: str) -> str:
    s = symbol.upper()
    if s.endswith("USDT"):
        return s[:-4]
    return s


def _planned_buy_quote_usdt(
    *,
    symbol: str,
    size_multiplier: float,
    budget: SpotBudgetConfig,
    positions: Dict[str, Dict[str, Any]],
    deploy_today: float,
    day_key: str,
) -> float:
    sym = symbol.upper()
    unit = float(budget.symbol_units_usdt.get(sym, 0.0) or 0.0)
    if unit <= 0:
        return 0.0
    leg = max(0.0, unit * max(0.0, float(size_multiplier or 1.0)))
    symbol_budget = float(budget.symbol_budgets_usdt.get(sym, 0.0) or 0.0)
    if symbol_budget <= 0.0:
        return 0.0

    pos = positions.get(sym) or {}
    deployed_symbol = effective_symbol_deployed(pos)
    if budget.deploy_decay_cfg.get("enabled", False):
        leg *= deploy_decay_multiplier(
            deployed_symbol, symbol_budget, budget.deploy_decay_cfg
        )

    global_cap = (
        float(budget.equity_anchor_usdt)
        * float(budget.target_deploy_pct)
        * float(budget.max_gross_notional_pct)
    )
    global_deployed = sum(
        effective_symbol_deployed(p or {}) for p in positions.values()
    )
    daily_cap = float(budget.equity_anchor_usdt) * float(budget.max_daily_deploy_pct)

    remain_symbol = max(0.0, symbol_budget - deployed_symbol)
    remain_global = max(0.0, global_cap - global_deployed)
    pending_today = pending_buy_quote_for_day(positions, day_key=day_key)
    remain_daily = max(0.0, daily_cap - deploy_today - pending_today)
    return max(0.0, min(leg, remain_symbol, remain_global, remain_daily))


def _pending_buy_max_age_hours(schedule_cfg: Optional[Dict[str, Any]] = None) -> float:
    env_default = max(1.0, _env_float("MLBOT_SPOT_PENDING_BUY_MAX_HOURS", 24.0))
    if isinstance(schedule_cfg, dict) and schedule_cfg:
        return schedule_pending_buy_max_age_hours(schedule_cfg, default=env_default)
    return env_default


def _apply_buy_fill(
    *,
    sym: str,
    positions: Dict[str, Dict[str, Any]],
    ledger: SpotAccumLedger,
    budget: SpotBudgetConfig,
    day_key: str,
    fill_qty: float,
    fill_quote: float,
    filled_at: str,
) -> None:
    quote = apply_buy_fill_to_position(
        positions.setdefault(
            sym,
            new_position_shell(
                sym, profit_take_ladder_cfg=budget.profit_take_ladder_cfg
            ),
        ),
        fill_qty=fill_qty,
        fill_quote_usdt=fill_quote,
        profit_take_ladder_cfg=budget.profit_take_ladder_cfg,
        filled_at=filled_at,
    )
    if quote > 0.0:
        ledger.add_buy(day_key, sym, quote)


def _finalize_buy_fill_from_order(
    *,
    sym: str,
    local_order_id: str,
    om: SpotOrderManager,
    positions: Dict[str, Dict[str, Any]],
    ledger: SpotAccumLedger,
    budget: SpotBudgetConfig,
    day_key: str,
    payload: Dict[str, Any],
) -> bool:
    status, filled_qty, fill_quote, _avg = parse_ccxt_fill(payload)
    pos = positions.get(sym)
    pending = pos.get("_pending_buy") if isinstance(pos, dict) else None
    if isinstance(pending, dict):
        delta_qty, delta_quote = pending_fill_delta(
            pending, filled_qty=filled_qty, filled_quote=fill_quote
        )
    else:
        delta_qty, delta_quote = filled_qty, fill_quote
    om.update_order_record(
        local_order_id,
        status=status,
        filled_quantity=filled_qty,
        filled_quote_usdt=fill_quote,
        raw_json=json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
    )
    if delta_qty > 0.0 and delta_quote > 0.0:
        _apply_buy_fill(
            sym=sym,
            positions=positions,
            ledger=ledger,
            budget=budget,
            day_key=day_key,
            fill_qty=delta_qty,
            fill_quote=delta_quote,
            filled_at=str(payload.get("datetime") or payload.get("timestamp") or ""),
        )
        logger.info(
            "[%s] buy fill delta qty=%.8f quote=%.2f status=%s",
            sym,
            delta_qty,
            delta_quote,
            status,
        )
    pos = positions.get(sym)
    pending = pos.get("_pending_buy") if isinstance(pos, dict) else None
    if isinstance(pending, dict):
        mark_pending_fill_recorded(
            pending, filled_qty=filled_qty, filled_quote=fill_quote
        )
    if status in OPEN_BUY_STATUSES:
        return delta_qty > 0.0
    if pos is not None:
        clear_pending_buy(pos)
    return status in {"closed", "filled"} or delta_qty > 0.0


def _cancel_stale_pending_buy(
    *,
    sym: str,
    pos: Dict[str, Any],
    om: SpotOrderManager,
    max_age_hours: float,
    now: datetime,
) -> bool:
    pending = pos.get("_pending_buy")
    if not isinstance(pending, dict):
        return False
    age_h = pending_buy_age_hours(pending, now=now)
    if age_h < max_age_hours:
        return False
    ex_id = str(pending.get("exchange_order_id") or "")
    if ex_id and om.api is not None and not om.shadow:
        try:
            om.cancel_exchange_order(sym, ex_id)
            logger.warning(
                "[%s] canceled stale pending buy after %.1fh exchange_order_id=%s",
                sym,
                age_h,
                ex_id,
            )
        except Exception as exc:
            logger.warning("[%s] cancel stale pending buy failed: %s", sym, exc)
            return False
    local_id = str(pending.get("local_order_id") or "")
    if local_id:
        om.update_order_record(local_id, status="canceled")
    clear_pending_buy(pos)
    return True


def _refresh_pending_buy_from_exchange(
    *,
    sym: str,
    pos: Dict[str, Any],
    om: SpotOrderManager,
    positions: Dict[str, Dict[str, Any]],
    ledger: SpotAccumLedger,
    budget: SpotBudgetConfig,
    day_key: str,
) -> None:
    pending = pos.get("_pending_buy")
    if not isinstance(pending, dict) or om.api is None or om.shadow:
        return
    ex_id = str(pending.get("exchange_order_id") or "")
    if not ex_id:
        return
    try:
        payload = om.api.fetch_order(sym, ex_id)
    except Exception as exc:
        logger.debug("[%s] fetch pending buy failed: %s", sym, exc)
        return
    status, filled_qty, fill_quote, _avg = parse_ccxt_fill(payload)
    local_id = str(pending.get("local_order_id") or "")
    if local_id:
        om.update_order_record(
            local_id,
            status=status,
            filled_quantity=filled_qty,
            filled_quote_usdt=fill_quote,
            raw_json=json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        )
    if status in {"closed", "filled"} or filled_qty > 0.0:
        _finalize_buy_fill_from_order(
            sym=sym,
            local_order_id=local_id,
            om=om,
            positions=positions,
            ledger=ledger,
            budget=budget,
            day_key=day_key,
            payload=payload,
        )
    elif status in {"canceled", "cancelled", "expired", "rejected"}:
        clear_pending_buy(pos)


def _spot_startup_recovery(
    *,
    om: SpotOrderManager,
    ledger: SpotAccumLedger,
    positions: Dict[str, Dict[str, Any]],
    budget: SpotBudgetConfig,
    symbols: List[str],
) -> None:
    """Load ledger, rebuild deploy from filled orders, reconcile pending limits."""
    max_age_h = _pending_buy_max_age_hours(budget.deploy_schedule_cfg)
    now = datetime.now(timezone.utc)
    day_key = _utc_day_key(now)

    order_rows = om.list_orders_for_symbols(symbols)
    orders_by_id = {str(row.get("order_id") or ""): row for row in order_rows}
    rebuilt = rebuild_positions_from_filled_orders(
        order_rows,
        symbols=symbols,
        profit_take_ladder_cfg=budget.profit_take_ladder_cfg,
    )
    merge_rebuilt_deploy_into_positions(
        positions,
        rebuilt,
        profit_take_ladder_cfg=budget.profit_take_ladder_cfg,
    )

    if om.api is not None and not om.shadow:
        prefix = om.client_prefix
        symbol_set = {s.upper() for s in symbols}
        try:
            open_orders = om.api.fetch_open_orders()
        except Exception as exc:
            logger.warning("spot startup: fetch_open_orders failed: %s", exc)
            open_orders = []
        for order in open_orders:
            info = order.get("info") if isinstance(order.get("info"), dict) else {}
            sym = normalize_spot_symbol(str(order.get("symbol") or ""))
            if sym not in symbol_set:
                continue
            side = str(order.get("side") or "").lower()
            if side != "buy":
                continue
            cid = str(
                order.get("clientOrderId")
                or order.get("client_order_id")
                or info.get("clientOrderId")
                or info.get("client_order_id")
                or ""
            )
            if prefix and cid and not cid.startswith(prefix):
                continue
            ex_id = str(
                order.get("id") or order.get("orderId") or info.get("orderId") or ""
            )
            local_id = (
                om.find_order_id(
                    exchange_order_id=ex_id,
                    client_order_id=cid,
                )
                or f"exchange_{ex_id}"
            )
            local_row = orders_by_id.get(local_id) or {}
            qty = float(order.get("amount") or order.get("origQty") or 0.0)
            px = float(order.get("price") or 0.0)
            quote_reserved = qty * px if px > 0 else 0.0
            ts_ms = order.get("timestamp")
            placed_at = (
                pd.Timestamp(int(ts_ms), unit="ms", tz="UTC").isoformat()
                if ts_ms
                else iso_now()
            )
            pos = positions.setdefault(
                sym,
                new_position_shell(
                    sym, profit_take_ladder_cfg=budget.profit_take_ladder_cfg
                ),
            )
            old_pending = pos.get("_pending_buy") if isinstance(pos, dict) else {}
            old_recorded_qty = 0.0
            old_recorded_quote = 0.0
            if isinstance(old_pending, dict) and (
                str(old_pending.get("exchange_order_id") or "") == ex_id
                or str(old_pending.get("client_order_id") or "") == cid
            ):
                old_recorded_qty = float(
                    old_pending.get("filled_quantity_recorded", 0.0) or 0.0
                )
                old_recorded_quote = float(
                    old_pending.get("filled_quote_recorded", 0.0) or 0.0
                )
            set_pending_buy(
                pos,
                local_order_id=local_id,
                exchange_order_id=ex_id,
                client_order_id=cid,
                quantity=qty,
                price=px if px > 0 else None,
                quote_reserved=quote_reserved,
                placed_at=placed_at,
                filled_quantity_recorded=max(
                    old_recorded_qty,
                    float(local_row.get("filled_quantity", 0.0) or 0.0),
                ),
                filled_quote_recorded=max(
                    old_recorded_quote,
                    float(local_row.get("filled_quote_usdt", 0.0) or 0.0),
                ),
            )
            _status, filled_qty, fill_quote, _avg = parse_ccxt_fill(order)
            if filled_qty > 0.0 and fill_quote > 0.0:
                _finalize_buy_fill_from_order(
                    sym=sym,
                    local_order_id=local_id,
                    om=om,
                    positions=positions,
                    ledger=ledger,
                    budget=budget,
                    day_key=day_key,
                    payload=order,
                )
            pending_after_fill = pos.get("_pending_buy")
            if not isinstance(pending_after_fill, dict):
                continue
            age_h = pending_buy_age_hours(pending_after_fill, now=now)
            if age_h >= max_age_h and ex_id:
                _cancel_stale_pending_buy(
                    sym=sym, pos=pos, om=om, max_age_hours=max_age_h, now=now
                )

    for sym in symbols:
        sym_u = sym.upper()
        pos = positions.get(sym_u)
        if pos is not None:
            _cancel_stale_pending_buy(
                sym=sym_u, pos=pos, om=om, max_age_hours=max_age_h, now=now
            )
            _refresh_pending_buy_from_exchange(
                sym=sym_u,
                pos=pos,
                om=om,
                positions=positions,
                ledger=ledger,
                budget=budget,
                day_key=day_key,
            )

    if om.api is None:
        return
    balances = om.api.get_total_balances()
    for sym in symbols:
        sym_u = sym.upper()
        base = _symbol_base_asset(sym_u)
        qty_live = float(balances.get(base, 0.0) or 0.0)
        if qty_live <= 0.0 and sym_u not in positions:
            continue
        if qty_live <= 0.0:
            pos = positions.get(sym_u)
            if pos is not None and not has_blocking_pending_buy(pos):
                positions.pop(sym_u, None)
            continue
        px = om.api.get_last_price(sym_u)
        pos = positions.setdefault(
            sym_u,
            new_position_shell(
                sym_u, profit_take_ladder_cfg=budget.profit_take_ladder_cfg
            ),
        )
        sync_position_qty_from_balance(pos, qty_live=qty_live, mark_price=px)
        deploy = float(pos.get("_spot_quote_deployed", 0.0) or 0.0)
        qty = float(pos.get("_qty_base", 0.0) or 0.0)
        if qty > 0.0 and deploy <= 0.0:
            logger.warning(
                "[%s] exchange qty=%.8f but ledger deploy=0; check spot_orders history",
                sym_u,
                qty,
            )


def _spot_process_pending_buys(
    *,
    om: SpotOrderManager,
    positions: Dict[str, Dict[str, Any]],
    ledger: SpotAccumLedger,
    budget: SpotBudgetConfig,
    symbols: List[str],
    day_key: str,
) -> Dict[str, int]:
    max_age_h = _pending_buy_max_age_hours(budget.deploy_schedule_cfg)
    now = datetime.now(timezone.utc)
    stats = {"pending_open": 0, "stale_local_order": 0, "api_error": 0}
    for sym in symbols:
        sym_u = sym.upper()
        pos = positions.get(sym_u)
        if pos is None:
            continue
        pending = pos.get("_pending_buy") if isinstance(pos, dict) else None
        if isinstance(pending, dict):
            stats["pending_open"] += 1
        if has_blocking_pending_buy(pos):
            _refresh_pending_buy_from_exchange(
                sym=sym_u,
                pos=pos,
                om=om,
                positions=positions,
                ledger=ledger,
                budget=budget,
                day_key=day_key,
            )
        if _cancel_stale_pending_buy(
            sym=sym_u, pos=pos, om=om, max_age_hours=max_age_h, now=now
        ):
            continue
        pending_after = pos.get("_pending_buy") if isinstance(pos, dict) else None
        if (
            isinstance(pending_after, dict)
            and pending_buy_age_hours(pending_after, now=now) >= max_age_h
        ):
            stats["stale_local_order"] += 1
            if (
                str(pending_after.get("exchange_order_id") or "").strip()
                and om.api is not None
                and not om.shadow
            ):
                stats["api_error"] += 1
    return stats


def _publish_spot_reconciliation_metrics(
    *,
    strategy_name: str,
    symbols: List[str],
    positions: Dict[str, Dict[str, Any]],
    om: SpotOrderManager,
    pending_stats: Dict[str, int],
) -> None:
    issue_counts: Dict[str, float] = {
        "stale_local_order": float(pending_stats.get("stale_local_order", 0) or 0),
        "api_error": float(pending_stats.get("api_error", 0) or 0),
        "position_mismatch": 0.0,
    }
    if om.api is not None:
        try:
            balances = om.api.get_total_balances()
            abs_tol = float(os.getenv("MLBOT_SPOT_RECON_QTY_ABS_TOL", "0.000001"))
            rel_tol = float(os.getenv("MLBOT_SPOT_RECON_QTY_REL_TOL", "0.02"))
            for sym in symbols:
                sym_u = sym.upper()
                base = _symbol_base_asset(sym_u)
                live_qty = float(balances.get(base, 0.0) or 0.0)
                local_qty = float(
                    ((positions.get(sym_u) or {}).get("_qty_base", 0.0) or 0.0)
                )
                tol = max(abs_tol, rel_tol * max(abs(live_qty), abs(local_qty), 1.0))
                if abs(live_qty - local_qty) > tol:
                    issue_counts["position_mismatch"] += 1.0
        except Exception:
            issue_counts["api_error"] += 1.0
            logger.warning("spot reconcile: balance check failed", exc_info=True)
    ok = all(float(v or 0.0) <= 0.0 for v in issue_counts.values())
    METRICS.update_reconciliation_metrics(
        scope="spot",
        strategy=strategy_name,
        symbol="ALL",
        ok=ok,
        issue_counts=issue_counts,
    )


def _build_spot_order_manager() -> SpotOrderManager:
    shadow = _env_bool("MLBOT_SPOT_SHADOW_MODE", True)
    enabled = _env_bool("MLBOT_SPOT_ORDER_MANAGER_ENABLED", True)
    if not enabled:
        shadow = True
    testnet = _env_bool("MLBOT_SPOT_TESTNET", False)
    api_key = os.getenv("BINANCE_SPOT_API_KEY", "")
    api_secret = os.getenv("BINANCE_SPOT_API_SECRET", "")
    api: Optional[SpotBinanceAPI] = None
    if not shadow:
        if not api_key or not api_secret:
            raise RuntimeError(
                "spot live requires BINANCE_SPOT_API_KEY/BINANCE_SPOT_API_SECRET when shadow=false"
            )
        api = SpotBinanceAPI(api_key=api_key, api_secret=api_secret, testnet=testnet)
    elif api_key and api_secret:
        # Allow shadow mode to still read balances/prices when keys are provided.
        api = SpotBinanceAPI(api_key=api_key, api_secret=api_secret, testnet=testnet)
    return SpotOrderManager(
        db_path=os.getenv("MLBOT_SPOT_DB_PATH", "data/spot_accum_live.db"),
        api=api,
        shadow=shadow,
        client_prefix=os.getenv("MLBOT_SPOT_CLIENT_ORDER_PREFIX", "sa"),
    )


def _startup_checks(
    *,
    constitution: Dict[str, Any],
    strategy_name: str,
    symbols: List[str],
    feature_bus_root: str,
    om: SpotOrderManager,
) -> None:
    spot_strats = set(spot_strategies_from_constitution(constitution))
    if strategy_name.lower() not in spot_strats:
        raise RuntimeError(
            f"{strategy_name} is not enabled in constitution spot.strategies={sorted(spot_strats)}"
        )
    if not symbols:
        raise RuntimeError("empty symbol list for spot_accum_simple live")
    if not Path(feature_bus_root).exists():
        logger.warning("feature bus root does not exist yet: %s", feature_bus_root)
    if _env_bool("MLBOT_ORDER_MANAGER_ENABLED", False):
        logger.warning(
            "MLBOT_ORDER_MANAGER_ENABLED=true is futures manager; keep it disabled for spot-only runner."
        )
    if om.api is None and not om.shadow:
        raise RuntimeError("spot order manager has no API while shadow=false")


def main() -> int:
    strategy_name = (
        os.getenv("MLBOT_SPOT_STRATEGY", "spot_accum_simple").strip().lower()
    )
    strategies_root = os.getenv(
        "MLBOT_SPOT_STRATEGIES_ROOT", "live/highcap/config/strategies"
    )
    constitution_yaml = resolve_constitution_yaml(
        strategies_root,
        override=os.getenv("MLBOT_SPOT_CONSTITUTION_YAML"),
    )
    constitution = load_constitution_dict(constitution_yaml)

    pkg = resolve_strategy_package_under_root(
        Path(strategies_root), strategy_name, allow_bad_candidates=False
    )
    meta_path = pkg / "meta.yaml"
    if not meta_path.exists():
        fallback_root = "config/strategies"
        fallback_pkg = resolve_strategy_package_under_root(
            Path(fallback_root), strategy_name, allow_bad_candidates=False
        )
        fallback_meta = fallback_pkg / "meta.yaml"
        if fallback_meta.exists():
            logger.warning(
                "spot strategy meta not found under %s, fallback to %s",
                strategies_root,
                fallback_root,
            )
            strategies_root = fallback_root
            pkg = fallback_pkg
            meta_path = fallback_meta
            constitution_yaml = resolve_constitution_yaml(
                strategies_root,
                override=os.getenv("MLBOT_SPOT_CONSTITUTION_YAML"),
            )
            constitution = load_constitution_dict(constitution_yaml)
        else:
            raise FileNotFoundError(f"missing strategy meta: {meta_path}")
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    strategy_meta = (
        meta.get("strategy") if isinstance(meta.get("strategy"), dict) else {}
    )
    tf = str(strategy_meta.get("timeframe", "120T"))
    symbols = [
        str(s).upper()
        for s in (strategy_meta.get("symbol_include") or [])
        if str(s).strip()
    ]
    sym_override = os.getenv("MLBOT_SPOT_SYMBOLS")
    if sym_override:
        symbols = _parse_symbols(sym_override)

    feature_bus_root = os.getenv("MLBOT_FEATURE_BUS_ROOT", "live/shared_feature_bus")
    poll_seconds = _env_float("MLBOT_SPOT_FEATURE_BUS_POLL_SECONDS", 5.0)
    max_stale = _env_float("MLBOT_SPOT_FEATURE_BUS_MAX_STALENESS_SECONDS", 1800.0)
    chain_debug = chain_debug_enabled("spot") or _env_bool(
        "MLBOT_SPOT_CHAIN_DEBUG", True
    )

    strategy = GenericLiveStrategy(
        strategy_name=strategy_name,
        strategies_root=strategies_root,
        trade_size=1.0,
        primary_timeframe=tf,
        bar_minutes=_tf_to_minutes(tf),
    )
    budget = _load_spot_budget_config(
        constitution=constitution,
        strategy_name=strategy_name,
        strategy=strategy,
    )
    from src.live_data_stream.feature_bus import (
        list_feature_bus_timeframe_dirs,
        resolve_disk_primary_timeframe,
    )

    bus_tf, bus_legacy = resolve_disk_primary_timeframe(feature_bus_root, tf)
    if bus_legacy:
        logger.warning(
            "spot feature bus: reading legacy features/primary/ (strategy tf=%s)",
            tf,
        )
    bus_dirs = list_feature_bus_timeframe_dirs(feature_bus_root)
    provider = ClassicFeatureBusProvider(
        feature_bus_root=feature_bus_root,
        symbols=symbols,
        primary_timeframe=bus_tf,
        timeframes=[bus_tf] if bus_tf == tf else [bus_tf, tf],
        max_staleness_seconds=max_stale,
    )
    om = _build_spot_order_manager()
    metrics_port = _env_int("MLBOT_METRICS_PORT", 9193)
    start_metrics_server(port=metrics_port)
    METRICS.publish_dashboard_catalog(strategies=[strategy_name], symbols=symbols)

    stats_collector: Optional[StatsCollector] = None
    funnel_flusher: Optional[FifteenMinFlusher] = None
    if not _env_bool("MLBOT_SPOT_FUNNEL_DISABLE", False):
        try:
            stats_collector = StatsCollector(
                db_path=str(default_live_monitor_db_path()),
                auto_cleanup=False,
            )
            funnel_flusher = FifteenMinFlusher(
                stats_collector,
                interval_s=_env_float("MLBOT_SPOT_FUNNEL_FLUSH_SECONDS", 900.0),
            )
            logger.info(
                "spot funnel: writing 15min snapshots to %s",
                default_live_monitor_db_path(),
            )
        except Exception:
            logger.exception("spot funnel: StatsCollector init failed; funnel disabled")
            stats_collector = None
            funnel_flusher = None
    _startup_checks(
        constitution=constitution,
        strategy_name=strategy_name,
        symbols=symbols,
        feature_bus_root=feature_bus_root,
        om=om,
    )
    ledger = SpotAccumLedger(
        os.getenv("MLBOT_SPOT_LEDGER_DB_PATH", "data/spot_accum_ledger.db")
    )
    positions = ledger.load_positions()
    _spot_startup_recovery(
        om=om,
        ledger=ledger,
        positions=positions,
        budget=budget,
        symbols=symbols,
    )
    ledger.save_positions(positions)

    logger.info(
        "spot runner started: strategy=%s meta_tf=%s bus_tf=%s bus_dirs=%s symbols=%s "
        "shadow=%s bus=%s metrics_port=%s",
        strategy_name,
        tf,
        bus_tf,
        bus_dirs,
        symbols,
        om.shadow,
        feature_bus_root,
        metrics_port,
    )
    if chain_debug:
        logger.info(
            "spot chain debug enabled: prints signal->order->ledger->sell checks"
        )
    logger.info(
        "spot eligibility log: %s (MLBOT_SPOT_ELIGIBILITY_LOG=false to disable)",
        "on" if spot_eligibility_log_enabled() else "off",
    )
    sched = budget.deploy_schedule_cfg or {}
    if sched.get("enabled"):
        logger.info(
            "spot deploy_schedule: tz=%s window=%s-%s pending_max_age_h=%.1f",
            sched.get("timezone"),
            sched.get("new_order_local_start"),
            sched.get("new_order_local_end"),
            _pending_buy_max_age_hours(sched),
        )
    else:
        logger.info("spot deploy_schedule: disabled (new limit may fire on any 2h bar)")
    last_account_sync = 0.0
    last_bus_status_log = 0.0
    last_reconcile_metrics_sync = 0.0
    reconcile_metrics_interval = max(
        10.0, float(os.getenv("MLBOT_SPOT_RECONCILE_METRICS_SECONDS", "30"))
    )
    bus_status_interval = max(
        60.0, float(os.getenv("MLBOT_SPOT_BUS_STATUS_LOG_SECONDS", "3600"))
    )

    while True:
        try:
            METRICS.update_system_health()
            events = provider.poll()
            if not events:
                now_mono = time.monotonic()
                if now_mono - last_bus_status_log >= bus_status_interval:
                    last_bus_status_log = now_mono
                    ages = {}
                    for sym in symbols:
                        age = provider.reader.latest_snapshot_age_seconds(
                            symbol=sym, timeframe=bus_tf
                        )
                        ages[sym] = None if age is None else round(float(age), 1)
                    logger.info(
                        "spot bus idle: no new %s rows (poll=%.0fs); snapshot_age_s=%s",
                        bus_tf,
                        poll_seconds,
                        ages,
                    )
            loop_day_key = _utc_day_key(datetime.now(timezone.utc))
            free_usdt = _spot_available_usdt_for_buy(om, positions)
            pending_stats = _spot_process_pending_buys(
                om=om,
                positions=positions,
                ledger=ledger,
                budget=budget,
                symbols=symbols,
                day_key=loop_day_key,
            )
            now_ts = time.time()
            if now_ts - last_reconcile_metrics_sync >= reconcile_metrics_interval:
                _publish_spot_reconciliation_metrics(
                    strategy_name=strategy_name,
                    symbols=symbols,
                    positions=positions,
                    om=om,
                    pending_stats=pending_stats,
                )
                last_reconcile_metrics_sync = now_ts
            for event in events:
                sym = event.symbol.upper()
                features = dict(event.features or {})
                close_px = float(features.get("close") or 0.0)
                ts = pd.Timestamp(event.timestamp)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                day_key = _utc_day_key(ts)

                METRICS.bars_processed.inc(1)
                METRICS.last_bar_age.labels(symbol=sym).set(0)
                METRICS.update_strategy_symbol_ohlc(
                    strategy=strategy_name,
                    symbol=sym,
                    timeframe=tf,
                    values=features,
                )
                snap_age = provider.reader.latest_snapshot_age_seconds(
                    symbol=sym, timeframe=tf
                )
                if snap_age is not None:
                    METRICS.feature_bus_snapshot_age_seconds.labels(symbol=sym).set(
                        snap_age
                    )

                pos = positions.get(sym)
                if pos and close_px > 0:
                    if "profit_take_ladder" not in pos:
                        pos["profit_take_ladder"] = dict(budget.profit_take_ladder_cfg)
                    if chain_debug:
                        qty0 = float(pos.get("_qty_base", 0.0) or 0.0)
                        cost0 = float(pos.get("_entry_notional_usdt", 0.0) or 0.0)
                        mtm = (
                            (qty0 * close_px / cost0) if qty0 > 0 and cost0 > 0 else 0.0
                        )
                        trigger = resolve_min_profit_multiple(
                            sym, pos.get("profit_take_ladder") or {}
                        )
                        logger.info(
                            "[%s] sell-check qty=%.8f cost=%.2f close=%.6f mtm=%.3fx trigger=%.2fx last_sell_day=%s",
                            sym,
                            qty0,
                            cost0,
                            close_px,
                            mtm,
                            trigger,
                            str(pos.get("_profit_ladder_last_sell_day") or ""),
                        )
                    out = maybe_spot_simple_partial_sell(
                        pos,
                        price_close=close_px,
                        now=ts.to_pydatetime(),
                    )
                    if chain_debug and out is None:
                        logger.info("[%s] sell-check not triggered this bar", sym)
                    if out:
                        sell_qty, reason = out
                        if om.api is not None:
                            sell_qty = om.api.amount_to_precision(sym, sell_qty)
                            limits = om.api.get_market_limits(sym)
                            if sell_qty < float(limits.get("min_amount", 0.0) or 0.0):
                                sell_qty = 0.0
                        if sell_qty > 0.0:
                            sell_result = om.place_order(
                                symbol=sym,
                                side="sell",
                                order_type="market",
                                quantity=sell_qty,
                            )
                            payload = sell_result.payload or {}
                            st, filled_qty, _fill_quote, _avg = parse_ccxt_fill(payload)
                            actual_sell = apply_sell_fill_to_position(
                                pos,
                                fill_qty=float(
                                    filled_qty if filled_qty > 0 else sell_qty
                                ),
                                exit_price=close_px,
                            )
                            if actual_sell <= 0.0:
                                apply_partial_sell_to_position(
                                    pos,
                                    sell_qty=sell_qty,
                                    exit_price=close_px,
                                )
                            pos["_profit_ladder_last_sell_day"] = day_key
                            logger.info(
                                "[%s] partial sell qty=%.8f px=%.6f reason=%s",
                                sym,
                                sell_qty,
                                close_px,
                                reason,
                            )
                            METRICS.orders_total.labels(strategy=strategy_name).inc()
                            if stats_collector is not None:
                                try:
                                    stats_collector.record_order_placed(
                                        sym, strategy_name
                                    )
                                except Exception:
                                    pass
                            METRICS.record_strategy_event(
                                scope="spot",
                                strategy=strategy_name,
                                symbol=sym,
                                event="spot_partial_sell",
                                side="sell",
                                price=close_px,
                            )
                            if float(pos.get("_qty_base", 0.0) or 0.0) <= 0.0:
                                positions.pop(sym, None)
                        elif chain_debug:
                            logger.info(
                                "[%s] sell-check triggered but qty rounded/min-lot to zero",
                                sym,
                            )

                intents = strategy.decide(
                    features=features,
                    symbol=sym,
                    features_by_timeframe=event.features_by_timeframe,
                    bars=event.bars,
                )

                planned = 0.0
                intent = None
                if intents:
                    intent = intents[0]
                    if str(intent.action or "").upper() == "LONG":
                        deploy_today = ledger.deploy_today_usdt(day_key)
                        planned = _planned_buy_quote_usdt(
                            symbol=sym,
                            size_multiplier=float(intent.size_multiplier or 1.0),
                            budget=budget,
                            positions=positions,
                            deploy_today=deploy_today,
                            day_key=day_key,
                        )

                elig = collect_spot_new_buy_report(
                    symbol=sym,
                    ts=ts,
                    features=features,
                    strategy=strategy,
                    deploy_schedule_cfg=budget.deploy_schedule_cfg,
                    budget=budget,
                    positions=positions,
                    ledger=ledger,
                    day_key=day_key,
                    intents=intents,
                    om_shadow=om.shadow,
                    planned_usdt=planned,
                    free_usdt=free_usdt,
                )
                log_spot_new_buy_eligibility(elig)

                if stats_collector is not None:
                    try:
                        stats_collector.record_bar_processed(1)
                        has_long_intent = bool(intents) and (
                            str(intents[0].action or "").upper() == "LONG"
                        )
                        can_submit = bool(elig.get("can_submit_new_buy"))
                        blockers = elig.get("blockers") or []
                        blocker_str = (
                            ", ".join(str(b) for b in blockers if b)
                            if isinstance(blockers, (list, tuple))
                            else str(blockers or "")
                        )
                        stats_collector.record_strategy_eval(
                            sym,
                            strategy_name,
                            funnel_for_spot_decision(
                                has_intent=has_long_intent,
                                can_submit=can_submit,
                                blocker=blocker_str,
                            ),
                        )
                    except Exception:
                        logger.debug("spot funnel record skipped", exc_info=True)

                if not intents:
                    if chain_debug:
                        log_spot_no_intent(sym, strategy, features)
                    continue
                if intent is None:
                    continue
                if str(intent.action or "").upper() != "LONG":
                    if chain_debug:
                        logger.info(
                            "[%s] signal-check intent=%s skipped (only LONG supported)",
                            sym,
                            str(intent.action or ""),
                        )
                    continue
                METRICS.signals_total.labels(strategy=strategy_name).inc()
                if not elig.get("can_submit_new_buy"):
                    continue

                planned = float(elig.get("planned_usdt") or 0.0)
                if planned <= 0.0 or close_px <= 0.0:
                    continue

                order_type = budget.entry_order_type
                entry_px = close_px
                if order_type == "limit":
                    entry_px = close_px * (
                        1.0 - max(0.0, budget.entry_limit_offset_bps) / 10000.0
                    )
                qty = planned / max(entry_px, 1e-9)
                if om.api is not None:
                    qty = om.api.amount_to_precision(sym, qty)
                    if order_type == "limit":
                        entry_px = om.api.price_to_precision(sym, entry_px)
                    limits = om.api.get_market_limits(sym)
                    if qty < float(limits.get("min_amount", 0.0) or 0.0):
                        if chain_debug:
                            logger.info(
                                "[%s] buy-skip qty below min_amount: qty=%.8f min=%.8f",
                                sym,
                                qty,
                                float(limits.get("min_amount", 0.0) or 0.0),
                            )
                        continue
                    if planned < float(limits.get("min_cost", 0.0) or 0.0):
                        if chain_debug:
                            logger.info(
                                "[%s] buy-skip quote below min_cost: quote=%.4f min=%.4f",
                                sym,
                                planned,
                                float(limits.get("min_cost", 0.0) or 0.0),
                            )
                        continue
                if qty <= 0.0:
                    if chain_debug:
                        logger.info("[%s] buy-skip qty<=0 after precision", sym)
                    continue

                try:
                    buy_result = om.place_order(
                        symbol=sym,
                        side="buy",
                        order_type=order_type,
                        quantity=qty,
                        price=(entry_px if order_type == "limit" else None),
                    )
                except Exception as exc:
                    if _is_insufficient_funds(exc):
                        logger.warning(
                            "[%s] buy skipped: insufficient USDT (planned=%.2f free=%s): %s",
                            sym,
                            planned,
                            f"{free_usdt:.2f}" if free_usdt is not None else "n/a",
                            exc,
                        )
                        continue
                    raise
                pos = positions.get(sym) or new_position_shell(
                    sym, profit_take_ladder_cfg=budget.profit_take_ladder_cfg
                )
                positions[sym] = pos
                payload = buy_result.payload or {}
                st, filled_qty, fill_quote, _avg = parse_ccxt_fill(payload)
                if order_type == "market" or st in {"closed", "filled"}:
                    if filled_qty > 0.0 and fill_quote > 0.0:
                        _apply_buy_fill(
                            sym=sym,
                            positions=positions,
                            ledger=ledger,
                            budget=budget,
                            day_key=day_key,
                            fill_qty=filled_qty,
                            fill_quote=fill_quote,
                            filled_at=ts.isoformat(),
                        )
                    else:
                        logger.warning(
                            "[%s] market buy submitted but no fill in response status=%s",
                            sym,
                            st,
                        )
                else:
                    set_pending_buy(
                        pos,
                        local_order_id=buy_result.order_id,
                        exchange_order_id=buy_result.exchange_order_id,
                        client_order_id=buy_result.client_order_id,
                        quantity=qty,
                        price=entry_px,
                        quote_reserved=planned,
                        placed_at=ts.isoformat(),
                    )
                    om.update_order_record(
                        buy_result.order_id,
                        status=st or "open",
                        raw_json=(
                            json.dumps(
                                payload, ensure_ascii=True, separators=(",", ":")
                            )
                            if payload
                            else None
                        ),
                    )
                logger.info(
                    "[%s] buy intent=%s qty=%.8f px=%.6f quote=%.2f type=%s status=%s",
                    sym,
                    intent.action,
                    qty,
                    entry_px,
                    planned,
                    order_type,
                    st or buy_result.status,
                )
                if chain_debug:
                    logger.info(
                        "[%s] ledger after buy qty=%.8f deployed=%.2f pending_reserved=%.2f day_deploy=%.2f",
                        sym,
                        float(pos.get("_qty_base", 0.0) or 0.0),
                        float(pos.get("_spot_quote_deployed", 0.0) or 0.0),
                        float(
                            (pos.get("_pending_buy") or {}).get("quote_reserved", 0.0)
                            or 0.0
                        ),
                        ledger.deploy_today_usdt(day_key),
                    )
                METRICS.orders_total.labels(strategy=strategy_name).inc()
                if stats_collector is not None:
                    try:
                        stats_collector.record_order_placed(sym, strategy_name)
                    except Exception:
                        pass
                METRICS.record_strategy_event(
                    scope="spot",
                    strategy=strategy_name,
                    symbol=sym,
                    event="spot_buy",
                    side="buy",
                    price=float(entry_px),
                )

            ledger.save_positions(positions)
            pos_rows = []
            for sym, p in positions.items():
                qty = float((p or {}).get("_qty_base", 0.0) or 0.0)
                if qty <= 0.0:
                    continue
                notional = float((p or {}).get("_entry_notional_usdt", 0.0) or 0.0)
                pos_rows.append(
                    {
                        "symbol": sym,
                        "side": "long",
                        "qty": qty,
                        "notional_usdt": notional,
                    }
                )
            METRICS.update_position_metrics(
                scope="spot",
                strategy=strategy_name,
                positions=pos_rows,
            )
            now = time.time()
            if now - last_account_sync >= max(10.0, poll_seconds):
                if om.api is not None:
                    try:
                        bal = om.api.get_total_balances()
                        usdt = float(bal.get("USDT", 0.0) or 0.0)
                        holdings_usdt = sum(
                            float((p or {}).get("_entry_notional_usdt", 0.0) or 0.0)
                            for p in positions.values()
                            if float((p or {}).get("_qty_base", 0.0) or 0.0) > 0.0
                        )
                        equity_usdt = usdt + holdings_usdt
                        METRICS.account_balance.labels(type="total").set(usdt)
                        METRICS.account_balance.labels(type="available").set(usdt)
                        METRICS.account_balance.labels(type="margin").set(equity_usdt)
                        METRICS.account_update_success.labels(scope="spot").set(1)
                        METRICS.account_update_age_seconds.labels(scope="spot").set(0)
                    except Exception:
                        METRICS.account_update_success.labels(scope="spot").set(0)
                last_account_sync = now
            if funnel_flusher is not None:
                try:
                    funnel_flusher.maybe_flush(symbol="ALL")
                except Exception:
                    logger.debug("spot funnel flush skipped", exc_info=True)
            time.sleep(max(0.5, poll_seconds))
        except KeyboardInterrupt:
            logger.info("spot runner stopped by user")
            ledger.save_positions(positions)
            if funnel_flusher is not None:
                try:
                    funnel_flusher.force_flush(symbol="ALL")
                except Exception:
                    logger.debug("spot funnel final flush skipped", exc_info=True)
            return 0
        except Exception as exc:
            if _is_insufficient_funds(exc):
                logger.warning(
                    "spot runner: insufficient balance, no order sent: %s", exc
                )
            else:
                logger.exception("spot runner loop error: %s", exc)
            time.sleep(max(1.0, poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
