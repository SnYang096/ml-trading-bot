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
    spot_strategies_from_constitution,
)
from src.order_management.spot_binance_api import SpotBinanceAPI
from src.order_management.spot_order_manager import SpotOrderManager
from src.time_series_model.live.metrics_exporter import METRICS, start_metrics_server
from src.time_series_model.live.generic_live_strategy import GenericLiveStrategy
from src.time_series_model.live.spot_accum_simple import (
    apply_partial_sell_to_position,
    deploy_decay_multiplier,
    maybe_spot_simple_partial_sell,
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

    min_interval = max(
        _env_int("MLBOT_SPOT_MIN_ORDER_INTERVAL_MINUTES", 0),
        int(accumulation.get("min_order_interval_minutes", 0) or 0),
        int(exec_cons.get("min_order_interval_minutes", 0) or 0),
    )
    max_new = int(strategy_limits.get("max_new_entries_per_day", 1) or 1)

    return SpotBudgetConfig(
        equity_anchor_usdt=float(account.get("backtest_equity_usdt", 10000) or 10000),
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
        profit_take_ladder_cfg=ladder,
    )


def _utc_day_key(ts: Any) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.strftime("%Y-%m-%d")


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
    deployed_symbol = float(pos.get("_spot_quote_deployed", 0.0) or 0.0)
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
        float((p or {}).get("_spot_quote_deployed", 0.0) or 0.0)
        for p in positions.values()
    )
    daily_cap = float(budget.equity_anchor_usdt) * float(budget.max_daily_deploy_pct)

    remain_symbol = max(0.0, symbol_budget - deployed_symbol)
    remain_global = max(0.0, global_cap - global_deployed)
    remain_daily = max(0.0, daily_cap - deploy_today)
    return max(0.0, min(leg, remain_symbol, remain_global, remain_daily))


def _sync_positions_with_exchange(
    *,
    api: Optional[SpotBinanceAPI],
    symbols: Iterable[str],
    positions: Dict[str, Dict[str, Any]],
) -> None:
    if api is None:
        return
    balances = api.get_total_balances()
    for symbol in symbols:
        sym = str(symbol).upper()
        base = _symbol_base_asset(sym)
        qty_live = float(balances.get(base, 0.0) or 0.0)
        pos = positions.get(sym)
        if qty_live <= 0.0:
            if pos:
                del positions[sym]
            continue
        px = api.get_last_price(sym)
        if pos is None:
            cost = qty_live * max(px, 0.0)
            positions[sym] = {
                "symbol": sym,
                "_qty_base": qty_live,
                "_entry_notional_usdt": cost,
                "_spot_quote_deployed": cost,
                "structural_exit": "spot_simple_profit_ladder",
                "profit_take_ladder": {},
            }
            continue
        old_qty = float(pos.get("_qty_base", 0.0) or 0.0)
        old_cost = float(pos.get("_entry_notional_usdt", 0.0) or 0.0)
        if old_qty > 0.0:
            pos["_entry_notional_usdt"] = old_cost * (qty_live / old_qty)
            pos["_spot_quote_deployed"] = float(
                pos.get("_spot_quote_deployed", old_cost) or old_cost
            ) * (qty_live / old_qty)
        else:
            pos["_entry_notional_usdt"] = qty_live * max(px, 0.0)
            pos["_spot_quote_deployed"] = pos["_entry_notional_usdt"]
        pos["_qty_base"] = qty_live


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
    chain_debug = _env_bool("MLBOT_SPOT_CHAIN_DEBUG", True)

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
    provider = ClassicFeatureBusProvider(
        feature_bus_root=feature_bus_root,
        symbols=symbols,
        primary_timeframe=tf,
        timeframes=[tf],
        max_staleness_seconds=max_stale,
    )
    om = _build_spot_order_manager()
    metrics_port = _env_int("MLBOT_METRICS_PORT", 9193)
    start_metrics_server(port=metrics_port)
    METRICS.publish_dashboard_catalog(strategies=[strategy_name], symbols=symbols)
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
    _sync_positions_with_exchange(api=om.api, symbols=symbols, positions=positions)
    ledger.save_positions(positions)

    logger.info(
        "spot runner started: strategy=%s tf=%s symbols=%s shadow=%s bus=%s metrics_port=%s",
        strategy_name,
        tf,
        symbols,
        om.shadow,
        feature_bus_root,
        metrics_port,
    )
    if chain_debug:
        logger.info(
            "spot chain debug enabled: prints signal->order->ledger->sell checks"
        )
    last_account_sync = 0.0

    while True:
        try:
            METRICS.update_system_health()
            events = provider.poll()
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
                            om.place_order(
                                symbol=sym,
                                side="sell",
                                order_type="market",
                                quantity=sell_qty,
                            )
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
                            METRICS.record_strategy_event(
                                scope="trend",
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
                if not intents:
                    if chain_debug:
                        logger.info("[%s] signal-check no intent", sym)
                    continue
                intent = intents[0]
                if str(intent.action or "").upper() != "LONG":
                    if chain_debug:
                        logger.info(
                            "[%s] signal-check intent=%s skipped (only LONG supported)",
                            sym,
                            str(intent.action or ""),
                        )
                    continue
                METRICS.signals_total.labels(strategy=strategy_name).inc()
                if chain_debug:
                    logger.info(
                        "[%s] signal LONG confidence=%.4f size_mult=%.4f",
                        sym,
                        float(intent.confidence or 0.0),
                        float(intent.size_multiplier or 1.0),
                    )

                if ledger.buy_entries_today(day_key) >= budget.max_new_entries_per_day:
                    if chain_debug:
                        logger.info(
                            "[%s] buy-skip day limit reached: entries_today=%d max=%d",
                            sym,
                            ledger.buy_entries_today(day_key),
                            budget.max_new_entries_per_day,
                        )
                    continue

                last_buy_ts = None
                cur = positions.get(sym) or {}
                raw_last = cur.get("_last_buy_ts")
                if raw_last:
                    try:
                        last_buy_ts = pd.Timestamp(raw_last, tz="UTC")
                    except Exception:
                        last_buy_ts = None
                if last_buy_ts is not None and budget.min_order_interval_minutes > 0:
                    mins = (ts - last_buy_ts).total_seconds() / 60.0
                    if mins < float(budget.min_order_interval_minutes):
                        if chain_debug:
                            logger.info(
                                "[%s] buy-skip min_interval: mins_since_last=%.1f required=%d",
                                sym,
                                mins,
                                budget.min_order_interval_minutes,
                            )
                        continue

                deploy_today = ledger.deploy_today_usdt(day_key)
                planned = _planned_buy_quote_usdt(
                    symbol=sym,
                    size_multiplier=float(intent.size_multiplier or 1.0),
                    budget=budget,
                    positions=positions,
                    deploy_today=deploy_today,
                )
                if planned <= 0.0 or close_px <= 0.0:
                    if chain_debug:
                        logger.info(
                            "[%s] buy-skip planned_quote=%.4f close=%.6f",
                            sym,
                            planned,
                            close_px,
                        )
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

                om.place_order(
                    symbol=sym,
                    side="buy",
                    order_type=order_type,
                    quantity=qty,
                    price=(entry_px if order_type == "limit" else None),
                )
                pos = positions.get(sym) or {
                    "symbol": sym,
                    "_qty_base": 0.0,
                    "_entry_notional_usdt": 0.0,
                    "_spot_quote_deployed": 0.0,
                    "structural_exit": "spot_simple_profit_ladder",
                    "profit_take_ladder": dict(budget.profit_take_ladder_cfg),
                }
                pos["_qty_base"] = float(pos.get("_qty_base", 0.0) or 0.0) + float(qty)
                pos["_entry_notional_usdt"] = float(
                    pos.get("_entry_notional_usdt", 0.0) or 0.0
                ) + float(planned)
                pos["_spot_quote_deployed"] = float(
                    pos.get("_spot_quote_deployed", 0.0) or 0.0
                ) + float(planned)
                pos["_last_buy_ts"] = ts.isoformat()
                positions[sym] = pos
                ledger.add_buy(day_key, sym, planned)
                logger.info(
                    "[%s] buy intent=%s qty=%.8f px=%.6f quote=%.2f type=%s",
                    sym,
                    intent.action,
                    qty,
                    entry_px,
                    planned,
                    order_type,
                )
                if chain_debug:
                    logger.info(
                        "[%s] ledger after buy qty=%.8f deployed=%.2f cost=%.2f day_deploy=%.2f",
                        sym,
                        float(pos.get("_qty_base", 0.0) or 0.0),
                        float(pos.get("_spot_quote_deployed", 0.0) or 0.0),
                        float(pos.get("_entry_notional_usdt", 0.0) or 0.0),
                        ledger.deploy_today_usdt(day_key),
                    )
                METRICS.orders_total.labels(strategy=strategy_name).inc()
                METRICS.record_strategy_event(
                    scope="trend",
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
                scope="trend",
                strategy=strategy_name,
                positions=pos_rows,
            )
            now = time.time()
            if now - last_account_sync >= max(10.0, poll_seconds):
                if om.api is not None:
                    try:
                        bal = om.api.get_total_balances()
                        usdt = float(bal.get("USDT", 0.0) or 0.0)
                        METRICS.account_balance.labels(type="total").set(usdt)
                        METRICS.account_balance.labels(type="available").set(usdt)
                        METRICS.account_balance.labels(type="margin").set(0.0)
                        METRICS.account_update_success.labels(scope="trend").set(1)
                        METRICS.account_update_age_seconds.labels(scope="trend").set(0)
                    except Exception:
                        METRICS.account_update_success.labels(scope="trend").set(0)
                last_account_sync = now
            time.sleep(max(0.5, poll_seconds))
        except KeyboardInterrupt:
            logger.info("spot runner stopped by user")
            ledger.save_positions(positions)
            return 0
        except Exception as exc:
            logger.exception("spot runner loop error: %s", exc)
            time.sleep(max(1.0, poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
