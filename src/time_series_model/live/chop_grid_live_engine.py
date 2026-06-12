"""Dry-run live engine for the standalone chop_grid strategy.

The engine is intentionally independent from GenericLiveStrategy because a grid
owns multiple limit orders and inventory levels, not one TradeIntent position.
It returns desired order actions and leaves exchange-specific adapters outside.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.config.multileg_config import load_multileg_effective_config
from src.features.semantic_chop import resolve_semantic_chop
from src.config.strategy_layout import resolve_strategy_config_input
from src.order_management.grid_execution_adapter import (
    GridExecutionResult,
    derive_multileg_client_order_id,
)
from src.order_management.multi_leg_reconciliation import (
    LocalOrderSnapshot,
    LocalPositionSnapshot,
    ReconciliationReport,
)
from src.time_series_model.grid.chop_grid_engine import GridEngineConfig
from src.time_series_model.live.regime_box_prefilter import stable_box_blocks_chop_entry

logger = logging.getLogger(__name__)


@dataclass
class GridOrder:
    order_id: str
    symbol: str
    side: str
    level: int
    price: float
    quantity: float
    status: str = "pending"
    created_at: str = ""
    exchange_order_id: str = ""
    client_order_id: str = ""
    filled_quantity: float = 0.0


@dataclass
class GridPosition:
    symbol: str
    side: str
    level: int
    entry_price: float
    quantity: float
    entry_time: str
    leg_id: str = ""
    protection_order_ids: List[str] = field(default_factory=list)
    # Original entry fill qty; kept through exchange sync for protection sizing.
    entry_quantity: float = 0.0


@dataclass
class GridState:
    grid_id: str = ""
    symbol: str = ""
    active: bool = False
    center: float = 0.0
    spacing: float = 0.0
    realized_pnl: float = 0.0
    pending_orders: List[GridOrder] = field(default_factory=list)
    inventory: List[GridPosition] = field(default_factory=list)
    last_timestamp: str = ""
    current_regime: str = "idle"
    last_reconciliation_ok: bool = True
    last_reconciliation_issues: List[str] = field(default_factory=list)
    # Completed round-trips per level key (L1..Ln, S1..Sn) within active grid segment.
    level_replenish_count: Dict[str, int] = field(default_factory=dict)
    # leg_ids for which a dust market_exit was already queued (dedup across cycles).
    pending_dust_exits: List[str] = field(default_factory=list)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _grid_position_from_state(row: Mapping[str, Any]) -> GridPosition:
    data = dict(row)
    qty = _as_float(data.get("quantity"))
    data["entry_quantity"] = _as_float(data.get("entry_quantity"), qty)
    return GridPosition(**data)


def _parse_leg_level_side(leg_id: str) -> tuple[int, str]:
    text = str(leg_id or "")
    for level in range(1, 20):
        if text.endswith(f"_L{level}"):
            return level, "LONG"
        if text.endswith(f"_S{level}"):
            return level, "SHORT"
    if "_L" in text.upper():
        return 0, "LONG"
    if "_S" in text.upper():
        return 0, "SHORT"
    return 0, "LONG"


def _normalize_entry_leg_id(leg_hint: str) -> str:
    text = str(leg_hint or "").strip()
    if text.endswith("_tp"):
        return text[: -len("_tp")]
    if text.endswith("_sl"):
        return text[: -len("_sl")]
    return text


_ENTRY_LEG_RE = re.compile(r"_(?:L|S)\d+(?:_r\d+)?$")


def _is_entry_leg_id(leg_id: str) -> bool:
    """True for entry leg ids like ``..._L1`` / ``..._S2`` / ``..._L1_r3``.

    Guards the late-fill recovery path against non-entry suffixes (``_dust``,
    stray protection ids) that must never be re-ingested as inventory.
    """
    return bool(_ENTRY_LEG_RE.search(str(leg_id or "")))


def _normalize_symbol(value: Any) -> str:
    raw = str(value or "").upper().strip()
    if not raw:
        return ""
    if "/" in raw:
        base, rest = raw.split("/", 1)
        quote = rest.split(":", 1)[0]
        return f"{base}{quote}"
    return raw.split(":", 1)[0]


def _exchange_position_quantity(row: Mapping[str, Any]) -> float:
    return abs(
        _as_float(
            row.get("size")
            or row.get("quantity")
            or row.get("contracts")
            or row.get("position_amount")
            or row.get("positionAmt")
        )
    )


def _exchange_order_keys(order: Mapping[str, Any]) -> set[str]:
    info = order.get("info") if isinstance(order.get("info"), dict) else {}
    return {
        str(
            order.get("order_id")
            or order.get("orderId")
            or order.get("id")
            or info.get("orderId")
            or ""
        ),
        str(
            order.get("client_order_id")
            or order.get("clientOrderId")
            or info.get("clientOrderId")
            or ""
        ),
    } - {""}


def _open_exchange_client_order_ids(
    open_orders: Iterable[Mapping[str, Any]],
) -> set[str]:
    out: set[str] = set()
    for order in open_orders:
        info = order.get("info") if isinstance(order.get("info"), dict) else {}
        for key in (
            order.get("client_order_id"),
            order.get("clientOrderId"),
            info.get("clientOrderId"),
            info.get("clientAlgoId"),
        ):
            cid = str(key or "").strip()
            if cid:
                out.add(cid)
    return out


def _load_grid_config(path: str | Path) -> GridEngineConfig:
    cfg_path = Path(path)
    config_dir, profile_path, engine_path = resolve_strategy_config_input(cfg_path)
    obj = load_multileg_effective_config(
        config_dir=config_dir,
        strategy_type="grid",
        profile_path=profile_path,
        engine_path=engine_path,
    )
    regime = obj.get("regime", {}) or {}
    inv = obj.get("inventory", {}) or {}
    spacing = inv.get("spacing", {}) or {}
    risk = obj.get("risk", {}) or {}
    live = obj.get("live", {}) or {}
    expected_costs = live.get("expected_costs", {}) if isinstance(live, dict) else {}
    if not isinstance(expected_costs, dict):
        expected_costs = {}
    return GridEngineConfig(
        box_window=int(regime.get("box_window", 120)),
        entry_chop_min=float(
            regime.get("entry_min", regime.get("entry_chop_min", 0.40))
        ),
        exit_chop_below=float(
            regime.get("exit_below", regime.get("exit_chop_below", 0.25))
        ),
        min_segment_bars=int(risk.get("min_segment_bars", 6)),
        max_segment_bars=int(risk.get("max_segment_bars", 120)),
        grid_atr_mult=float(spacing.get("atr_mult", 0.50)),
        grid_min_pct=float(spacing.get("min_pct", 0.004)),
        max_levels_per_side=int(inv.get("max_levels_per_side", 3)),
        tp_spacing_mult=float(
            risk.get("tp_spacing_mult", inv.get("tp_spacing_mult", 1.0))
        ),
        fee_bps=float(expected_costs.get("fee_bps", risk.get("fee_bps", 4.0))),
        maker_fee_bps=float(
            expected_costs.get(
                "maker_fee_bps", risk.get("maker_fee_bps", risk.get("fee_bps", 4.0))
            )
        ),
        taker_fee_bps=float(
            expected_costs.get(
                "taker_fee_bps", risk.get("taker_fee_bps", risk.get("fee_bps", 4.0))
            )
        ),
        forced_exit_slippage_bps=float(
            expected_costs.get(
                "forced_exit_slippage_bps",
                risk.get("forced_exit_slippage_bps", 0.0),
            )
        ),
        funding_cost_bps_per_8h=float(
            expected_costs.get(
                "funding_cost_bps_per_8h",
                risk.get("funding_cost_bps_per_8h", 0.0),
            )
        ),
        max_loss_per_grid=risk.get("max_loss_per_grid", 0.03),
        max_open_levels_total=risk.get("max_open_levels_total", 6),
        max_replenish_per_level_per_segment=_parse_max_replenish(
            inv.get("max_replenish_per_level_per_segment")
        ),
    )


def _parse_max_replenish(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "null", "none"}:
        return None
    return int(raw)


def _level_side_key(level: int, side: str) -> str:
    s = str(side).upper()
    prefix = "L" if s in {"LONG", "BUY"} else "S"
    return f"{prefix}{int(level)}"


class ChopGridLiveEngine:
    """Dry-run grid engine that produces place/cancel/market_exit actions.

    Deployment packages under ``live/highcap/config/strategies`` intentionally keep
    only ``meta.yaml`` plus ``archetypes/``. Runtime mode, state paths and adapters
    come from the live runner CLI/env, not from research YAML.
    """

    def __init__(
        self,
        *,
        config_path: str | Path = "live/highcap/config/strategies/chop_grid",
        state_path: str | Path = "results/chop_grid/live_state.json",
        level_notional: float = 1.0,
        metrics_strategy: str = "",
        bar_simulation: bool = True,
    ) -> None:
        self.config_path = Path(config_path)
        self.state_path = Path(state_path)
        cfg_path = Path(config_path)
        config_dir, profile_path, engine_path = resolve_strategy_config_input(cfg_path)
        self._multileg_cfg = load_multileg_effective_config(
            config_dir=config_dir,
            strategy_type="grid",
            profile_path=profile_path,
            engine_path=engine_path,
        )
        self.regime = self._multileg_cfg.get("regime", {}) or {}
        self._prefilter_rules = self._multileg_cfg.get("rules", []) or []
        self.cfg = _load_grid_config(self.config_path)
        self.level_notional = float(level_notional)
        self.state = self.load_state()
        self._pending_actions: List[Dict[str, Any]] = []
        self.metrics_strategy = str(metrics_strategy or "")
        self.bar_simulation = bool(bar_simulation)
        self._live_exchange_has_activity = False

    def _emit_chop_bar_outcome(self, symbol: str, *, outcome: str) -> None:
        from src.order_management.hedge_engine_metrics import (
            record_multi_leg_engine_bar_outcome,
        )

        if not self.metrics_strategy:
            return
        record_multi_leg_engine_bar_outcome(
            metrics_strategy=self.metrics_strategy,
            symbol=symbol,
            engine="chop_grid",
            outcome=outcome,
        )

    def load_state(self) -> GridState:
        if not self.state_path.exists():
            return GridState()
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        return GridState(
            grid_id=str(raw.get("grid_id", "")),
            symbol=str(raw.get("symbol", "")),
            active=bool(raw.get("active", False)),
            center=_as_float(raw.get("center")),
            spacing=_as_float(raw.get("spacing")),
            realized_pnl=_as_float(raw.get("realized_pnl")),
            pending_orders=[GridOrder(**o) for o in raw.get("pending_orders", [])],
            inventory=[_grid_position_from_state(p) for p in raw.get("inventory", [])],
            last_timestamp=str(raw.get("last_timestamp", "")),
            current_regime=str(raw.get("current_regime", "idle")),
            last_reconciliation_ok=bool(raw.get("last_reconciliation_ok", True)),
            last_reconciliation_issues=[
                str(x) for x in raw.get("last_reconciliation_issues", [])
            ],
            level_replenish_count={
                str(k): int(v)
                for k, v in (raw.get("level_replenish_count") or {}).items()
            },
            pending_dust_exits=[
                str(x) for x in (raw.get("pending_dust_exits") or []) if str(x)
            ],
        )

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(asdict(self.state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def local_order_snapshots(self) -> List[LocalOrderSnapshot]:
        """Expose pending orders for exchange reconciliation."""
        snapshots = [
            LocalOrderSnapshot(
                order_id=o.order_id,
                symbol=o.symbol,
                side=o.side,
                quantity=max(0.0, o.quantity - o.filled_quantity),
                price=o.price,
                exchange_order_id=o.exchange_order_id,
                client_order_id=o.client_order_id,
            )
            for o in self.state.pending_orders
            if o.status not in {"filled", "canceled"}
        ]
        for pos in self.state.inventory:
            for protection_id in pos.protection_order_ids:
                pid = str(protection_id or "").strip()
                if not pid:
                    continue
                snapshots.append(
                    LocalOrderSnapshot(
                        order_id=pid,
                        symbol=pos.symbol,
                        side="SELL" if str(pos.side).upper() == "LONG" else "BUY",
                        quantity=pos.quantity,
                        price=self._tp_price_for_position(pos) or 0.0,
                        exchange_order_id=pid,
                    )
                )
        return snapshots

    def local_position_snapshots(self) -> List[LocalPositionSnapshot]:
        """Expose inventory for exchange position reconciliation."""
        return [
            LocalPositionSnapshot(
                symbol=p.symbol,
                side=p.side,
                quantity=p.quantity,
            )
            for p in self.state.inventory
        ]

    def is_stale_active_ghost(self) -> bool:
        """True when ``active`` is set but nothing real remains on this segment."""
        return bool(
            self.state.active
            and not self.state.pending_orders
            and not self.state.inventory
            and not self._live_exchange_has_activity
        )

    def clear_stale_active_if_ghost(self) -> bool:
        """Drop a ghost segment so it cannot block the concurrency cap.

        Called from the shared gate before slot accounting so symbols that have
        not received a bar yet this cycle still release stale ``active`` flags.
        """
        if not self.is_stale_active_ghost():
            return False
        logger.warning(
            "chop_grid stale active state reset: symbol=%s grid_id=%s",
            self.state.symbol,
            self.state.grid_id,
        )
        self.state.active = False
        self.state.current_regime = "idle"
        self.save_state()
        gate = getattr(self, "_concurrency_gate", None)
        if gate is not None:
            gate.notify_deactivation(self.state.symbol, "chop_grid")
        return True

    def holds_real_grid_slot(self) -> bool:
        """True iff this engine's active segment really occupies a concurrency slot.

        An ``active`` segment carrying no pending orders, no inventory and no live
        exchange activity is a ghost (cleared via :meth:`clear_stale_active_if_ghost`),
        so it must not count toward ``max_concurrent_multi_leg_symbols`` and block
        other symbols from starting.
        """
        if not bool(getattr(self.state, "active", False)):
            return False
        return bool(
            self.state.pending_orders
            or self.state.inventory
            or self._live_exchange_has_activity
        )

    def on_execution_results(self, results: Iterable[GridExecutionResult]) -> None:
        """Persist exchange/client ids returned by the execution adapter."""
        for result in results:
            raw = result.raw or {}
            local_id = str(raw.get("local_order_id") or raw.get("order_id") or "")
            if result.action == "place":
                order = self._find_order(
                    local_id=local_id, client_id=result.client_order_id
                )
                if order is not None:
                    order.exchange_order_id = result.order_id
                    order.client_order_id = result.client_order_id
                    order.status = str(result.status or "submitted")
            elif result.action == "cancel":
                order = self._find_order(
                    local_id=local_id,
                    exchange_id=result.order_id,
                    client_id=result.client_order_id,
                )
                if order is not None and result.status in {"canceled", "shadow"}:
                    order.status = "canceled"
            elif result.action == "place_protection":
                raw = result.raw or {}
                pos = self._find_position(str(raw.get("leg_id") or ""))
                if pos is not None and result.order_id:
                    pos.protection_order_ids.append(result.order_id)
            elif result.action == "market_exit":
                leg_id = str(raw.get("leg_id") or "")
                if not leg_id:
                    order_id = str(
                        raw.get("local_order_id") or raw.get("order_id") or ""
                    )
                    if order_id.endswith("_dust"):
                        leg_id = order_id[: -len("_dust")]
                status = str(result.status or "").lower()
                if status in {"skipped_no_position", "filled", "closed"}:
                    self._clear_pending_dust_exit(leg_id)
        self.state.pending_orders = [
            o for o in self.state.pending_orders if o.status != "canceled"
        ]
        self.save_state()

    @staticmethod
    def _ghost_pending_ttl_seconds() -> float:
        """TTL after which a local-only pending order that never received an
        exchange/client id is treated as a ghost and pruned. ``<= 0`` disables."""
        raw = os.getenv("MLBOT_CHOP_GRID_GHOST_PENDING_TTL_S", "1800").strip()
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 1800.0

    @staticmethod
    def _parse_ts(value: Any) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    def _order_age_seconds(
        self, order: Optional[GridOrder], now_value: Any
    ) -> Optional[float]:
        if order is None:
            return None
        created = self._parse_ts(order.created_at)
        now = self._parse_ts(now_value)
        if created is None or now is None:
            return None
        # created/now share the engine's bar-timestamp source; if one is tz-aware
        # and the other naive, fromisoformat would mismatch -> guard the subtraction.
        if (created.tzinfo is None) != (now.tzinfo is None):
            return None
        return (now - created).total_seconds()

    def on_reconciliation_report(self, report: ReconciliationReport) -> None:
        missing_ids = {str(o.order_id) for o in report.missing_exchange_orders}
        if missing_ids:
            prunable_ids = {
                str(o.order_id)
                for o in self.state.pending_orders
                if str(o.order_id) in missing_ids
                and bool(o.exchange_order_id or o.client_order_id)
            }
            ttl = self._ghost_pending_ttl_seconds()
            now_value = self.state.last_timestamp
            kept_ids: List[str] = []
            expired_local_only: List[str] = []
            for oid in sorted(missing_ids - prunable_ids):
                age = self._order_age_seconds(self._find_order(local_id=oid), now_value)
                if ttl > 0 and age is not None and age >= ttl:
                    expired_local_only.append(oid)
                else:
                    kept_ids.append(oid)
            if expired_local_only:
                expired_set = set(expired_local_only)
                before = len(self.state.pending_orders)
                self.state.pending_orders = [
                    o
                    for o in self.state.pending_orders
                    if str(o.order_id) not in expired_set
                ]
                logger.warning(
                    "chop_grid: pruned %d stale local-only pending order(s) "
                    "(never mapped to exchange within %.0fs): %s",
                    before - len(self.state.pending_orders),
                    ttl,
                    expired_local_only[:12],
                )
            if kept_ids:
                logger.info(
                    "chop_grid: keep %d local-only missing order(s) pending "
                    "(no exchange/client id): %s",
                    len(kept_ids),
                    kept_ids[:12],
                )
            if prunable_ids:
                before = len(self.state.pending_orders)
                self.state.pending_orders = [
                    o
                    for o in self.state.pending_orders
                    if str(o.order_id) not in prunable_ids
                ]
                dropped = before - len(self.state.pending_orders)
                if dropped:
                    logger.warning(
                        "chop_grid: pruned %d mapped pending order(s) missing on exchange "
                        "(reconcile): %s",
                        dropped,
                        sorted(prunable_ids)[:12],
                    )
            missing_protection_ids = {
                str(pid)
                for pos in self.state.inventory
                for pid in pos.protection_order_ids
                if str(pid) in missing_ids
            }
            if missing_protection_ids:
                for pos in self.state.inventory:
                    pos.protection_order_ids = [
                        str(pid)
                        for pid in pos.protection_order_ids
                        if str(pid) not in missing_protection_ids
                    ]
                logger.warning(
                    "chop_grid: pruned %d stale protection order id(s) missing on exchange "
                    "(reconcile): %s",
                    len(missing_protection_ids),
                    sorted(missing_protection_ids)[:12],
                )
        issues: List[str] = []
        issues.extend(
            f"missing_exchange_order:{o.order_id}"
            for o in report.missing_exchange_orders
        )
        issues.extend(
            "orphan_exchange_order:"
            f"{o.get('order_id') or o.get('orderId') or o.get('client_order_id')}"
            for o in report.orphan_exchange_orders
        )
        issues.extend(
            f"position_mismatch:{m.symbol}:{m.side}:{m.local_quantity}->{m.exchange_quantity}"
            for m in report.position_mismatches
        )
        self.state.last_reconciliation_ok = report.ok
        self.state.last_reconciliation_issues = issues
        self.save_state()

    def on_execution_report(self, report: Dict[str, Any]) -> None:
        """Apply normalized user-stream fill updates to local pending inventory."""
        if self._handle_protection_fill(report):
            self.save_state()
            return
        order = self._find_order(
            exchange_id=str(report.get("order_id") or ""),
            client_id=str(report.get("client_order_id") or ""),
        )
        leg_hint = _normalize_entry_leg_id(
            str(report.get("leg_id") or report.get("local_order_id") or "")
        )
        if order is None and leg_hint:
            order = self._find_order(local_id=leg_hint)
        if order is None:
            if self._ingest_late_entry_fill(report, leg_hint):
                self.save_state()
            return
        status = str(report.get("status") or "").upper()
        filled_qty = _as_float(report.get("filled_qty"), order.filled_quantity)
        last_px = _as_float(report.get("last_filled_price"), order.price)
        order.filled_quantity = max(
            order.filled_quantity, min(filled_qty, order.quantity)
        )
        order.status = status.lower() if status else order.status
        if status == "FILLED" or order.filled_quantity >= order.quantity:
            pos_side = "LONG" if order.side == "BUY" else "SHORT"
            fill_qty = order.filled_quantity or order.quantity
            self.state.inventory.append(
                GridPosition(
                    symbol=order.symbol,
                    side=pos_side,
                    level=order.level,
                    entry_price=last_px if last_px > 0 else order.price,
                    quantity=fill_qty,
                    entry_quantity=fill_qty,
                    entry_time=str(
                        report.get("trade_time") or self.state.last_timestamp
                    ),
                    leg_id=order.order_id,
                )
            )
            self._pending_actions.extend(
                self._protection_actions(
                    order_id=order.order_id,
                    pos=self.state.inventory[-1],
                    timestamp=str(
                        report.get("trade_time") or self.state.last_timestamp
                    ),
                )
            )
            self.state.pending_orders = [
                o for o in self.state.pending_orders if o.order_id != order.order_id
            ]
        self.save_state()

    def _ingest_late_entry_fill(self, report: Dict[str, Any], leg_hint: str) -> bool:
        """Recover inventory + protection when a fill arrives after pending prune."""
        leg_id = _normalize_entry_leg_id(leg_hint)
        if not leg_id or not _is_entry_leg_id(leg_id):
            return False
        if self._find_position(leg_id) is not None:
            return False
        status = str(report.get("status") or "").upper()
        if status != "FILLED":
            return False
        filled_qty = _as_float(report.get("filled_qty"), 0.0)
        if filled_qty <= 0:
            return False
        level, pos_side = _parse_leg_level_side(leg_id)
        last_px = _as_float(
            report.get("last_filled_price") or report.get("avg_price"), 0.0
        )
        symbol = _normalize_symbol(report.get("symbol") or self.state.symbol)
        if not symbol:
            return False
        pos = GridPosition(
            symbol=symbol,
            side=pos_side,
            level=level,
            entry_price=last_px,
            quantity=filled_qty,
            entry_quantity=filled_qty,
            entry_time=str(report.get("trade_time") or self.state.last_timestamp),
            leg_id=leg_id,
        )
        self.state.inventory.append(pos)
        self._pending_actions.extend(
            self._protection_actions(
                order_id=leg_id,
                pos=pos,
                timestamp=str(report.get("trade_time") or self.state.last_timestamp),
            )
        )
        logger.info(
            "chop_grid late entry fill ingested: symbol=%s leg_id=%s qty=%.8f",
            symbol,
            leg_id,
            filled_qty,
        )
        return True

    def pop_pending_actions(self) -> List[Dict[str, Any]]:
        actions = list(self._pending_actions)
        self._pending_actions.clear()
        return actions

    def actions_ensure_protection(
        self,
        *,
        exchange_positions: Iterable[Mapping[str, Any]],
        exchange_orders: Iterable[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Place missing reduce-only TP/SL for open exchange legs (live reconcile)."""
        if self.bar_simulation or not self.state.active or self.state.spacing <= 0:
            return []
        sym = str(self.state.symbol or "").upper()
        if not sym:
            return []

        before_snapshot = [
            (str(p.symbol or "").upper(), str(p.side).upper(), float(p.quantity or 0.0))
            for p in self.state.inventory
        ]
        self._sync_inventory_from_exchange(exchange_positions, symbol=sym)
        after_snapshot = [
            (str(p.symbol or "").upper(), str(p.side).upper(), float(p.quantity or 0.0))
            for p in self.state.inventory
        ]
        inventory_changed = before_snapshot != after_snapshot
        open_orders = [dict(o) for o in exchange_orders]
        actions: List[Dict[str, Any]] = []
        ts = self.state.last_timestamp or ""
        side_remaining = self._exchange_side_qty_map(exchange_positions, symbol=sym)
        for pos in self.state.inventory:
            if str(pos.symbol or "").upper() != sym:
                continue
            pos_side = str(pos.side).upper()
            leg_qty = float(pos.quantity or 0.0)
            alloc_qty = min(leg_qty, side_remaining.get(pos_side, 0.0))
            side_remaining[pos_side] = max(
                0.0, side_remaining.get(pos_side, 0.0) - alloc_qty
            )
            if alloc_qty <= 0:
                continue
            covered_qty = self._open_tp_covered_qty(pos, open_orders)
            need_qty = alloc_qty
            if need_qty <= 0 or covered_qty >= need_qty * 0.99:
                continue
            remaining_qty = max(0.0, need_qty - covered_qty)
            ref_px = self._tp_price_for_position(pos) or pos.entry_price
            if self._is_dust_notional(remaining_qty, ref_px):
                if not self._should_emit_dust_exit(
                    pos, remaining_qty, ref_px, ts, open_orders
                ):
                    continue
                self._mark_pending_dust_exit(pos.leg_id)
                actions.append(
                    {
                        "action": "market_exit",
                        "order_id": f"{pos.leg_id}_dust",
                        "leg_id": pos.leg_id,
                        "symbol": pos.symbol,
                        "side": pos.side,
                        "level": pos.level,
                        "quantity": remaining_qty,
                        "exit_price": ref_px,
                        "reason": "dust_below_min_notional",
                        "timestamp": ts,
                    }
                )
                continue
            action_pos = GridPosition(
                symbol=pos.symbol,
                side=pos.side,
                level=pos.level,
                entry_price=pos.entry_price,
                quantity=remaining_qty,
                entry_quantity=float(pos.entry_quantity or pos.quantity or 0.0),
                entry_time=pos.entry_time,
                leg_id=pos.leg_id,
                protection_order_ids=list(pos.protection_order_ids),
            )
            new_actions = self._protection_actions(
                order_id=pos.leg_id,
                pos=action_pos,
                timestamp=ts,
            )
            open_client_ids = _open_exchange_client_order_ids(open_orders)
            filtered_actions = []
            for action in new_actions:
                if derive_multileg_client_order_id(action) in open_client_ids:
                    continue
                if self._is_dust_notional(
                    float(action.get("quantity") or 0.0),
                    float(action.get("price") or action.get("trigger_price") or ref_px),
                ):
                    continue
                if str(action.get("protection_type") or "") == "take_profit":
                    # Catch-up protection must close if price already crossed TP.
                    action["post_only"] = False
                    action["time_in_force"] = "GTC"
                filtered_actions.append(action)
            actions.extend(filtered_actions)
        if actions:
            logger.info(
                "chop_grid ensure_protection: symbol=%s inventory=%d actions=%d",
                sym,
                len(self.state.inventory),
                len(actions),
            )
            self.save_state()
        elif inventory_changed:
            self.save_state()
        return actions

    def sync_live_exchange_state(
        self,
        *,
        exchange_positions: Iterable[Mapping[str, Any]],
        exchange_orders: Iterable[Mapping[str, Any]],
    ) -> None:
        """Use exchange truth to avoid opening a fresh grid over live exposure."""
        self._live_exchange_has_activity = False
        if self.bar_simulation:
            return
        sym = str(self.state.symbol or "").upper()
        if not sym:
            # If state was lost, infer symbol from exchange rows later in the daemon.
            sym = ""
        open_grid_orders = [
            dict(o)
            for o in exchange_orders
            if self._is_chop_grid_exchange_order(o)
            and (not sym or _normalize_symbol(o.get("symbol")) == sym)
        ]
        positions = [
            dict(p)
            for p in exchange_positions
            if (not sym or _normalize_symbol(p.get("symbol")) == sym)
            and _exchange_position_quantity(p) > 0
        ]
        has_local_chop = bool(self.state.inventory or self.state.pending_orders)
        # Do not treat foreign legs (e.g. trend_scalp) as chop-owned exposure.
        self._live_exchange_has_activity = bool(open_grid_orders) or (
            bool(positions) and has_local_chop
        )
        if not self._live_exchange_has_activity:
            return
        if not self.state.active:
            if not open_grid_orders:
                return
            logger.warning(
                "chop_grid live exchange activity blocks new grid: symbol=%s "
                "open_orders=%d positions=%d",
                sym or "unknown",
                len(open_grid_orders),
                len(positions),
            )
            self.state.active = True
            self.state.current_regime = "chop_grid"
            if not self.state.symbol and positions:
                self.state.symbol = _normalize_symbol(positions[0].get("symbol"))
            if not self.state.symbol and open_grid_orders:
                self.state.symbol = _normalize_symbol(open_grid_orders[0].get("symbol"))
        if self.state.symbol:
            self._sync_inventory_from_exchange(
                positions,
                symbol=str(self.state.symbol).upper(),
            )
        self.save_state()

    def _sync_inventory_from_exchange(
        self,
        exchange_positions: Iterable[Mapping[str, Any]],
        *,
        symbol: str,
    ) -> None:
        """Align local inventory with hedge exchange truth.

        Hedge mode reports at most one position per (symbol, positionSide). We
        aggregate per side, then:
          * **Prune** local legs whose side has no exchange position.
          * **Cap** total local qty per side to exchange qty when local is over.
          * **Seed** a synthetic leg when exchange has qty but local has none on
            that side (covers user-stream missed fills).

        Individual leg entry prices are trusted as-is to avoid duplicate legs
        from Binance's avg-price rounding.
        """
        sym = str(symbol).upper()
        exchange_by_side: Dict[str, Dict[str, float]] = {}
        for raw in exchange_positions:
            row = dict(raw)
            if _normalize_symbol(row.get("symbol")) != sym:
                continue
            qty = _exchange_position_quantity(row)
            if qty <= 0:
                continue
            side_raw = str(row.get("side") or row.get("positionSide") or "").lower()
            pos_side = "LONG" if side_raw in {"long", "buy"} else "SHORT"
            entry = _as_float(row.get("entry_price") or row.get("entryPrice"))
            if entry <= 0:
                continue
            bucket = exchange_by_side.setdefault(pos_side, {"qty": 0.0, "entry": entry})
            bucket["qty"] += qty

        other_symbol_legs = [
            p for p in self.state.inventory if str(p.symbol or "").upper() != sym
        ]
        symbol_legs = [
            p for p in self.state.inventory if str(p.symbol or "").upper() == sym
        ]

        kept_after_prune: List[GridPosition] = []
        for pos in symbol_legs:
            pos_side = str(pos.side).upper()
            if pos_side not in exchange_by_side:
                logger.info(
                    "chop_grid prune stale inventory: symbol=%s side=%s leg_id=%s "
                    "(no exchange position)",
                    sym,
                    pos_side,
                    pos.leg_id,
                )
                continue
            kept_after_prune.append(pos)

        capped: List[GridPosition] = []
        for side in ("LONG", "SHORT"):
            side_legs = [p for p in kept_after_prune if str(p.side).upper() == side]
            if not side_legs:
                continue
            ex_qty = exchange_by_side.get(side, {}).get("qty", 0.0)
            local_total = sum(float(p.quantity or 0.0) for p in side_legs)
            if local_total <= ex_qty * 1.01:
                capped.extend(side_legs)
                continue
            running = 0.0
            for pos in side_legs:
                leg_qty = float(pos.quantity or 0.0)
                if running + leg_qty <= ex_qty * 1.01:
                    capped.append(pos)
                    running += leg_qty
                    continue
                remaining = max(0.0, ex_qty - running)
                if remaining > 0:
                    pos.quantity = remaining
                    capped.append(pos)
                    running = ex_qty
                else:
                    logger.info(
                        "chop_grid prune excess inventory: symbol=%s side=%s "
                        "leg_id=%s local_qty=%.8f exchange_qty=%.8f",
                        sym,
                        side,
                        pos.leg_id,
                        leg_qty,
                        ex_qty,
                    )

        sides_with_local = {str(p.side).upper() for p in capped}
        for side, bucket in exchange_by_side.items():
            if side in sides_with_local:
                continue
            entry = float(bucket["entry"])
            qty = float(bucket["qty"])
            leg_id = self._match_leg_id_for_fill(side, entry)
            capped.append(
                GridPosition(
                    symbol=sym,
                    side=side,
                    level=0,
                    entry_price=entry,
                    quantity=qty,
                    entry_quantity=qty,
                    entry_time=self.state.last_timestamp,
                    leg_id=leg_id,
                )
            )

        self.state.inventory = other_symbol_legs + capped
        active_leg_ids = {str(p.leg_id) for p in self.state.inventory if p.leg_id}
        self.state.pending_dust_exits = [
            leg_id
            for leg_id in self.state.pending_dust_exits
            if leg_id in active_leg_ids
        ]

    def _dust_exit_action(
        self, pos: GridPosition, quantity: float, ref_px: float, timestamp: str
    ) -> Dict[str, Any]:
        return {
            "action": "market_exit",
            "order_id": f"{pos.leg_id}_dust",
            "leg_id": pos.leg_id,
            "symbol": pos.symbol,
            "side": pos.side,
            "level": pos.level,
            "quantity": quantity,
            "exit_price": ref_px,
            "reason": "dust_below_min_notional",
            "timestamp": timestamp,
        }

    def _should_emit_dust_exit(
        self,
        pos: GridPosition,
        quantity: float,
        ref_px: float,
        timestamp: str,
        open_orders: List[Dict[str, Any]],
    ) -> bool:
        leg_id = str(pos.leg_id or "")
        if leg_id and leg_id in self.state.pending_dust_exits:
            return False
        dust_action = self._dust_exit_action(pos, quantity, ref_px, timestamp)
        dust_cid = derive_multileg_client_order_id(dust_action)
        if dust_cid in _open_exchange_client_order_ids(open_orders):
            if leg_id:
                self._mark_pending_dust_exit(leg_id)
            return False
        return True

    def _mark_pending_dust_exit(self, leg_id: str) -> None:
        leg = str(leg_id or "").strip()
        if not leg or leg in self.state.pending_dust_exits:
            return
        self.state.pending_dust_exits.append(leg)

    def _clear_pending_dust_exit(self, leg_id: str) -> None:
        leg = str(leg_id or "").strip()
        if not leg:
            return
        self.state.pending_dust_exits = [
            x for x in self.state.pending_dust_exits if x != leg
        ]

    @staticmethod
    def _min_notional_usd() -> float:
        raw = os.getenv("MLBOT_CHOP_GRID_MIN_NOTIONAL_USD", "5").strip()
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return 5.0

    def _is_dust_notional(self, quantity: float, price: float) -> bool:
        qty = float(quantity or 0.0)
        px = float(price or 0.0)
        if qty <= 0 or px <= 0:
            return False
        return qty * px < self._min_notional_usd()

    def _exchange_side_qty_map(
        self,
        exchange_positions: Iterable[Mapping[str, Any]],
        *,
        symbol: str,
    ) -> Dict[str, float]:
        sym = str(symbol).upper()
        out: Dict[str, float] = {}
        for raw in exchange_positions:
            row = dict(raw)
            if _normalize_symbol(row.get("symbol")) != sym:
                continue
            qty = _exchange_position_quantity(row)
            if qty <= 0:
                continue
            side_raw = str(row.get("side") or row.get("positionSide") or "").lower()
            pos_side = "LONG" if side_raw in {"long", "buy"} else "SHORT"
            out[pos_side] = out.get(pos_side, 0.0) + qty
        return out

    def _match_leg_id_for_fill(self, pos_side: str, entry_price: float) -> str:
        best_id = f"{self.state.grid_id}_{'L1' if pos_side == 'LONG' else 'S1'}"
        best_dist = float("inf")
        for order in list(self.state.pending_orders):
            oside = "LONG" if order.side == "BUY" else "SHORT"
            if oside != pos_side:
                continue
            dist = abs(order.price - entry_price)
            if dist < best_dist:
                best_dist = dist
                best_id = order.order_id
        if best_dist < max(self.state.spacing * 0.5, entry_price * 0.002):
            return best_id
        gid = self.state.grid_id or ""
        return f"{gid}_{'L1' if pos_side == 'LONG' else 'S1'}"

    def _has_open_protection(
        self, pos: GridPosition, open_orders: List[Dict[str, Any]]
    ) -> bool:
        need = float(pos.quantity or 0.0)
        if need <= 0:
            return False
        return self._open_tp_covered_qty(pos, open_orders) >= need * 0.99

    def _open_tp_covered_qty(
        self, pos: GridPosition, open_orders: List[Dict[str, Any]]
    ) -> float:
        tp_px = self._tp_price_for_position(pos)
        if tp_px is None:
            return 0.0
        protection_ids = {str(oid) for oid in pos.protection_order_ids if str(oid)}
        if protection_ids:
            live_ids = {
                key
                for order in open_orders
                for key in _exchange_order_keys(order)
                if key
            }
            pos.protection_order_ids = [
                oid for oid in pos.protection_order_ids if str(oid) in live_ids
            ]
            protection_ids = {str(oid) for oid in pos.protection_order_ids if str(oid)}
        pos_side = str(pos.side).upper()
        covered_qty = 0.0
        for order in open_orders:
            if protection_ids and _exchange_order_keys(order).isdisjoint(
                protection_ids
            ):
                continue
            o_side = str(order.get("side") or "").lower()
            o_pos = str(
                order.get("position_side")
                or order.get("positionSide")
                or (order.get("info") or {}).get("positionSide")
                or ""
            ).upper()
            if not o_pos:
                # Without Hedge Mode side we cannot distinguish protection from
                # the opposite grid entry, so fail closed and place protection.
                continue
            if o_pos != pos_side:
                continue
            if pos_side == "LONG" and o_side != "sell":
                continue
            if pos_side == "SHORT" and o_side != "buy":
                continue
            price = _as_float(order.get("price"), 0.0)
            if price <= 0:
                continue
            o_qty = _as_float(
                order.get("quantity") or order.get("remaining"),
                0.0,
            )
            if o_qty <= 0:
                o_qty = _as_float(order.get("filled"), 0.0)
            if o_qty <= 0:
                continue
            if abs(price - tp_px) <= max(self.state.spacing * 0.15, price * 0.001):
                covered_qty += o_qty
            elif abs(price - tp_px) <= self.state.spacing * 2:
                covered_qty += o_qty
        return covered_qty

    def _is_chop_grid_exchange_order(self, order: Mapping[str, Any]) -> bool:
        info = order.get("info") or {}
        cid = str(order.get("client_order_id") or info.get("clientOrderId") or "")
        return cid.startswith("cg_")

    def _tp_distance(self) -> float:
        """Take-profit distance = grid spacing * tp_spacing_mult (decoupled exit)."""
        return self.state.spacing * float(
            getattr(self.cfg, "tp_spacing_mult", 1.0) or 1.0
        )

    def _tp_price_for_position(self, pos: GridPosition) -> Optional[float]:
        if self.state.spacing <= 0:
            return None
        tp_distance = self._tp_distance()
        if pos.side == "LONG":
            return pos.entry_price + tp_distance
        return pos.entry_price - tp_distance

    def on_bar(
        self,
        *,
        symbol: str,
        timestamp: str,
        high: float,
        low: float,
        close: float,
        atr: float,
        features: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Process one completed bar and return dry-run order actions."""
        actions: List[Dict[str, Any]] = []
        chop = resolve_semantic_chop(features, default=0.0)
        if chop is None:
            chop = 0.0
        is_box = stable_box_blocks_chop_entry(
            features,
            self.regime,
            rules=self._prefilter_rules,
        )
        wanted_enter = chop >= self.cfg.entry_chop_min and not is_box
        if self.state.symbol == symbol:
            self.clear_stale_active_if_ghost()
        active_at_open = self.state.active and self.state.symbol == symbol
        should_enter = wanted_enter and not self.state.active
        should_exit = self.state.active and chop < self.cfg.exit_chop_below

        self.state.last_timestamp = timestamp
        if not self.state.active and should_enter:
            gate = getattr(self, "_concurrency_gate", None)
            if gate is not None and not gate.allow_new_segment(symbol):
                should_enter = False
            else:
                actions.extend(self._start_grid(symbol, timestamp, close, atr))

        if self.state.active and self.state.symbol == symbol:
            if self.bar_simulation:
                actions.extend(self._simulate_fills(timestamp, high, low))
                actions.extend(self._simulate_targets(timestamp, high, low))
            else:
                actions.extend(self._maybe_replenish_empty_levels(symbol, timestamp))
            if self._risk_stop(close):
                should_exit = True

        if should_exit:
            actions.extend(
                self._exit_grid(timestamp, close, reason="regime_or_risk_exit")
            )

        from src.time_series_model.live.multileg_funnel import chop_grid_bar_outcome

        outcome = chop_grid_bar_outcome(
            active_at_open=active_at_open,
            wanted_enter=wanted_enter,
            is_box=is_box,
            chop=chop,
            entry_chop_min=self.cfg.entry_chop_min,
            actions=actions,
        )
        self._last_bar_audit = {
            "engine": "chop_grid",
            "chop": chop,
            "is_box": is_box,
            "wanted_enter": wanted_enter,
            "active_at_open": active_at_open,
            "should_enter": should_enter,
            "should_exit": should_exit,
            "outcome": outcome,
        }
        self._emit_chop_bar_outcome(symbol, outcome=outcome)
        self.save_state()
        return actions

    def _start_grid(
        self, symbol: str, timestamp: str, close: float, atr: float
    ) -> List[Dict[str, Any]]:
        spacing = max(self.cfg.grid_atr_mult * atr, self.cfg.grid_min_pct * close)
        self.state = GridState(
            grid_id=f"{symbol}_{timestamp}",
            symbol=symbol,
            active=True,
            center=close,
            spacing=spacing,
            last_timestamp=timestamp,
            current_regime="chop_grid",
            level_replenish_count={},
        )
        actions: List[Dict[str, Any]] = []
        for level in range(1, self.cfg.max_levels_per_side + 1):
            qty = self.level_notional / max(close, 1e-12)
            buy = GridOrder(
                order_id=f"{self.state.grid_id}_L{level}",
                symbol=symbol,
                side="BUY",
                level=level,
                price=close - spacing * level,
                quantity=qty,
                created_at=timestamp,
            )
            sell = GridOrder(
                order_id=f"{self.state.grid_id}_S{level}",
                symbol=symbol,
                side="SELL",
                level=level,
                price=close + spacing * level,
                quantity=qty,
                created_at=timestamp,
            )
            self.state.pending_orders.extend([buy, sell])
            actions.extend(
                [
                    {
                        "action": "place",
                        "order_type": "limit",
                        "expected_liquidity": "maker",
                        "expected_fee_bps": self._maker_fee_bps(),
                        **asdict(buy),
                    },
                    {
                        "action": "place",
                        "order_type": "limit",
                        "expected_liquidity": "maker",
                        "expected_fee_bps": self._maker_fee_bps(),
                        **asdict(sell),
                    },
                ]
            )
        return actions

    def _simulate_fills(
        self, timestamp: str, high: float, low: float
    ) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        remaining: List[GridOrder] = []
        for order in self.state.pending_orders:
            filled = (order.side == "BUY" and low <= order.price) or (
                order.side == "SELL" and high >= order.price
            )
            if not filled:
                remaining.append(order)
                continue
            pos_side = "LONG" if order.side == "BUY" else "SHORT"
            self.state.inventory.append(
                GridPosition(
                    symbol=order.symbol,
                    side=pos_side,
                    level=order.level,
                    entry_price=order.price,
                    quantity=order.quantity,
                    entry_quantity=order.quantity,
                    entry_time=timestamp,
                    leg_id=order.order_id,
                )
            )
            actions.append(
                {
                    "action": "fill",
                    "order_id": order.order_id,
                    "fill_price": order.price,
                }
            )
        self.state.pending_orders = remaining
        return actions

    def _simulate_targets(
        self, timestamp: str, high: float, low: float
    ) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        remaining: List[GridPosition] = []
        fee = self._maker_fee_bps() / 10000.0
        tp_distance = self._tp_distance()
        for pos in self.state.inventory:
            if pos.side == "LONG":
                target = pos.entry_price + tp_distance
                hit = high >= target
                pnl = (
                    target - pos.entry_price
                ) * pos.quantity - 2.0 * fee * pos.entry_price * pos.quantity
            else:
                target = pos.entry_price - tp_distance
                hit = low <= target
                pnl = (
                    pos.entry_price - target
                ) * pos.quantity - 2.0 * fee * pos.entry_price * pos.quantity
            if hit:
                self.state.realized_pnl += pnl
                actions.append(
                    {
                        "action": "take_profit",
                        "symbol": pos.symbol,
                        "side": pos.side,
                        "level": pos.level,
                        "exit_price": target,
                        "pnl": pnl,
                        "entry_liquidity": "maker",
                        "exit_liquidity": "maker",
                        "fee_bps_charged": 2.0 * self._maker_fee_bps(),
                        "timestamp": timestamp,
                    }
                )
                actions.extend(self._after_level_tp_closed(pos, timestamp))
            else:
                remaining.append(pos)
        self.state.inventory = remaining
        return actions

    def _risk_stop(self, close: float) -> bool:
        if self.cfg.max_open_levels_total is not None:
            if len(self.state.inventory) > int(self.cfg.max_open_levels_total):
                return True
        if self.cfg.max_loss_per_grid is None:
            return False
        mtm = self.state.realized_pnl
        for pos in self.state.inventory:
            if pos.side == "LONG":
                mtm += (close - pos.entry_price) * pos.quantity
            else:
                mtm += (pos.entry_price - close) * pos.quantity
        return mtm <= -abs(float(self.cfg.max_loss_per_grid)) * max(
            self.level_notional, 1e-12
        )

    def _exit_grid(
        self, timestamp: str, close: float, *, reason: str
    ) -> List[Dict[str, Any]]:
        logger.info(
            "chop_grid exit_grid: symbol=%s reason=%s pending_orders=%d inventory_levels=%d close=%s",
            getattr(self.state, "symbol", ""),
            reason,
            len(self.state.pending_orders),
            len(self.state.inventory),
            close,
        )
        actions: List[Dict[str, Any]] = []
        for order in self.state.pending_orders:
            actions.append(
                {
                    "action": "cancel",
                    "order_id": order.order_id,
                    "exchange_order_id": order.exchange_order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "level": order.level,
                    "price": order.price,
                    "reason": reason,
                }
            )
        for pos in self.state.inventory:
            actions.append(
                {
                    "action": "market_exit",
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "level": pos.level,
                    "quantity": pos.quantity,
                    "exit_price": close,
                    "reason": reason,
                    "entry_liquidity": "maker",
                    "exit_liquidity": "taker",
                    "fee_bps_charged": self._maker_fee_bps() + self._taker_fee_bps(),
                    "slippage_bps_charged": self.cfg.forced_exit_slippage_bps,
                    "timestamp": timestamp,
                }
            )
        self.state.pending_orders = []
        self.state.inventory = []
        self.state.level_replenish_count = {}
        self.state.active = False
        self.state.current_regime = "idle"
        gate = getattr(self, "_concurrency_gate", None)
        if gate is not None:
            gate.notify_deactivation(self.state.symbol, "chop_grid")
        return actions

    def _may_replenish_level(self, level_key: str) -> bool:
        max_r = self.cfg.max_replenish_per_level_per_segment
        if max_r is None:
            return True
        return int(self.state.level_replenish_count.get(level_key, 0)) <= int(max_r)

    def _grid_entry_price(self, level: int, pos_side: str) -> float:
        lv = int(level)
        if str(pos_side).upper() in {"LONG", "BUY"}:
            return self.state.center - self.state.spacing * lv
        return self.state.center + self.state.spacing * lv

    def _entry_order_id(self, level: int, pos_side: str) -> str:
        key = _level_side_key(level, pos_side)
        trips = int(self.state.level_replenish_count.get(key, 0))
        base = f"{self.state.grid_id}_{key}"
        if trips <= 0:
            return base
        return f"{base}_r{trips}"

    def _has_pending_at_level(self, level: int, pos_side: str) -> bool:
        want_side = "BUY" if str(pos_side).upper() in {"LONG", "BUY"} else "SELL"
        for order in self.state.pending_orders:
            if int(order.level) == int(level) and order.side == want_side:
                return True
        return False

    def _has_inventory_at_level(self, level: int, pos_side: str) -> bool:
        want = "LONG" if str(pos_side).upper() in {"LONG", "BUY"} else "SHORT"
        for pos in self.state.inventory:
            if int(pos.level) == int(level) and str(pos.side).upper() == want:
                return True
        return False

    def _replenish_actions_for_level(
        self, symbol: str, level: int, pos_side: str, timestamp: str
    ) -> List[Dict[str, Any]]:
        if not self.state.active or self.state.spacing <= 0:
            return []
        level_key = _level_side_key(level, pos_side)
        if not self._may_replenish_level(level_key):
            return []
        if self._has_pending_at_level(level, pos_side) or self._has_inventory_at_level(
            level, pos_side
        ):
            return []
        price = self._grid_entry_price(level, pos_side)
        order_side = "BUY" if str(pos_side).upper() in {"LONG", "BUY"} else "SELL"
        qty = self.level_notional / max(self.state.center, 1e-12)
        order = GridOrder(
            order_id=self._entry_order_id(level, pos_side),
            symbol=symbol,
            side=order_side,
            level=int(level),
            price=price,
            quantity=qty,
            created_at=timestamp,
        )
        self.state.pending_orders.append(order)
        return [
            {
                "action": "place",
                "order_type": "limit",
                "expected_liquidity": "maker",
                "expected_fee_bps": self._maker_fee_bps(),
                **asdict(order),
            }
        ]

    def _after_level_tp_closed(
        self, pos: GridPosition, timestamp: str
    ) -> List[Dict[str, Any]]:
        level_key = _level_side_key(pos.level, pos.side)
        self.state.inventory = [
            p for p in self.state.inventory if p.leg_id != pos.leg_id
        ]
        self.state.level_replenish_count[level_key] = (
            int(self.state.level_replenish_count.get(level_key, 0)) + 1
        )
        return self._replenish_actions_for_level(
            pos.symbol, int(pos.level), pos.side, timestamp
        )

    def _maybe_replenish_empty_levels(
        self, symbol: str, timestamp: str
    ) -> List[Dict[str, Any]]:
        """Reconcile fallback for missed TP execution reports.

        Only acts on levels with ``level_replenish_count[key] >= 1`` (i.e. at
        least one round-trip already recorded). This guards against
        ``_sync_inventory_from_exchange`` clearing a position without a TP
        signal, which would otherwise look identical to a post-TP empty level
        and cause phantom duplicate orders.
        """
        if not self.state.active or self.state.symbol != symbol:
            return []
        if (
            self._live_exchange_has_activity
            and not self.state.pending_orders
            and not self.state.inventory
        ):
            return []
        actions: List[Dict[str, Any]] = []
        for level in range(1, self.cfg.max_levels_per_side + 1):
            for pos_side in ("LONG", "SHORT"):
                key = _level_side_key(level, pos_side)
                if int(self.state.level_replenish_count.get(key, 0)) <= 0:
                    continue
                actions.extend(
                    self._replenish_actions_for_level(
                        symbol, level, pos_side, timestamp
                    )
                )
        return actions

    def _protection_report_kind(self, report: Mapping[str, Any]) -> str:
        prot = str(report.get("protection_type") or "").strip().lower()
        if prot in {"stop_loss", "sl", "stop", "stop_market"}:
            return "stop_loss"
        if prot in {"take_profit", "tp", "take_profit_market"}:
            return "take_profit"
        order_type = str(report.get("order_type") or "").strip().upper()
        if order_type in {
            "STOP",
            "STOP_MARKET",
            "STOP_LOSS",
            "STOP_LOSS_LIMIT",
            "TRAILING_STOP_MARKET",
        }:
            return "stop_loss"
        if order_type in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET", "TAKE_PROFIT_LIMIT"}:
            return "take_profit"
        cid = str(report.get("client_order_id") or "")
        leg_hint = str(report.get("leg_id") or report.get("local_order_id") or "")
        if cid.endswith("_sl") or leg_hint.endswith("_sl"):
            return "stop_loss"
        if cid.endswith("_tp") or leg_hint.endswith("_tp"):
            return "take_profit"
        return "take_profit"

    def _handle_protection_fill(self, report: Dict[str, Any]) -> bool:
        """Detect TP/SL protection fills so replenish + count++ stay accurate.

        Matches by:
          1. ``report.order_id`` (exchange id) against ``pos.protection_order_ids``
             — populated by ``on_execution_results(action="place_protection")``.
          2. ``report.client_order_id`` against the same id list (some streams
             surface only the deterministic client id we supplied).
          3. ``report.leg_id`` (or ``local_order_id``) — orchestrator-provided
             hint of the entry order id, with optional ``_tp`` suffix.
        """
        status = str(report.get("status") or "").upper()
        if status != "FILLED":
            return False
        ex_id = str(report.get("order_id") or "")
        cid = str(report.get("client_order_id") or "")
        leg_hint = _normalize_entry_leg_id(
            str(report.get("leg_id") or report.get("local_order_id") or "")
        )
        kind = self._protection_report_kind(report)
        filled_qty = _as_float(report.get("filled_qty"), 0.0)
        ts = str(report.get("trade_time") or self.state.last_timestamp)

        def _apply_fill(pos: GridPosition) -> bool:
            if kind == "stop_loss":
                close_qty = filled_qty if filled_qty > 0 else float(pos.quantity or 0.0)
                remaining = max(0.0, float(pos.quantity or 0.0) - close_qty)
                # Partial SL fill: shrink the leg but keep it (and its protection)
                # so the rest of the position stays tracked. Only a full close
                # falls through to the existing close+replenish bookkeeping.
                if remaining > 1e-8:
                    pos.quantity = remaining
                    if ex_id:
                        pos.protection_order_ids = [
                            pid for pid in pos.protection_order_ids if str(pid) != ex_id
                        ]
                    logger.info(
                        "chop_grid partial SL fill: leg_id=%s closed_qty=%.8f "
                        "remaining=%.8f",
                        pos.leg_id,
                        close_qty,
                        remaining,
                    )
                    return True
            self._pending_actions.extend(self._after_level_tp_closed(pos, ts))
            return True

        for pos in list(self.state.inventory):
            prot_ids = {str(x) for x in pos.protection_order_ids if str(x)}
            if (ex_id and ex_id in prot_ids) or (cid and cid in prot_ids):
                return _apply_fill(pos)
        if leg_hint:
            pos = self._find_position(leg_hint)
            if pos is not None:
                return _apply_fill(pos)
        return False

    def _protection_actions(
        self, *, order_id: str, pos: GridPosition, timestamp: str
    ) -> List[Dict[str, Any]]:
        """Create native exchange protection orders for a filled grid leg."""

        if self.state.spacing <= 0:
            return []
        tp_distance = self._tp_distance()
        if pos.side == "LONG":
            tp = pos.entry_price + tp_distance
            sl = pos.entry_price - self.state.spacing * (
                self.cfg.max_levels_per_side + 1
            )
        else:
            tp = pos.entry_price - tp_distance
            sl = pos.entry_price + self.state.spacing * (
                self.cfg.max_levels_per_side + 1
            )
        return [
            {
                "action": "place_protection",
                "order_id": f"{order_id}_tp",
                "leg_id": order_id,
                "symbol": pos.symbol,
                "side": pos.side,
                "quantity": pos.quantity,
                "price": tp,
                "trigger_price": tp,
                "order_type": "limit",
                "protection_type": "take_profit",
                "reduce_only": True,
                "post_only": False,
                "time_in_force": "GTC",
                "timestamp": timestamp,
            },
            {
                "action": "place_protection",
                "order_id": f"{order_id}_sl",
                "leg_id": order_id,
                "symbol": pos.symbol,
                "side": pos.side,
                "quantity": pos.quantity,
                "trigger_price": sl,
                "protection_type": "stop_loss",
                "timestamp": timestamp,
            },
        ]

    def _maker_fee_bps(self) -> float:
        if self.cfg.maker_fee_bps is not None:
            return float(self.cfg.maker_fee_bps)
        return float(self.cfg.fee_bps)

    def _taker_fee_bps(self) -> float:
        if self.cfg.taker_fee_bps is not None:
            return float(self.cfg.taker_fee_bps)
        return float(self.cfg.fee_bps)

    def _find_order(
        self,
        *,
        local_id: str = "",
        exchange_id: str = "",
        client_id: str = "",
    ) -> Optional[GridOrder]:
        for order in self.state.pending_orders:
            if local_id and order.order_id == local_id:
                return order
            if exchange_id and order.exchange_order_id == exchange_id:
                return order
            if client_id and order.client_order_id == client_id:
                return order
        return None

    def _find_position(self, leg_id: str) -> Optional[GridPosition]:
        if not leg_id:
            return None
        for pos in self.state.inventory:
            if pos.leg_id == leg_id:
                return pos
        return None
