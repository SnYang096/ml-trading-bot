"""B 层 Trend 持仓 TruthSync — SQLite 投影唯一写入口。

P1 实现：所有 SQLite positions 表写入经 ``project_to_sqlite()``，
消灭「PT 写一处、脚本写一处、CMS 逻辑再猜」的多入口问题。

P2 实现：统一 bootstrap — ``bootstrap_position_from_exchange()`` 和
``on_restart()`` 合并 run_live 与 sync 脚本的双入口。

Not a standalone daemon — imported by existing live processes.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from src.order_management.models import (
    Position,
    PositionSide,
    PositionStatus,
)

logger = logging.getLogger(__name__)


class TrendPositionTruthSync:
    """B 层 Trend 持仓 TruthSync — SQLite 投影唯一写入口。

    Args:
        symbol: 交易对符号（如 "BTCUSDT"）
        storage_factory: callable 返回 Storage 实例（或 None）
    """

    def __init__(
        self,
        symbol: str,
        storage_factory: Callable[[], Any],
    ) -> None:
        self.symbol = symbol
        self._storage_factory = storage_factory

    def _storage(self) -> Any:
        """获取 Storage 实例。"""
        try:
            return self._storage_factory()
        except Exception:
            return None

    @staticmethod
    def _as_dt(value: Any) -> datetime:
        """Convert various datetime representations to a datetime object."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    def project_to_sqlite(
        self,
        position_id: str,
        pos: Dict[str, Any],
        *,
        status: PositionStatus = PositionStatus.OPEN,
        exit_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
    ) -> str:
        """唯一 SQLite positions 写入口。

        Best-effort mirror of the in-memory position dict into the SQLite
        ``positions`` table.  Includes dedup merge: if another OPEN row
        already exists for the same symbol+side, reuse it instead of
        creating a duplicate.

        Returns the canonical ``position_id`` written (may differ after
        dedup merge).
        """
        storage = self._storage()
        if storage is None:
            return position_id

        side_raw = str(pos.get("side") or "").upper()
        side = PositionSide.LONG if side_raw in {"LONG", "BUY"} else PositionSide.SHORT

        try:
            entry_price = float(pos.get("entry_price") or 0.0)
        except (TypeError, ValueError):
            entry_price = 0.0

        try:
            qty = float(pos.get("qty") or 0.0)
        except (TypeError, ValueError):
            qty = 0.0

        if not position_id or (status == PositionStatus.OPEN and qty <= 0):
            return position_id

        try:
            existing = storage.get_position(position_id)
        except Exception:
            existing = None

        # ── Dedup: if another OPEN position already exists for the same
        #     symbol+side (e.g. exchange_sync created before bootstrap
        #     JSON loaded), reuse it instead of creating a duplicate row. ──
        if existing is None and status == PositionStatus.OPEN:
            try:
                for p in storage.get_open_positions(self.symbol) or []:
                    if p.side == side and p.status == PositionStatus.OPEN:
                        logger.info(
                            "[%s] merged duplicate position %s -> existing %s",
                            self.symbol,
                            position_id,
                            p.position_id,
                        )
                        existing = p
                        position_id = str(p.position_id or position_id)
                        break
            except Exception:
                pass

        record = existing or Position(
            position_id=position_id,
            symbol=self.symbol,
            side=side,
            entry_time=self._as_dt(pos.get("entry_time")),
        )

        record.symbol = self.symbol
        record.side = side
        record.entry_price = entry_price
        record.initial_size = pos.get("initial_size") or record.initial_size or qty
        record.current_size = 0.0 if status == PositionStatus.CLOSED else qty
        record.total_cost = entry_price * (record.initial_size or qty)
        record.status = status
        record.stop_loss_price = pos.get("stop_loss_price")
        record.take_profit_price = pos.get("take_profit_price")
        record.strategy_id = str(pos.get("archetype") or "") or record.strategy_id
        record.archetype = str(pos.get("archetype") or "") or record.archetype
        record.add_count = int(pos.get("_add_position_seq") or record.add_count or 0)
        record.parent_position_id = pos.get("_parent_pid") or record.parent_position_id
        record.notes = pos.get("notes") or record.notes
        record.unrealized_pnl = pos.get("unrealized_pnl") or record.unrealized_pnl
        record.realized_pnl = pos.get("realized_pnl") or record.realized_pnl

        if status == PositionStatus.CLOSED:
            record.exit_time = datetime.now(timezone.utc)
            record.exit_price = exit_price
            record.exit_reason = exit_reason

        try:
            if existing is None:
                storage.create_position(record)
            else:
                storage.update_position(record)
        except Exception:
            logger.warning(
                "[%s] persist position record skipped: %s",
                self.symbol,
                position_id,
                exc_info=True,
            )

        return position_id

    # ── P2: Unified Bootstrap ────────────────────────────────────

    @staticmethod
    def _make_pid(symbol: str) -> str:
        """Generate canonical position_id: ``{BASE}:live_{uuid12}``."""
        base = symbol.upper().removesuffix("USDT") if symbol.upper().endswith("USDT") else symbol.upper()
        return f"{base}:live_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _to_json_safe(value: Any) -> Any:
        if isinstance(value, datetime):
            return {"__datetime__": value.isoformat()}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): TrendPositionTruthSync._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [TrendPositionTruthSync._to_json_safe(v) for v in value]
        return str(value)

    @staticmethod
    def _write_tracker_state(
        *, state_path: Path, symbol: str, positions: Dict[str, Dict[str, Any]],
        merge: bool = False,
    ) -> None:
        """Write position dict to JSON tracker file.

        Args:
            merge: If ``True``, merge *positions* into existing file rather than
                overwriting — required for multi-leg DR to preserve all legs.
        """
        state_path.parent.mkdir(parents=True, exist_ok=True)
        existing_positions: Dict[str, Any] = {}
        if merge and state_path.exists():
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
                existing_positions = dict(data.get("positions") or {})
            except Exception:
                pass
        if merge:
            existing_positions.update(TrendPositionTruthSync._to_json_safe(positions))
            safe_positions = existing_positions
        else:
            safe_positions = TrendPositionTruthSync._to_json_safe(positions)
        payload = {
            "version": 1,
            "symbol": symbol,
            "positions": safe_positions,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "_bootstrap_from_exchange": True,
        }
        tmp = state_path.with_suffix(state_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        tmp.replace(state_path)

    @staticmethod
    def _entry_time_from_trades(
        api: Any, ccxt_symbol: str, *, side: str
    ) -> Optional[datetime]:
        """Best-effort entry time from exchange recent trades."""
        try:
            trades = api.exchange.fetch_my_trades(ccxt_symbol, limit=50)
        except Exception:
            return None
        if not trades:
            return None
        want = "sell" if str(side).lower() == "short" else "buy"
        candidates: List[datetime] = []
        for t in trades:
            if str(t.get("side", "")).lower() != want:
                continue
            ts = t.get("datetime")
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                candidates.append(ts)
        if not candidates:
            return None
        return min(candidates)

    @staticmethod
    def bootstrap_position_from_exchange(
        *,
        symbol: str,
        side: str,
        entry_price: float,
        qty: float,
        execution_yaml: Optional[Path] = None,
        archetype: str = "tpc",
        bar_minutes: int = 120,
        atr_pct: float = 0.01,
        entry_time: Optional[datetime] = None,
        api: Any = None,
        ccxt_symbol: Optional[str] = None,
        state_path: Optional[Path] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Build a position dict from exchange data — unified bootstrap.

        Prefers ``build_position_dict`` + ``execution.yaml`` when available.
        Falls back to conservative 1% ATR / 1.5R stop / 3R target.

        Returns ``(position_id, pos_dict)``.
        """
        if not symbol or qty <= 0 or entry_price <= 0:
            raise ValueError(f"invalid bootstrap params for {symbol}")

        sym_raw = symbol.upper().removesuffix("USDT") if symbol.upper().endswith("USDT") else symbol.upper()
        sym_full = symbol.upper()  # full pair e.g. "BTCUSDT" for tracker JSON
        pid = TrendPositionTruthSync._make_pid(symbol)
        side_norm = side.lower()
        action = "SHORT" if side_norm in {"short", "sell"} else "LONG"
        entry = float(entry_price)
        qty_f = float(qty)
        atr = max(1e-6, entry * max(1e-6, float(atr_pct)))

        # Try entry_time from exchange trades
        if entry_time is None and api is not None and ccxt_symbol:
            entry_time = TrendPositionTruthSync._entry_time_from_trades(
                api, ccxt_symbol, side=side_norm
            )
        if entry_time is None:
            entry_time = datetime.now(timezone.utc)

        pos_dict: Optional[Dict[str, Any]] = None

        # Prefer build_position_dict with execution.yaml
        if execution_yaml is not None and execution_yaml.is_file():
            try:
                from src.time_series_model.core.trade_intent import TradeIntent
                from src.time_series_model.live.execution_profile_apply import (
                    rr_constraints_from_exec_params,
                )
                from src.time_series_model.live.generic_live_strategy import (
                    ExecutionParamGenerator,
                )
                from src.time_series_model.live.position_logic import build_position_dict

                execution_cfg = yaml.safe_load(
                    execution_yaml.read_text(encoding="utf-8")
                ) or {}
                gen = ExecutionParamGenerator(execution_cfg)
                exec_params = gen.generate_params(evidence_score=0.5)
                rr = rr_constraints_from_exec_params(exec_params)
                ep = {"rr_constraints": rr, "strategy_specific": {}}

                intent = TradeIntent(
                    action=action,
                    symbol=sym_raw,
                    archetype=archetype,
                    execution_profile=ep,
                    position_id=pid,
                )
                pos_dict = build_position_dict(
                    intent=intent,
                    entry_price=entry,
                    atr=atr,
                    bar_minutes=bar_minutes,
                    entry_time=entry_time,
                )
                pos_dict["qty"] = qty_f
                pos_dict["symbol"] = sym_raw
                pos_dict["archetype"] = archetype
                logger.info(
                    "[%s] bootstrap via execution.yaml: pid=%s side=%s sl=%s tp=%s",
                    symbol, pid, action,
                    pos_dict.get("stop_loss_price"),
                    pos_dict.get("take_profit_price"),
                )
            except Exception:
                logger.warning(
                    "[%s] execution.yaml bootstrap failed, falling back to conservative defaults",
                    symbol, exc_info=True,
                )
                pos_dict = None

        # Fallback: conservative defaults
        if pos_dict is None:
            atr_est = entry * 0.01
            stop_r = 1.5
            sl_price = (
                entry - (stop_r * atr_est)
                if action == "LONG"
                else entry + (stop_r * atr_est)
            )
            tp_price = (
                entry + (2.0 * stop_r * atr_est)
                if action == "LONG"
                else entry - (2.0 * stop_r * atr_est)
            )
            pos_dict = {
                "position_id": pid,
                "symbol": sym_raw,
                "side": action,
                "entry_price": entry,
                "qty": qty_f,
                "entry_time": entry_time,
                "archetype": archetype,
                "status": "open",
                "atr_at_entry": atr_est,
                "stop_loss_r": stop_r,
                "bar_minutes": bar_minutes,
                "stop_loss_price": round(sl_price, 4),
                "take_profit_price": round(tp_price, 4),
                "high_water_mark": entry if action == "LONG" else None,
                "low_water_mark": entry if action == "SHORT" else None,
            }
            logger.info(
                "[%s] bootstrap conservative: pid=%s side=%s sl=%.4f (1%% ATR, 1.5R)",
                symbol, pid, action, sl_price,
            )

        pos_dict["_bootstrap_from_exchange"] = True

        # Write JSON tracker state (merge=True to preserve other legs)
        # Use full symbol (BTCUSDT) for JSON compatibility with PositionTracker.restore_from_disk
        if state_path is not None:
            TrendPositionTruthSync._write_tracker_state(
                state_path=state_path,
                symbol=sym_full,
                positions={pid: pos_dict},
                merge=True,
            )

        return pid, pos_dict

    def on_restart(
        self,
        *,
        api: Any,
        tracker: Any,
        state_path: Path,
        execution_yaml: Optional[Path] = None,
        archetype: str = "tpc",
        bar_minutes: int = 120,
        force_exchange: bool = False,
    ) -> Dict[str, Any]:
        """Unified restart entry point: restore JSON → bootstrap missing → merge SQLite.

        Args:
            api: BinanceAPI instance
            tracker: PositionTracker instance
            state_path: JSON state file path for this symbol
            execution_yaml: Path to execution.yaml (optional)
            archetype: Strategy archetype
            bar_minutes: Bar timeframe in minutes
            force_exchange: DR mode — always rebuild from exchange

        Returns:
            Report dict with ``restored``, ``bootstrapped``, ``merged`` counts.
        """
        report: Dict[str, Any] = {
            "symbol": self.symbol,
            "restored": 0,
            "bootstrapped": 0,
            "merged": 0,
        }

        # Step 1: Try restore from disk
        if not force_exchange and hasattr(tracker, "restore_from_disk"):
            try:
                restored = int(tracker.restore_from_disk() or 0)
                report["restored"] = restored
                if restored > 0:
                    logger.info(
                        "[%s] on_restart: restored %d positions from JSON",
                        self.symbol, restored,
                    )
            except Exception:
                logger.warning(
                    "[%s] on_restart: restore_from_disk failed", self.symbol,
                    exc_info=True,
                )

        # Step 2: Bootstrap missing sides from exchange (per-side granularity)
        # Always query exchange; skip only sides tracker already has.
        tracker_sides: set = set()
        for pos in (getattr(tracker, "_positions", None) or {}).values():
            side = str(pos.get("side", "")).lower()
            if side in {"long", "buy"}:
                tracker_sides.add("long")
            elif side in {"short", "sell"}:
                tracker_sides.add("short")

        if force_exchange and hasattr(tracker, "_positions"):
            tracker._positions.clear()
            tracker_sides.clear()
            if hasattr(tracker, "_persist_state"):
                tracker._persist_state()

        try:
            exchange_positions = api.get_positions() or []
        except Exception:
            logger.warning(
                "[%s] on_restart: failed to query exchange positions",
                self.symbol, exc_info=True,
            )
            exchange_positions = []

        for pos in exchange_positions:
            raw_sym = str(
                pos.get("symbol", "")
            ).replace("/", "").split(":")[0].upper().strip()
            if raw_sym != self.symbol.upper():
                continue
            qty = abs(float(pos.get("size") or pos.get("contracts") or 0))
            if qty <= 0:
                continue
            side_raw = str(pos.get("side", "")).lower()
            side = "short" if side_raw == "short" else "long"
            # Skip sides that tracker already has (unless force_exchange)
            if not force_exchange and side in tracker_sides:
                continue
            entry = float(pos.get("entry_price") or 0.0)
            if entry <= 0:
                continue

            ccxt_sym = str(pos.get("symbol", ""))
            pid, pos_dict = self.bootstrap_position_from_exchange(
                symbol=self.symbol,
                side=side,
                entry_price=entry,
                qty=qty,
                execution_yaml=execution_yaml,
                archetype=archetype,
                bar_minutes=bar_minutes,
                ccxt_symbol=ccxt_sym,
                api=api,
                state_path=state_path,
            )

            # Load into tracker memory
            if hasattr(tracker, "_positions"):
                tracker._positions[pid] = pos_dict
                if hasattr(tracker, "_persist_state"):
                    tracker._persist_state()

            # Project to SQLite
            canonical = self.project_to_sqlite(pid, pos_dict)
            if hasattr(tracker, "_rekey_in_memory") and canonical != pid:
                tracker._rekey_in_memory(pid, canonical, pos_dict)
                if hasattr(tracker, "_persist_state"):
                    tracker._persist_state()

            report["bootstrapped"] += 1
            logger.warning(
                "[%s] on_restart bootstrap: pid=%s side=%s qty=%.4f entry=%.4f",
                self.symbol, pid, side, qty, entry,
            )

        # Step 3: Merge SQLite projection
        storage = self._storage()
        if storage is not None:
            try:
                for pos_pid, pos in list(
                    getattr(tracker, "_positions", {}).items()
                ):
                    self.project_to_sqlite(pos_pid, pos)
                    report["merged"] += 1
            except Exception:
                logger.warning(
                    "[%s] on_restart: SQLite merge failed", self.symbol,
                    exc_info=True,
                )

        return report

    # ── Convenience: project a Position object directly ──

    def project_position_object(
        self,
        position: Position,
        *,
        status: Optional[PositionStatus] = None,
        exit_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
    ) -> str:
        """Convenience wrapper: project an existing ``Position`` object to SQLite.

        Converts the dataclass fields into the pos-dict format expected by
        ``project_to_sqlite()`` so callers don't need to know the internal
        key names.
        """
        if status is None:
            status = position.status
        side_val = (
            position.side.value
            if hasattr(position.side, "value")
            else str(position.side)
        )
        pos: Dict[str, Any] = {
            "side": side_val,
            "entry_price": float(position.entry_price or 0.0),
            "initial_size": float(position.initial_size or 0.0),
            "qty": float(position.current_size or position.initial_size or 0.0),
            "entry_time": position.entry_time,
            "stop_loss_price": position.stop_loss_price,
            "take_profit_price": position.take_profit_price,
            "archetype": position.archetype or position.strategy_id or "",
            "_add_position_seq": position.add_count,
            "_parent_pid": position.parent_position_id,
            "notes": position.notes,
            "unrealized_pnl": position.unrealized_pnl,
            "realized_pnl": position.realized_pnl,
        }
        return self.project_to_sqlite(
            position.position_id,
            pos,
            status=status,
            exit_price=exit_price,
            exit_reason=exit_reason,
        )

    # ── P3: Periodic Reconcile ────────────────────────────────────

    def periodic_reconcile(
        self,
        *,
        api: Any,
        tracker: Any,
        execution_yaml: Optional[Path] = None,
        archetype: str = "tpc",
        bar_minutes: int = 120,
    ) -> Dict[str, Any]:
        """Periodic position reconciliation — exchange vs tracker vs SQLite.

        Compares three sources, auto-heals discrepancies:
        - orphan open (exchange flat, SQLite open) → close SQLite + tracker
        - duplicate rows → keep newest, close rest
        - tracker missing (exchange has position) → auto-bootstrap

        Returns:
            ``issue_counts`` dict with reconciliation bucket counts.
        """
        from src.order_management.execution_truth_sync import (
            publish_reconciliation_metrics,
        )

        issue_counts: Dict[str, float] = {}

        # ── Read exchange legs (save ccxt_symbol for bootstrap) ──
        try:
            exchange_positions = api.get_positions() or []
        except Exception:
            logger.warning(
                "[%s] periodic_reconcile: exchange query failed", self.symbol,
                exc_info=True,
            )
            issue_counts["api_error"] = 1
            publish_reconciliation_metrics(
                scope="trend", strategy="all", symbol=self.symbol,
                issue_counts=issue_counts, ok=False, source="periodic_reconcile",
            )
            return issue_counts

        # Filter to this symbol
        ex_legs: Dict[str, Dict[str, Any]] = {}  # side → leg (with ccxt_symbol)
        for pos in exchange_positions:
            raw_sym = str(
                pos.get("symbol", "")
            ).replace("/", "").split(":")[0].upper().strip()
            if raw_sym != self.symbol.upper():
                continue
            qty = abs(float(pos.get("size") or pos.get("contracts") or 0))
            if qty <= 0:
                continue
            side = "short" if str(pos.get("side", "")).lower() == "short" else "long"
            ex_legs[side] = {
                "quantity": qty,
                "entry_price": float(pos.get("entry_price") or 0.0),
                "ccxt_symbol": str(pos.get("symbol", "")),  # preserve original format
            }

        # ── Read tracker memory ──
        # NOTE: periodic_reconcile runs in a thread (run_in_executor).
        # tracker._positions dict mutations are protected by GIL (single-step
        # atomicity), but logical consistency with main-thread enforce_all
        # is best-effort. See runbook "Concurrent Access" section.
        tracker_positions = getattr(tracker, "_positions", {})
        tracker_by_side: Dict[str, Dict[str, Any]] = {}
        tracker_side_qty: Dict[str, float] = {}  # aggregated qty per side (add-position)
        for pid, pos in tracker_positions.items():
            side = str(pos.get("side", "")).lower()
            if side in {"long", "buy"}:
                norm = "long"
            elif side in {"short", "sell"}:
                norm = "short"
            else:
                continue
            tracker_by_side[norm] = pos  # keep last entry for bootstrap fallback
            tracker_side_qty[norm] = tracker_side_qty.get(norm, 0.0) + float(pos.get("qty") or 0.0)

        # ── Read SQLite open ──
        storage = self._storage()
        sqlite_open: List[Any] = []
        if storage is not None:
            try:
                sqlite_open = list(storage.get_open_positions(self.symbol) or [])
            except Exception:
                pass

        # ── Check 1: exchange flat + SQLite open → sqlite_orphan_open ──
        orphan_count = 0
        for row in sqlite_open:
            row_side = row.side.value if hasattr(row.side, "value") else str(row.side)
            row_side_norm = "short" if row_side.lower() in {"short", "sell"} else "long"
            if row_side_norm not in ex_legs:
                # Exchange flat but SQLite open — auto-close SQLite
                logger.warning(
                    "[%s] periodic_reconcile: sqlite_orphan_open pid=%s side=%s",
                    self.symbol, row.position_id, row_side,
                )
                self.project_position_object(
                    row,
                    status=PositionStatus.CLOSED,
                    exit_reason="periodic_reconcile_exchange_flat",
                )
                # Sync tracker memory: remove orphan position
                tracker_positions = getattr(tracker, "_positions", {})
                if row.position_id in tracker_positions:
                    del tracker_positions[row.position_id]
                    if hasattr(tracker, "_persist_state"):
                        tracker._persist_state()
                orphan_count += 1
        if orphan_count:
            issue_counts["sqlite_orphan_open"] = orphan_count

        # ── Check 2: SQLite duplicate for same symbol+side → heal ──
        # Refresh sqlite_open after Check 1 may have closed orphans
        try:
            sqlite_open = list(storage.get_open_positions(self.symbol) or [])
        except Exception:
            sqlite_open = []  # best-effort; stale data acceptable
        from collections import defaultdict
        side_rows: Dict[str, List[Any]] = defaultdict(list)
        for row in sqlite_open:
            row_side = row.side.value if hasattr(row.side, "value") else str(row.side)
            row_side_norm = "short" if row_side.lower() in {"short", "sell"} else "long"
            side_rows[row_side_norm].append(row)
        dup_closed = 0
        for side, rows in side_rows.items():
            if len(rows) <= 1:
                continue
            # Keep newest (latest entry_time), close the rest
            def _row_ts(r: Any) -> float:
                et = getattr(r, "entry_time", None)
                if isinstance(et, datetime):
                    return et.timestamp()
                try:
                    return float(et or 0)
                except (TypeError, ValueError):
                    return 0.0
            rows_sorted = sorted(rows, key=_row_ts, reverse=True)
            for stale_row in rows_sorted[1:]:
                logger.warning(
                    "[%s] periodic_reconcile: closing duplicate pid=%s side=%s",
                    self.symbol, stale_row.position_id, side,
                )
                self.project_position_object(
                    stale_row,
                    status=PositionStatus.CLOSED,
                    exit_reason="periodic_reconcile_duplicate",
                )
                # Remove from tracker memory too
                tracker_positions = getattr(tracker, "_positions", {})
                if stale_row.position_id in tracker_positions:
                    del tracker_positions[stale_row.position_id]
                dup_closed += 1
        if dup_closed:
            if hasattr(tracker, "_persist_state"):
                tracker._persist_state()
            issue_counts["duplicate_position_row_closed"] = dup_closed

        # ── Check 1 完成后重建 tracker_by_side + tracker_side_qty ──
        tracker_positions = getattr(tracker, "_positions", {})
        tracker_by_side: Dict[str, Dict[str, Any]] = {}
        tracker_side_qty: Dict[str, float] = {}
        for pid, pos in tracker_positions.items():
            side = str(pos.get("side", "")).lower()
            if side in {"long", "buy"}:
                norm = "long"
            elif side in {"short", "sell"}:
                norm = "short"
            else:
                continue
            tracker_by_side[norm] = pos
            tracker_side_qty[norm] = tracker_side_qty.get(norm, 0.0) + float(pos.get("qty") or 0.0)

        # ── Check 3: exchange qty != tracker qty (aggregated per side) ──
        for side, ex_leg in ex_legs.items():
            t_qty = tracker_side_qty.get(side, 0.0)
            e_qty = float(ex_leg["quantity"])
            if abs(t_qty - e_qty) > max(1e-8, e_qty * 0.02):
                issue_counts["tracker_exchange_qty_mismatch"] = (
                    issue_counts.get("tracker_exchange_qty_mismatch", 0) + 1
                )

        # ── Check 4: exchange leg + tracker missing → auto-bootstrap ──
        state_path = getattr(tracker, "state_path", None)
        for side, ex_leg in ex_legs.items():
            if side not in tracker_by_side:
                logger.warning(
                    "[%s] periodic_reconcile: exchange has %s but tracker missing — bootstrapping",
                    self.symbol, side,
                )
                try:
                    # Use original ccxt_symbol from exchange response
                    ccxt_sym = ex_leg.get("ccxt_symbol") or (
                        f"{self.symbol.upper().removesuffix('USDT')}/USDT:USDT"
                    )
                    pid, pos_dict = self.bootstrap_position_from_exchange(
                        symbol=self.symbol,
                        side=side,
                        entry_price=ex_leg["entry_price"],
                        qty=ex_leg["quantity"],
                        execution_yaml=execution_yaml,
                        archetype=archetype,
                        bar_minutes=bar_minutes,
                        api=api,
                        ccxt_symbol=ccxt_sym,
                        state_path=state_path,
                    )
                    # Load into tracker memory + rekey
                    tracker_positions = getattr(tracker, "_positions", None)
                    if tracker_positions is not None:
                        tracker_positions[pid] = pos_dict
                    canonical = self.project_to_sqlite(pid, pos_dict)
                    if hasattr(tracker, "_rekey_in_memory") and canonical != pid:
                        tracker._rekey_in_memory(pid, canonical, pos_dict)
                    if hasattr(tracker, "_persist_state"):
                        tracker._persist_state()
                    issue_counts["bootstrap_from_exchange"] = (
                        issue_counts.get("bootstrap_from_exchange", 0) + 1
                    )
                except Exception:
                    logger.warning(
                        "[%s] periodic_reconcile: bootstrap failed for %s",
                        self.symbol, side, exc_info=True,
                    )

        # ── Publish metrics ──
        publish_reconciliation_metrics(
            scope="trend",
            strategy="all",
            symbol=self.symbol,
            issue_counts=issue_counts,
            source="periodic_reconcile",
        )

        logger.debug(
            "[%s] periodic_reconcile done: issues=%s",
            self.symbol, issue_counts,
        )
        return issue_counts

    # ── P4: CMS Read-Only Projection ──────────────────────────────

    @staticmethod
    def list_open_projections(
        storage: Any,
        *,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Read-only projection API for CMS — list all open positions from SQLite.

        No dedup needed: SQLite is already single-write via TTS (P1).
        """
        if storage is None:
            return []
        try:
            if symbol:
                rows = storage.get_open_positions(symbol) or []
            else:
                rows = storage.get_open_positions() or []
        except Exception:
            return []

        out: List[Dict[str, Any]] = []
        for row in rows:
            side_val = (
                row.side.value if hasattr(row.side, "value") else str(row.side)
            )
            out.append({
                "position_id": str(row.position_id or ""),
                "symbol": str(row.symbol or "").upper(),
                "side": side_val.lower(),
                "entry_price": float(row.entry_price or 0.0),
                "quantity": float(row.current_size or row.initial_size or 0.0),
                "initial_size": float(row.initial_size or 0.0),
                "entry_time": (
                    row.entry_time.isoformat() if isinstance(row.entry_time, datetime)
                    else str(row.entry_time or "")
                ),
                "stop_loss_price": row.stop_loss_price,
                "take_profit_price": row.take_profit_price,
                "strategy": str(row.strategy_id or row.archetype or ""),
                "archetype": str(row.archetype or ""),
                "unrealized_pnl": row.unrealized_pnl,
                "scope": "trend",
            })
        return out
