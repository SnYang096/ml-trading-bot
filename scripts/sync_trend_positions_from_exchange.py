#!/usr/bin/env python3
"""Mirror B·Trend exchange futures positions into local SQLite (+ optional JSON).

Use when console / account summary diverges from Binance (orphan local rows,
missing bootstrap after manual exchange fills, or JSON tracker out of sync).

This is **not** a substitute for live PositionTracker on new entries — it
reconciles **reporting** state from the exchange as source of truth.

Example (inside quant-trend-swing container):

  python3 scripts/sync_trend_positions_from_exchange.py --dry-run
  python3 scripts/sync_trend_positions_from_exchange.py --write-json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.order_management.binance_api import BinanceAPI
from src.order_management.models import Position, PositionSide, PositionStatus
from src.order_management.storage import Storage
from src.order_management.trend_position_truth_sync import TrendPositionTruthSync

logger = logging.getLogger("sync_trend_positions_from_exchange")


def _raw_symbol(ccxt_symbol: str) -> str:
    return str(ccxt_symbol or "").replace("/", "").split(":")[0].upper().strip()


def _norm_side(side: str) -> str:
    s = str(side or "").lower()
    if s in {"buy", "long"}:
        return "long"
    if s in {"sell", "short"}:
        return "short"
    return s


def _exchange_legs(api: BinanceAPI) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """symbol, side → exchange leg (qty, entry, upnl, mark)."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for pos in api.get_positions() or []:
        sym = _raw_symbol(str(pos.get("symbol") or ""))
        side = _norm_side(str(pos.get("side") or ""))
        qty = abs(float(pos.get("size") or 0.0))
        if not sym or qty <= 0:
            continue
        key = (sym, side)
        out[key] = {
            "symbol": sym,
            "side": side,
            "quantity": qty,
            "entry_price": float(pos.get("entry_price") or 0.0),
            "mark_price": float(pos.get("mark_price") or 0.0),
            "unrealized_pnl_usdt": float(pos.get("unrealized_pnl") or 0.0),
            "ccxt_symbol": str(pos.get("symbol") or ""),
        }
    return out


def _local_open_rows(storage: Storage) -> List[Position]:
    return list(storage.get_open_positions() or [])


def _group_local(
    rows: Iterable[Position],
) -> Dict[Tuple[str, str], List[Position]]:
    grouped: Dict[Tuple[str, str], List[Position]] = defaultdict(list)
    for row in rows:
        sym = str(row.symbol or "").upper()
        side = _norm_side(
            row.side.value if hasattr(row.side, "value") else str(row.side)
        )
        grouped[(sym, side)].append(row)
    return grouped


def _bootstrap_position_id(symbol: str) -> str:
    # DEPRECATED (P2): prefer TrendPositionTruthSync._make_pid()
    base = symbol.removesuffix("USDT") if symbol.endswith("USDT") else symbol
    return f"{base}:exchange_sync_{int(datetime.now(timezone.utc).timestamp() * 1e6)}"


def _make_open_position(
    *,
    leg: Mapping[str, Any],
    strategy_id: str,
) -> Position:
    # DEPRECATED (P2): prefer TrendPositionTruthSync.bootstrap_position_from_exchange()
    sym = str(leg["symbol"])
    side = PositionSide.LONG if leg["side"] == "long" else PositionSide.SHORT
    qty = float(leg["quantity"])
    entry = float(leg["entry_price"])
    now = datetime.now(timezone.utc)
    return Position(
        position_id=_bootstrap_position_id(sym),
        symbol=sym,
        side=side,
        entry_time=now,
        entry_price=entry,
        initial_size=qty,
        current_size=qty,
        total_cost=entry * qty,
        status=PositionStatus.OPEN,
        strategy_id=strategy_id,
        archetype=strategy_id,
        notes="exchange_sync",
    )


def _fetch_mark_prices(api: BinanceAPI, symbols: Iterable[str]) -> Dict[str, float]:
    """Best-effort mark/last for stale closes when exchange leg is flat."""
    out: Dict[str, float] = {}
    for sym in sorted({str(s or "").upper() for s in symbols if str(s or "").strip()}):
        try:
            px = api.get_ticker_price(sym)
            if px is not None and float(px) > 0:
                out[sym] = float(px)
        except Exception as exc:
            logger.warning("mark price fetch failed for %s: %s", sym, exc)
    return out


def _close_position_row(
    row: Position,
    *,
    reason: str,
    closing_price: Optional[float] = None,
) -> Position:
    row.status = PositionStatus.CLOSED
    row.exit_time = datetime.now(timezone.utc)
    row.exit_reason = reason
    qty = float(row.current_size or row.initial_size or 0.0)
    entry = float(row.entry_price or 0.0)
    row.current_size = 0.0
    # Prefer exchange mark; fallback to stored unrealized or entry price.
    if closing_price is not None and closing_price > 0:
        row.exit_price = float(closing_price)
    elif row.unrealized_pnl is not None and entry > 0 and qty > 0:
        upnl = float(row.unrealized_pnl)
        side = _norm_side(
            row.side.value if hasattr(row.side, "value") else str(row.side)
        )
        if side == "long":
            row.exit_price = entry + upnl / qty
        else:
            row.exit_price = entry - upnl / qty
    elif row.exit_price is None and entry > 0:
        row.exit_price = entry
    # Compute realized PnL
    if row.exit_price is not None and entry > 0 and qty > 0:
        ep = float(row.exit_price)
        side = _norm_side(
            row.side.value if hasattr(row.side, "value") else str(row.side)
        )
        if side == "long":
            row.realized_pnl = (ep - entry) * qty
        else:
            row.realized_pnl = (entry - ep) * qty
    return row


def sync_trend_positions(
    *,
    api: BinanceAPI,
    db_path: Path,
    strategy_id: str = "tpc",
    qty_tol_pct: float = 0.02,
    dry_run: bool = True,
    close_stale: bool = True,
) -> Dict[str, Any]:
    storage = Storage(str(db_path))

    # ── Per-symbol TTS: SQLite 投影唯一写入口 ──
    _tts_by_symbol: Dict[str, TrendPositionTruthSync] = {}

    def _tts_for(sym: str) -> TrendPositionTruthSync:
        return _tts_by_symbol.setdefault(
            sym,
            TrendPositionTruthSync(symbol=sym, storage_factory=lambda: storage),
        )

    exchange = _exchange_legs(api)
    local_rows = _local_open_rows(storage)
    local = _group_local(local_rows)

    report: Dict[str, Any] = {
        "dry_run": dry_run,
        "db_path": str(db_path),
        "exchange_legs": len(exchange),
        "local_open_rows": len(local_rows),
        "inserted": [],
        "updated": [],
        "closed": [],
        "unchanged": [],
    }

    for key, leg in exchange.items():
        sym, side = key
        qty_ex = float(leg["quantity"])
        locals_for_key = list(local.get(key) or [])
        loc_qty = sum(float(p.current_size or 0.0) for p in locals_for_key)
        tol = max(1e-8, qty_ex * qty_tol_pct)

        if not locals_for_key:
            pid = _bootstrap_position_id(sym)
            action = {
                "kind": "insert",
                "position_id": pid,
                "symbol": sym,
                "side": side,
                "quantity": qty_ex,
                "entry_price": leg["entry_price"],
            }
            report["inserted"].append(action)
            if not dry_run:
                new_pos = _make_open_position(leg=leg, strategy_id=strategy_id)
                _tts_for(sym).project_position_object(new_pos)
            continue

        if abs(loc_qty - qty_ex) <= tol:
            report["unchanged"].append(
                {"symbol": sym, "side": side, "quantity": qty_ex}
            )
            continue

        primary = sorted(
            locals_for_key,
            key=lambda p: float(p.current_size or 0.0),
            reverse=True,
        )[0]
        action = {
            "kind": "resize",
            "position_id": primary.position_id,
            "symbol": sym,
            "side": side,
            "local_qty": loc_qty,
            "exchange_qty": qty_ex,
            "entry_price": leg["entry_price"],
        }
        report["updated"].append(action)
        if not dry_run:
            primary.current_size = qty_ex
            primary.initial_size = max(float(primary.initial_size or 0.0), qty_ex)
            primary.entry_price = float(
                leg["entry_price"] or primary.entry_price or 0.0
            )
            primary.unrealized_pnl = float(leg.get("unrealized_pnl_usdt") or 0.0)
            _tts_for(sym).project_position_object(primary)

        extras = [p for p in locals_for_key if p.position_id != primary.position_id]
        if extras and close_stale:
            dup_mark = (
                float(leg.get("mark_price") or 0.0)
                if float(leg.get("mark_price") or 0.0) > 0
                else None
            )
            for row in extras:
                closed = {
                    "kind": "close_duplicate",
                    "position_id": row.position_id,
                    "symbol": sym,
                    "side": side,
                }
                report["closed"].append(closed)
                if not dry_run:
                    closed_row = _close_position_row(
                        row,
                        reason="exchange_sync_duplicate",
                        closing_price=dup_mark,
                    )
                    _tts_for(sym).project_position_object(
                        closed_row,
                        status=PositionStatus.CLOSED,
                        exit_price=closed_row.exit_price,
                        exit_reason=closed_row.exit_reason,
                    )

    if close_stale:
        stale_symbols: List[str] = []
        for key, rows in local.items():
            if key in exchange:
                continue
            for row in rows:
                stale_symbols.append(str(row.symbol or "").upper())
        mark_by_symbol = _fetch_mark_prices(api, stale_symbols)

        for key, rows in local.items():
            if key in exchange:
                continue
            for row in rows:
                sym_u = str(row.symbol or "").upper()
                stale_mark = mark_by_symbol.get(sym_u)
                closed = {
                    "kind": "close_stale",
                    "position_id": row.position_id,
                    "symbol": row.symbol,
                    "side": _norm_side(
                        row.side.value if hasattr(row.side, "value") else str(row.side)
                    ),
                    "closing_price": stale_mark,
                }
                report["closed"].append(closed)
                if not dry_run:
                    closed_row = _close_position_row(
                        row,
                        reason="exchange_sync_flat",
                        closing_price=stale_mark,
                    )
                    _tts_for(str(row.symbol or "").upper()).project_position_object(
                        closed_row,
                        status=PositionStatus.CLOSED,
                        exit_price=closed_row.exit_price,
                        exit_reason=closed_row.exit_reason,
                    )

    report["summary"] = {
        "insert": len(report["inserted"]),
        "update": len(report["updated"]),
        "close": len(report["closed"]),
        "unchanged": len(report["unchanged"]),
    }
    return report


def _write_json_snapshots(
    *,
    api: BinanceAPI,
    state_dir: Path,
    execution_yaml: Path,
    archetype: str,
    bar_minutes: int,
    atr_pct: float,
    dry_run: bool,
) -> int:
    boot_path = _REPO_ROOT / "scripts" / "bootstrap_position_tracker_from_exchange.py"
    spec = importlib.util.spec_from_file_location("bootstrap_pt", boot_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load bootstrap module from {boot_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return int(
        mod.bootstrap(
            api=api,
            state_dir=state_dir,
            execution_yaml=execution_yaml,
            archetype=archetype,
            bar_minutes=bar_minutes,
            atr_pct=atr_pct,
            dry_run=dry_run,
        )
    )


def disaster_recovery(
    *,
    api: BinanceAPI,
    db_path: Path,
    state_dir: Path,
    execution_yaml: Path,
    archetype: str = "tpc",
    bar_minutes: int = 120,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """DR mode: rebuild all JSON + SQLite from exchange using TTS.on_restart()."""
    storage = Storage(str(db_path))
    exchange_positions = api.get_positions() or []

    report: Dict[str, Any] = {
        "mode": "disaster_recovery",
        "dry_run": dry_run,
        "exchange_legs": len(exchange_positions),
        "bootstrapped": [],
    }

    for pos in exchange_positions:
        raw_sym = _raw_symbol(str(pos.get("symbol") or ""))
        if not raw_sym:
            continue
        qty = abs(float(pos.get("size") or pos.get("contracts") or 0))
        if qty <= 0:
            continue
        side = "short" if str(pos.get("side", "")).lower() == "short" else "long"
        entry = float(pos.get("entry_price") or 0.0)
        if entry <= 0:
            continue

        ccxt_sym = str(pos.get("symbol", ""))
        tts = TrendPositionTruthSync(
            symbol=raw_sym,
            storage_factory=lambda: storage,
        )

        state_path = state_dir / f"{raw_sym}.json"
        exec_yaml = execution_yaml if execution_yaml.is_file() else None

        if dry_run:
            pid = TrendPositionTruthSync._make_pid(raw_sym)
            report["bootstrapped"].append(
                {
                    "symbol": raw_sym,
                    "side": side,
                    "quantity": qty,
                    "entry_price": entry,
                    "position_id": pid,
                    "dry_run": True,
                }
            )
            logger.info(
                "DR DRY RUN %s: pid=%s side=%s qty=%.6f entry=%.4f",
                raw_sym,
                pid,
                side,
                qty,
                entry,
            )
            continue

        pid, pos_dict = TrendPositionTruthSync.bootstrap_position_from_exchange(
            symbol=raw_sym,
            side=side,
            entry_price=entry,
            qty=qty,
            execution_yaml=exec_yaml,
            archetype=archetype,
            bar_minutes=bar_minutes,
            api=api,
            ccxt_symbol=ccxt_sym,
            state_path=state_path,
        )
        tts.project_to_sqlite(pid, pos_dict)
        report["bootstrapped"].append(
            {
                "symbol": raw_sym,
                "side": side,
                "quantity": qty,
                "entry_price": entry,
                "position_id": pid,
            }
        )
        logger.info(
            "DR %s: pid=%s side=%s qty=%.6f entry=%.4f sl=%s",
            raw_sym,
            pid,
            side,
            qty,
            entry,
            pos_dict.get("stop_loss_price"),
        )

    report["summary"] = {"bootstrapped": len(report["bootstrapped"])}
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--db-path",
        default=os.getenv(
            "MLBOT_ORDER_MANAGEMENT_DB_PATH",
            str(_REPO_ROOT / "data" / "order_management.db"),
        ),
    )
    p.add_argument("--strategy-id", default="tpc")
    p.add_argument("--qty-tol-pct", type=float, default=0.02)
    p.add_argument(
        "--no-close-stale",
        action="store_true",
        help="Do not close local open rows when exchange is flat on that symbol+side",
    )
    p.add_argument(
        "--write-json",
        action="store_true",
        help="Also rewrite position_tracker/*.json via bootstrap helper",
    )
    p.add_argument(
        "--state-dir",
        default=os.getenv(
            "MLBOT_POSITION_TRACKER_STATE_DIR",
            "live/highcap/data/position_tracker",
        ),
    )
    p.add_argument(
        "--execution-yaml",
        default="live/highcap/config/strategies/tpc/archetypes/execution.yaml",
    )
    p.add_argument("--archetype", default="tpc")
    p.add_argument("--bar-minutes", type=int, default=120)
    p.add_argument("--atr-pct", type=float, default=0.01)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--disaster-recovery",
        action="store_true",
        help="DR mode: rebuild all JSON + SQLite from exchange via TTS.on_restart()",
    )
    args = p.parse_args()

    api_key = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_FUTURES_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv(
        "BINANCE_FUTURES_API_SECRET", ""
    )
    if not api_key or not api_secret:
        raise SystemExit("BINANCE_API_KEY / BINANCE_API_SECRET not set")

    api = BinanceAPI(api_key, api_secret, testnet=False)

    if args.disaster_recovery:
        report = disaster_recovery(
            api=api,
            db_path=Path(args.db_path),
            state_dir=Path(args.state_dir),
            execution_yaml=Path(args.execution_yaml),
            archetype=str(args.archetype).lower().strip(),
            bar_minutes=int(args.bar_minutes),
            dry_run=bool(args.dry_run),
        )
    else:
        report = sync_trend_positions(
            api=api,
            db_path=Path(args.db_path),
            strategy_id=str(args.strategy_id).lower().strip(),
            qty_tol_pct=float(args.qty_tol_pct),
            dry_run=bool(args.dry_run),
            close_stale=not bool(args.no_close_stale),
        )
        if args.write_json:
            report["json_written"] = _write_json_snapshots(
                api=api,
                state_dir=Path(args.state_dir),
                execution_yaml=Path(args.execution_yaml),
                archetype=str(args.archetype).lower().strip(),
                bar_minutes=int(args.bar_minutes),
                atr_pct=float(args.atr_pct),
                dry_run=bool(args.dry_run),
            )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
