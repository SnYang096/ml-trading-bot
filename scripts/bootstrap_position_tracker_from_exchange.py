#!/usr/bin/env python3
"""Bootstrap PositionTracker JSON snapshots from live Binance positions.

Use when open positions exist on exchange but local tracker state was lost
(process restart before persist, or first deploy of position_tracker snapshots).

Example (on server, inside quant-trend-swing container):

  python3 scripts/bootstrap_position_tracker_from_exchange.py \\
    --state-dir /app/live/highcap/data/position_tracker \\
    --execution-yaml /app/live/highcap/config/strategies/tpc/archetypes/execution.yaml \\
    --archetype tpc
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.order_management.binance_api import BinanceAPI
from src.time_series_model.core.trade_intent import TradeIntent
from src.time_series_model.live.execution_profile_apply import (
    rr_constraints_from_exec_params,
)
from src.time_series_model.live.generic_live_strategy import ExecutionParamGenerator
from src.time_series_model.live.position_logic import build_position_dict

logger = logging.getLogger("bootstrap_position_tracker")


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return {"__datetime__": value.isoformat()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    return str(value)


def _write_tracker_state(
    *, state_path: Path, symbol: str, positions: Dict[str, Dict[str, Any]]
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "symbol": symbol,
        "positions": _to_json_safe(positions),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "_bootstrap_from_exchange": True,
    }
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    tmp.replace(state_path)


def _raw_symbol(ccxt_symbol: str) -> str:
    """ccxt symbol → Binance raw symbol (e.g. ETH/USDT:USDT → ETHUSDT)."""
    return str(ccxt_symbol or "").replace("/", "").split(":")[0].upper().strip()


def _estimate_atr(entry_price: float, atr_pct: float) -> float:
    return max(1e-6, float(entry_price) * max(1e-6, float(atr_pct)))


def _entry_time_from_trades(
    api: BinanceAPI, ccxt_symbol: str, *, side: str
) -> Optional[datetime]:
    try:
        trades = api.exchange.fetch_my_trades(ccxt_symbol, limit=50)
    except Exception as exc:
        logger.warning("fetch_my_trades failed for %s: %s", ccxt_symbol, exc)
        return None
    if not trades:
        return None
    # For a short, opening legs are sells; use oldest sell in recent window.
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


def _build_position_from_exchange(
    *,
    api: BinanceAPI,
    pos: Dict[str, Any],
    execution_cfg: Dict[str, Any],
    archetype: str,
    bar_minutes: int,
    atr_pct: float,
) -> tuple[str, Dict[str, Any]]:
    raw_sym = _raw_symbol(str(pos.get("symbol", "")))
    side = str(pos.get("side", "long")).lower()
    action = "SHORT" if side == "short" else "LONG"
    entry_price = float(pos.get("entry_price") or 0.0)
    qty = abs(float(pos.get("size") or 0.0))
    if entry_price <= 0 or qty <= 0:
        raise ValueError(f"invalid exchange position for {raw_sym}")

    gen = ExecutionParamGenerator(execution_cfg)
    exec_params = gen.generate_params(evidence_score=0.5)
    rr = rr_constraints_from_exec_params(exec_params)
    ep = {"rr_constraints": rr, "strategy_specific": {}}

    intent = TradeIntent(
        action=action,
        symbol=raw_sym,
        archetype=archetype,
        execution_profile=ep,
        position_id=f"{raw_sym.removesuffix('USDT')}:bootstrap_{int(datetime.now(timezone.utc).timestamp() * 1e6)}",
    )

    ccxt_sym = str(pos.get("symbol", ""))
    entry_time = _entry_time_from_trades(api, ccxt_sym, side=side) or datetime.now(
        timezone.utc
    )
    atr = _estimate_atr(entry_price, atr_pct)
    out = build_position_dict(
        intent=intent,
        entry_price=entry_price,
        atr=atr,
        bar_minutes=bar_minutes,
        entry_time=entry_time,
    )
    out["qty"] = qty
    out["symbol"] = raw_sym
    out["archetype"] = archetype
    out["_bootstrap_from_exchange"] = True
    return str(intent.position_id), out


def bootstrap(
    *,
    api: BinanceAPI,
    state_dir: Path,
    execution_yaml: Path,
    archetype: str,
    bar_minutes: int,
    atr_pct: float,
    dry_run: bool,
) -> int:
    execution_cfg = yaml.safe_load(execution_yaml.read_text(encoding="utf-8")) or {}
    positions = api.get_positions()
    if not positions:
        logger.info("No open exchange positions.")
        return 0

    state_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for pos in positions:
        raw_sym = _raw_symbol(str(pos.get("symbol", "")))
        pid, pos_dict = _build_position_from_exchange(
            api=api,
            pos=pos,
            execution_cfg=execution_cfg,
            archetype=archetype,
            bar_minutes=bar_minutes,
            atr_pct=atr_pct,
        )
        out_path = state_dir / f"{raw_sym}.json"
        if not dry_run:
            _write_tracker_state(
                state_path=out_path,
                symbol=raw_sym,
                positions={pid: pos_dict},
            )
        if dry_run:
            logger.info(
                "DRY RUN %s: pid=%s side=%s qty=%.6f entry=%.4f",
                raw_sym,
                pid,
                pos_dict.get("side"),
                float(pos_dict.get("qty") or 0),
                float(pos_dict.get("entry_price") or 0),
            )
        else:
            logger.info(
                "Wrote %s pid=%s side=%s qty=%.6f entry=%.4f sl=%s",
                out_path,
                pid,
                pos_dict.get("side"),
                float(pos_dict.get("qty") or 0),
                float(pos_dict.get("entry_price") or 0),
                pos_dict.get("stop_loss_price"),
            )
        written += 1
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
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
    p.add_argument(
        "--atr-pct",
        type=float,
        default=0.01,
        help="ATR estimate as fraction of entry price when rebuilding stop/R logic",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    api_key = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_FUTURES_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET") or os.getenv(
        "BINANCE_FUTURES_API_SECRET", ""
    )
    if not api_key or not api_secret:
        raise SystemExit("BINANCE_API_KEY / BINANCE_API_SECRET not set")

    api = BinanceAPI(api_key, api_secret, testnet=False)
    n = bootstrap(
        api=api,
        state_dir=Path(args.state_dir),
        execution_yaml=Path(args.execution_yaml),
        archetype=str(args.archetype).lower().strip(),
        bar_minutes=int(args.bar_minutes),
        atr_pct=float(args.atr_pct),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps({"written": n, "state_dir": str(args.state_dir)}, indent=2))


if __name__ == "__main__":
    main()
