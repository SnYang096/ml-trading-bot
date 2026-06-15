"""Telegram alerts for multi-leg account equity moves and new exchange positions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from src.monitoring.telegram import (
    format_account_equity_change_message,
    format_account_open_position_message,
    send_account_watch_alerts,
)
from src.time_series_model.core.constitution.account_risk_guard import (
    snapshot_from_binance_balance,
)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def position_keys(positions: Iterable[Mapping[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for pos in positions:
        sym = str(pos.get("symbol") or "").upper()
        if not sym:
            continue
        try:
            amt = float(pos.get("positionAmt") or pos.get("quantity") or 0.0)
        except (TypeError, ValueError):
            amt = 0.0
        if amt == 0.0:
            side = str(pos.get("side") or "").lower()
            if side in {"long", "short"}:
                out.add(f"{sym}:{side}")
            continue
        side = "long" if amt > 0 else "short"
        out.add(f"{sym}:{side}")
    return out


def detect_new_positions(
    previous: Set[str], current: Set[str]
) -> List[str]:
    return sorted(current - previous)


@dataclass
class AccountWatchState:
    anchor_equity: float = 0.0
    open_positions: Set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> "AccountWatchState":
        if not path.is_file():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            anchor_equity=float(raw.get("anchor_equity") or 0.0),
            open_positions=set(str(x) for x in (raw.get("open_positions") or [])),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "anchor_equity": self.anchor_equity,
                    "open_positions": sorted(self.open_positions),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _make_multi_leg_api(testnet: bool):
    from src.order_management.binance_api import BinanceAPI

    if testnet:
        api_key = os.getenv("MULTI_LEG_BINANCE_FUTURES_TESTNET_API_KEY", "")
        api_secret = os.getenv("MULTI_LEG_BINANCE_FUTURES_TESTNET_API_SECRET", "")
    else:
        api_key = os.getenv("MULTI_LEG_BINANCE_FUTURES_API_KEY", "") or os.getenv(
            "MULTI_LEG_BINANCE_API_KEY", ""
        )
        api_secret = os.getenv("MULTI_LEG_BINANCE_FUTURES_API_SECRET", "") or os.getenv(
            "MULTI_LEG_BINANCE_API_SECRET", ""
        )
    if not api_key or not api_secret:
        raise RuntimeError(
            "multi-leg API keys missing "
            "(MULTI_LEG_BINANCE_FUTURES_TESTNET_* or MULTI_LEG_BINANCE_FUTURES_*)"
        )
    return BinanceAPI(
        api_key=api_key, api_secret=api_secret, testnet=testnet, use_proxy=None
    )


def fetch_multi_leg_snapshot(*, testnet: bool) -> Tuple[float, List[Dict[str, Any]]]:
    api = _make_multi_leg_api(testnet=testnet)
    snap = snapshot_from_binance_balance(
        balance=api.get_account_balance(),
        positions=api.get_positions(),
    )
    positions = api.get_positions()
    return float(snap.equity or 0.0), list(positions or [])


def run_account_watch_once(
    *,
    scope: str = "multi_leg",
    state_path: Path,
    testnet: bool = True,
    change_threshold_pct: Optional[float] = None,
    dry_run: bool = False,
    force_notify: bool = False,
) -> Dict[str, Any]:
    threshold = (
        float(change_threshold_pct)
        if change_threshold_pct is not None
        else _env_float("MLBOT_ACCOUNT_TG_CHANGE_PCT", 0.03)
    )
    state = AccountWatchState.load(state_path)
    equity, positions = fetch_multi_leg_snapshot(testnet=testnet)
    current_keys = position_keys(positions)
    new_keys = detect_new_positions(state.open_positions, current_keys)

    if state.anchor_equity <= 0.0 and equity > 0.0:
        state.anchor_equity = equity

    messages: List[str] = []
    equity_msg = format_account_equity_change_message(
        scope=scope,
        anchor=state.anchor_equity,
        current=equity,
        threshold_pct=threshold,
    )
    if equity_msg:
        messages.append(equity_msg)
        state.anchor_equity = equity

    if new_keys:
        messages.append(format_account_open_position_message(scope=scope, keys=new_keys))

    state.open_positions = current_keys
    if not dry_run:
        state.save(state_path)

    sent = 0
    if dry_run:
        for msg in messages:
            print(msg)
    else:
        sent = send_account_watch_alerts(
            messages, scope=scope, force_notify=force_notify
        )

    return {
        "scope": scope,
        "testnet": testnet,
        "equity_usdt": equity,
        "anchor_equity_usdt": state.anchor_equity,
        "change_threshold_pct": threshold,
        "open_positions": sorted(current_keys),
        "new_positions": new_keys,
        "messages": messages,
        "telegram_sent": sent,
        "dry_run": dry_run,
    }
