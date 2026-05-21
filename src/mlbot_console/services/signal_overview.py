"""Per-symbol trade map + strategy signal summary for overview table."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlbot_console.services import ohlcv_reader
from mlbot_console.services.ohlcv_reader import bars_1min_bounds
from mlbot_console.services.spot_eligibility import spot_eligibility_summary
from mlbot_console.services.trade_markers import collect_markers

SCOPE_LABELS: Dict[str, str] = {
    "trend": "B·Trend",
    "spot": "A·Spot",
    "multi_leg": "C·Multi-leg",
}


def _summarize_markers(markers: List[Dict[str, Any]], scope: str) -> Dict[str, Any]:
    subset = [m for m in markers if m.get("scope") == scope]
    counts: Dict[str, int] = {"entry": 0, "exit": 0, "pending": 0, "other": 0}
    last: Optional[Dict[str, Any]] = None
    for m in subset:
        ev = str(m.get("event") or "other").lower()
        st = str(m.get("status") or "filled").lower()
        if st == "pending":
            counts["pending"] += 1
        elif ev in counts:
            counts[ev] += 1
        else:
            counts["other"] += 1
        if last is None or int(m.get("time") or 0) >= int(last.get("time") or 0):
            last = m
    parts: List[str] = []
    if counts["entry"]:
        parts.append(f"{counts['entry']} 开")
    if counts["exit"]:
        parts.append(f"{counts['exit']} 平")
    if counts["pending"]:
        parts.append(f"{counts['pending']} pending")
    summary = ", ".join(parts) if parts else "—"
    last_line = "—"
    if last:
        last_line = (
            f"{last.get('event')} {last.get('side')} "
            f"@{_fmt_ts(int(last.get('time') or 0))}"
        )
    return {
        "scope": scope,
        "label": SCOPE_LABELS.get(scope, scope),
        "markers_total": len(subset),
        "entry": counts["entry"],
        "exit": counts["exit"],
        "pending": counts["pending"],
        "last_event": last.get("event") if last else None,
        "last_time": int(last.get("time")) if last else None,
        "last_summary": last_line,
        "summary": summary,
    }


def _fmt_ts(ts: int) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m-%d %H:%M")


def build_signal_overview(
    symbols: List[str],
    *,
    feature_bus_root: Path,
    trend_db: Path,
    spot_db: Path,
    multi_leg_db: Path,
    timeframe: str = "2h",
    lookback_days: int = 7,
) -> List[Dict[str, Any]]:
    now = int(datetime.now(tz=timezone.utc).timestamp())
    since_ts = now - lookback_days * 86400
    rows: List[Dict[str, Any]] = []

    for sym in symbols:
        sym_u = sym.upper()
        latest = ohlcv_reader.latest_bar_meta(feature_bus_root, sym_u)
        path = feature_bus_root / "bars_1min" / f"{sym_u}.parquet"
        _, _, bars_rows = bars_1min_bounds(path)

        markers = collect_markers(
            trend_db=trend_db,
            spot_db=spot_db,
            multi_leg_db=multi_leg_db,
            symbol=sym_u,
            scopes=["trend", "spot", "multi_leg"],
            start_ts=since_ts,
            end_ts=now,
            include_pending=True,
        )

        spot_sig = spot_eligibility_summary(
            feature_bus_root=feature_bus_root,
            spot_db=spot_db,
            symbol=sym_u,
            timeframe=timeframe,
        )
        spot_row = {
            "scope": "spot",
            "label": "A·Spot",
            "can_buy": spot_sig.get("can_buy"),
            "weekly_ema_200_position": spot_sig.get("weekly_ema_200_position"),
            "blockers": spot_sig.get("blockers") or [],
            "pending_orders": len(spot_sig.get("pending_orders") or []),
            "summary": _spot_summary(spot_sig),
        }

        rows.append(
            {
                "symbol": sym_u,
                "latest_bar": latest,
                "bars_1min_rows": bars_rows,
                "map_href": f"/trade-map?symbol={sym_u}",
                "strategies": {
                    "trend": _summarize_markers(markers, "trend"),
                    "spot": spot_row,
                    "multi_leg": _summarize_markers(markers, "multi_leg"),
                },
            }
        )
    return rows


def _spot_summary(spot_sig: Dict[str, Any]) -> str:
    if spot_sig.get("can_buy"):
        base = "可买"
    else:
        base = "不可买"
    wk = spot_sig.get("weekly_ema_200_position")
    wk_s = f"{wk:.4f}" if wk is not None and wk == wk else "n/a"
    blockers = spot_sig.get("blockers") or []
    extra = f" · blockers={','.join(blockers)}" if blockers else ""
    pending = spot_sig.get("pending_orders") or []
    pend_s = f" · {len(pending)} pending" if pending else ""
    return f"{base} · wk={wk_s}{pend_s}{extra}"
