"""Per-symbol trade map + strategy signal summary for overview table."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlbot_console.services import ohlcv_reader
from mlbot_console.services.ohlcv_reader import bars_1min_bounds
from mlbot_console.services.spot_eligibility import spot_eligibility_summary
from mlbot_console.services.strategy_registry import (
    account_layer_label,
    strategy_account_layer,
    strategies_for_layer,
)
from mlbot_console.services.trade_markers import collect_markers
from mlbot_console.services.trend_funnel import (
    aggregate_funnel_by_strategy,
    fetch_funnel_snapshots,
)

SCOPE_LABELS: Dict[str, str] = {
    "trend": "B·Trend",
    "spot": "A·Spot",
    "multi_leg": "C·Multi-leg",
}

_FUNNEL_INT_KEYS = (
    "regime_passed",
    "regime_denied",
    "prefilter_passed",
    "prefilter_denied",
    "direction",
    "gate_passed",
)


def _summarize_marker_list(markers: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {"entry": 0, "exit": 0, "pending": 0, "other": 0}
    last: Optional[Dict[str, Any]] = None
    for m in markers:
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
        "markers_total": len(markers),
        "entry": counts["entry"],
        "exit": counts["exit"],
        "pending": counts["pending"],
        "last_event": last.get("event") if last else None,
        "last_time": int(last.get("time")) if last else None,
        "last_summary": last_line,
        "summary": summary,
    }


def _summarize_markers(markers: List[Dict[str, Any]], scope: str) -> Dict[str, Any]:
    subset = [m for m in markers if m.get("scope") == scope]
    base = _summarize_marker_list(subset)
    base["scope"] = scope
    base["label"] = SCOPE_LABELS.get(scope, scope)
    return base


def _markers_by_strategy(
    markers: List[Dict[str, Any]], scope: str
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for m in markers:
        if m.get("scope") != scope:
            continue
        strat = str(m.get("strategy") or scope).lower()
        grouped.setdefault(strat, []).append(m)
    return grouped


def _funnel_summary_line(stats: Dict[str, int]) -> str:
    if not stats:
        return ""
    parts: List[str] = []
    rp = int(stats.get("regime_passed") or 0)
    rd = int(stats.get("regime_denied") or 0)
    pp = int(stats.get("prefilter_passed") or 0)
    pd = int(stats.get("prefilter_denied") or 0)
    dr = int(stats.get("direction") or 0)
    gp = int(stats.get("gate_passed") or 0)
    if rp or rd:
        parts.append(f"regime {rp}✓/{rd}✗")
    if pp or pd:
        parts.append(f"pre {pp}✓/{pd}✗")
    if dr:
        parts.append(f"dir {dr}")
    if gp:
        parts.append(f"gate {gp}✓")
    return " · ".join(parts)


def _build_scope_block(
    *,
    scope: str,
    markers: List[Dict[str, Any]],
    funnel_by_strategy: Dict[str, Dict[str, int]],
    spot_sig: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    grouped = _markers_by_strategy(markers, scope)
    by_strategy: Dict[str, Dict[str, Any]] = {}

    for strat, subset in sorted(grouped.items()):
        row = _summarize_marker_list(subset)
        row["strategy"] = strat
        row["account_layer"] = strategy_account_layer(strat)
        funnel = funnel_by_strategy.get(strat) or {}
        row["funnel"] = {k: int(funnel.get(k) or 0) for k in _FUNNEL_INT_KEYS}
        row["funnel_summary"] = _funnel_summary_line(row["funnel"])
        by_strategy[strat] = row

    if scope == "spot":
        default_strat = "spot_accum_simple"
        if default_strat not in by_strategy and spot_sig is not None:
            by_strategy[default_strat] = {
                "strategy": default_strat,
                "account_layer": "spot",
                "summary": _spot_summary(spot_sig),
                "last_summary": _spot_last_line(spot_sig),
                "can_buy": spot_sig.get("can_buy"),
                "blockers": spot_sig.get("blockers") or [],
                "weekly_ema_200_position": spot_sig.get("weekly_ema_200_position"),
                "markers_total": 0,
                "entry": 0,
                "exit": 0,
                "pending": len(spot_sig.get("pending_orders") or []),
                "funnel": {},
                "funnel_summary": "",
            }
        elif default_strat in by_strategy and spot_sig is not None:
            row = by_strategy[default_strat]
            row["can_buy"] = spot_sig.get("can_buy")
            row["blockers"] = spot_sig.get("blockers") or []
            row["weekly_ema_200_position"] = spot_sig.get("weekly_ema_200_position")
            if row.get("summary") == "—":
                row["summary"] = _spot_summary(spot_sig)

    for strat in strategies_for_layer(scope):
        if strat in by_strategy:
            continue
        funnel = funnel_by_strategy.get(strat) or {}
        if not funnel:
            continue
        by_strategy[strat] = {
            "strategy": strat,
            "account_layer": scope,
            "summary": _funnel_summary_line(funnel) or "—",
            "last_summary": "—",
            "markers_total": 0,
            "entry": 0,
            "exit": 0,
            "pending": 0,
            "funnel": {k: int(funnel.get(k) or 0) for k in _FUNNEL_INT_KEYS},
            "funnel_summary": _funnel_summary_line(funnel),
        }

    aggregate = _summarize_markers(markers, scope)
    aggregate["by_strategy"] = by_strategy
    aggregate["account_layer"] = scope
    aggregate["account_layer_label"] = account_layer_label(scope)
    return aggregate


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
    live_monitor_db: Path,
    timeframe: str = "2h",
    lookback_days: int = 7,
    funnel_windows: int = 96,
) -> List[Dict[str, Any]]:
    now = int(datetime.now(tz=timezone.utc).timestamp())
    since_ts = now - lookback_days * 86400
    funnel_snapshots = fetch_funnel_snapshots(
        live_monitor_db, symbol="", limit=funnel_windows
    )
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

        sym_funnel = aggregate_funnel_by_strategy(funnel_snapshots, symbol=sym_u)
        trend_funnel = {
            k: v for k, v in sym_funnel.items() if strategy_account_layer(k) == "trend"
        }
        spot_funnel = {
            k: v for k, v in sym_funnel.items() if strategy_account_layer(k) == "spot"
        }
        multileg_funnel = {
            k: v
            for k, v in sym_funnel.items()
            if strategy_account_layer(k) == "multi_leg"
        }

        rows.append(
            {
                "symbol": sym_u,
                "latest_bar": latest,
                "bars_1min_rows": bars_rows,
                "map_href": f"/trade-map?symbol={sym_u}",
                "strategies": {
                    "trend": _build_scope_block(
                        scope="trend",
                        markers=markers,
                        funnel_by_strategy=trend_funnel,
                    ),
                    "spot": _build_scope_block(
                        scope="spot",
                        markers=markers,
                        funnel_by_strategy=spot_funnel,
                        spot_sig=spot_sig,
                    ),
                    "multi_leg": _build_scope_block(
                        scope="multi_leg",
                        markers=markers,
                        funnel_by_strategy=multileg_funnel,
                    ),
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


def _spot_last_line(spot_sig: Dict[str, Any]) -> str:
    pending = spot_sig.get("pending_orders") or []
    if pending:
        p0 = pending[0]
        return f"pending {p0.get('side')} @{p0.get('price')}"
    if spot_sig.get("can_buy"):
        return "eligible"
    blockers = spot_sig.get("blockers") or []
    if blockers:
        return f"blocked: {blockers[0]}"
    return "—"
