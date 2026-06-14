"""Rebalance advisor: composite risk-on + NAV band alerts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from mlbot_console.services.exchange_balances import build_exchange_ledger
from mlbot_console.services.regime_live import build_live_layers
from mlbot_console.services.regime_ops import fetch_regime_ops_snapshot

_COMPOSITE_LABELS = {
    "risk_on": "risk-on（偏多 beta）",
    "neutral": "中性",
    "risk_off": "risk-off（收缩 beta）",
}

_SCOPE_LABELS = {
    "spot": "A·Spot",
    "rolling": "A·Rolling",
    "trend": "B·Trend",
    "multi_leg": "C·Multi-leg",
}

_ALERT_RANK = {"OK": 0, "WATCH": 1, "REBALANCE_SUGGEST": 2}


def _max_alert(current: str, candidate: str) -> str:
    if _ALERT_RANK.get(candidate, 0) > _ALERT_RANK.get(current, 0):
        return candidate
    return current


def load_rebalance_config(project_root: Path) -> Dict[str, Any]:
    path = project_root / "config" / "monitoring" / "rebalance_targets.yaml"
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _eval_atomic(when: str, ctx: Mapping[str, Any]) -> bool:
    clause = str(when or "").strip()
    if not clause:
        return False
    m = re.match(
        r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*(==|>=|<=|>|<)\s*(.+)$",
        clause,
    )
    if not m:
        return False
    key, op, raw_rhs = m.group(1), m.group(2), m.group(3).strip()
    lhs = ctx.get(key)
    if lhs is None:
        return False
    rhs_raw = raw_rhs.strip().strip("'\"")
    try:
        rhs_num = float(rhs_raw)
        lhs_num = float(lhs)
        if op == ">=":
            return lhs_num >= rhs_num
        if op == "<=":
            return lhs_num <= rhs_num
        if op == ">":
            return lhs_num > rhs_num
        if op == "<":
            return lhs_num < rhs_num
        if op == "==":
            return lhs_num == rhs_num
        return False
    except (TypeError, ValueError):
        if op != "==":
            return False
        return str(lhs).strip().lower() == rhs_raw.strip().lower()


def _eval_when(when: str, ctx: Mapping[str, Any]) -> bool:
    parts = re.split(r"\s+and\s+", str(when or "").strip(), flags=re.IGNORECASE)
    return all(_eval_atomic(p.strip(), ctx) for p in parts if p.strip())


def _score_input(inp: Dict[str, Any], ctx: Mapping[str, Any]) -> int:
    best = 0
    for rule in inp.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        when = rule.get("when")
        if when and _eval_when(str(when), ctx):
            try:
                best = max(best, int(rule.get("score") or 0))
            except (TypeError, ValueError):
                pass
    return best


def compute_composite(
    ctx: Mapping[str, Any], config: Mapping[str, Any]
) -> Dict[str, Any]:
    comp = config.get("composite") or {}
    total = 0
    breakdown: List[Dict[str, Any]] = []
    for inp in comp.get("inputs") or []:
        if not isinstance(inp, dict):
            continue
        try:
            weight = int(inp.get("weight") or 1)
        except (TypeError, ValueError):
            weight = 1
        score = _score_input(inp, ctx)
        weighted = score * weight
        total += weighted
        breakdown.append(
            {
                "id": inp.get("id"),
                "score": score,
                "weight": weight,
                "weighted": weighted,
            }
        )
    label = "neutral"
    for row in comp.get("map") or []:
        if not isinstance(row, dict):
            continue
        try:
            cap = int(row.get("max_total") or 0)
        except (TypeError, ValueError):
            continue
        if total <= cap:
            label = str(row.get("label") or "neutral")
            break
    return {
        "label": label,
        "label_title": _COMPOSITE_LABELS.get(label, label),
        "total_score": total,
        "breakdown": breakdown,
    }


def _scope_equity(
    ledger: Mapping[str, Any], scope: str
) -> Tuple[Optional[float], bool]:
    for row in ledger.get("accounts") or []:
        if str(row.get("scope")) != scope:
            continue
        if not row.get("ok"):
            return None, False
        return float(row.get("equity_usdt") or 0.0), True
    return None, False


def _band_status(
    nav_pct: float,
    band: Mapping[str, float],
    *,
    tolerance: float,
    hard_tolerance: float,
) -> str:
    target = float(band.get("target") or 0.0)
    lo = float(band.get("min") or 0.0)
    hi = float(band.get("max") or 0.0)
    if nav_pct < lo or nav_pct > hi:
        return "OUT_OF_BAND"
    delta = abs(nav_pct - target)
    if delta > hard_tolerance:
        return "REBALANCE_SUGGEST"
    if delta > tolerance:
        return "WATCH"
    return "OK"


def build_allocation(
    *,
    ledger: Mapping[str, Any],
    composite_label: str,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    scopes_cfg = config.get("scopes") or {}
    bands_all = (config.get("bands") or {}).get(composite_label) or {}
    tolerance = float(config.get("tolerance_pct") or 0.05)
    hard_tol = float(config.get("hard_tolerance_pct") or 0.12)

    equities: Dict[str, float] = {}
    present: Dict[str, bool] = {}
    for scope in scopes_cfg:
        eq, ok = _scope_equity(ledger, scope)
        meta = scopes_cfg.get(scope) or {}
        if meta.get("optional") and not ok:
            continue
        if eq is not None and ok:
            equities[scope] = eq
            present[scope] = True
        elif not meta.get("optional"):
            equities[scope] = 0.0
            present[scope] = False

    total_nav = sum(equities.values())
    scope_rows: List[Dict[str, Any]] = []
    worst = "OK"
    rank = {"OK": 0, "WATCH": 1, "OUT_OF_BAND": 2, "REBALANCE_SUGGEST": 3, "MISSING": 1}

    for scope, meta in scopes_cfg.items():
        if scope not in equities and meta.get("optional"):
            scope_rows.append(
                {
                    "scope": scope,
                    "label": meta.get("label") or _SCOPE_LABELS.get(scope, scope),
                    "layer": meta.get("layer"),
                    "optional": True,
                    "configured": False,
                    "equity_usdt": None,
                    "nav_pct": None,
                    "status": "NOT_CONFIGURED",
                }
            )
            continue
        eq = equities.get(scope, 0.0)
        present_scope = present.get(scope, False)
        if not present_scope:
            st = "MISSING"
            nav_pct = None
        else:
            nav_pct = (eq / total_nav) if total_nav > 0 else None
            band = bands_all.get(scope) or {}
            if band:
                st = _band_status(
                    float(nav_pct or 0.0),
                    band,
                    tolerance=tolerance,
                    hard_tolerance=hard_tol,
                )
            else:
                st = "OK"
        band = bands_all.get(scope) or {}
        if rank.get(st, 0) > rank.get(worst, 0):
            worst = st
        scope_rows.append(
            {
                "scope": scope,
                "label": meta.get("label") or _SCOPE_LABELS.get(scope, scope),
                "layer": meta.get("layer"),
                "optional": bool(meta.get("optional")),
                "configured": present_scope,
                "equity_usdt": eq if present_scope else None,
                "nav_pct": nav_pct,
                "band": band or None,
                "status": st,
            }
        )

    alert = "OK"
    if worst in {"OUT_OF_BAND", "REBALANCE_SUGGEST"}:
        alert = "REBALANCE_SUGGEST"
    elif worst in {"WATCH", "MISSING"}:
        alert = "WATCH"

    return {
        "total_nav_usdt": total_nav,
        "scopes": scope_rows,
        "alert": alert,
        "tolerance_pct": tolerance,
        "hard_tolerance_pct": hard_tol,
    }


def build_suggestions(
    *,
    allocation: Mapping[str, Any],
    composite: Mapping[str, Any],
    layers: Mapping[str, Any],
) -> List[str]:
    out: List[str] = []
    comp = str(composite.get("label") or "neutral")
    a_beta = _a_beta_share(allocation)
    for row in allocation.get("scopes") or []:
        st = row.get("status")
        scope = row.get("scope")
        label = row.get("label")
        nav_pct = row.get("nav_pct")
        band = row.get("band") or {}
        target = band.get("target")
        if st in {"WATCH", "OUT_OF_BAND", "REBALANCE_SUGGEST"} and nav_pct is not None:
            if target is not None and nav_pct < float(target):
                out.append(
                    f"{label} 占比 {nav_pct:.0%} 低于目标 {float(target):.0%}"
                    f"（composite={comp}）— 可考虑增加 beta 敞口"
                )
            elif target is not None and nav_pct > float(target):
                out.append(
                    f"{label} 占比 {nav_pct:.0%} 高于目标 {float(target):.0%}"
                    f"（composite={comp}）— 可考虑减仓"
                )
    if comp == "risk_on" and a_beta < 0.15:
        out.append(
            "composite=risk_on 但 A 层（spot+rolling）合计偏低 — "
            "牛市宜提高现货/长期持仓占比，勿仅靠 B 加长持有"
        )
    if comp == "risk_off":
        b_row = next(
            (r for r in allocation.get("scopes") or [] if r.get("scope") == "trend"),
            None,
        )
        if b_row and float(b_row.get("nav_pct") or 0) > 0.5:
            out.append(
                "composite=risk_off 但 B·Trend 占比仍高 — 检查是否应用紧失效、避免裸扛 beta"
            )
    bus = (layers.get("feature_bus") or {}) if isinstance(layers, dict) else {}
    if bus.get("stale"):
        out.append("Feature bus 数据陈旧 — 调仓建议仅供参考，请先确认特征刷新")
    if not out:
        out.append("各账户占比在目标带内，无需调仓")
    return out[:6]


def _a_beta_share(allocation: Mapping[str, Any]) -> float:
    return sum(
        float(r.get("nav_pct") or 0.0)
        for r in allocation.get("scopes") or []
        if r.get("layer") == "a" and r.get("nav_pct") is not None
    )


def _contradiction_alert(
    composite: Mapping[str, Any],
    allocation: Mapping[str, Any],
) -> str:
    """Composite vs NAV mismatch (design §4.2), independent of band tolerance."""
    comp = str(composite.get("label") or "neutral")
    alert = "OK"
    a_beta = _a_beta_share(allocation)
    if comp == "risk_on":
        if a_beta < 0.10:
            alert = _max_alert(alert, "REBALANCE_SUGGEST")
        elif a_beta < 0.15:
            alert = _max_alert(alert, "WATCH")
    if comp == "risk_off":
        b_row = next(
            (r for r in allocation.get("scopes") or [] if r.get("scope") == "trend"),
            None,
        )
        b_pct = b_row.get("nav_pct") if b_row else None
        if b_pct is not None:
            b_f = float(b_pct)
            if b_f > 0.55:
                alert = _max_alert(alert, "REBALANCE_SUGGEST")
            elif b_f > 0.50:
                alert = _max_alert(alert, "WATCH")
        roll = next(
            (r for r in allocation.get("scopes") or [] if r.get("scope") == "rolling"),
            None,
        )
        if roll and roll.get("configured") and roll.get("nav_pct") is not None:
            r_f = float(roll["nav_pct"])
            if r_f > 0.15:
                alert = _max_alert(alert, "REBALANCE_SUGGEST")
            elif r_f > 0.10:
                alert = _max_alert(alert, "WATCH")
    return alert


def build_regime_cockpit(
    *,
    strategies_root: Path,
    project_root: Path,
    feature_bus_root: Path,
    symbol: str = "BTCUSDT",
    window_days: int = 7,
) -> Dict[str, Any]:
    config = load_rebalance_config(project_root)
    stale_minutes = int(config.get("feature_bus_stale_minutes") or 240)
    live = build_live_layers(
        strategies_root=strategies_root,
        project_root=project_root,
        feature_bus_root=feature_bus_root,
        symbol=symbol,
        window_days=window_days,
        stale_minutes=stale_minutes,
    )
    ctx = live.get("composite_context") or {}
    composite = compute_composite(ctx, config)
    ledger = build_exchange_ledger()
    allocation = build_allocation(
        ledger=ledger,
        composite_label=str(composite.get("label") or "neutral"),
        config=config,
    )
    layers = live.get("layers") or {}
    suggestions = build_suggestions(
        allocation=allocation,
        composite=composite,
        layers=live,
    )
    ops = fetch_regime_ops_snapshot(strategies_root, project_root=project_root)
    alert = str(allocation.get("alert") or "OK")
    alert = _max_alert(alert, _contradiction_alert(composite, allocation))
    if live.get("feature_bus", {}).get("stale") and alert == "OK":
        alert = "WATCH"

    return {
        "as_of": live.get("feature_bus", {}).get("as_of"),
        "symbol": live.get("symbol"),
        "feature_bus": live.get("feature_bus"),
        "layers": layers,
        "composite": composite,
        "allocation": {
            **allocation,
            "composite": composite.get("label"),
            "suggestions": suggestions,
            "alert": alert,
        },
        "ops": ops,
    }
