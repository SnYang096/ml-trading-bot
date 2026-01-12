from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Protocol

import yaml


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(x))))


def _entropy(p: Dict[str, float]) -> float:
    # Natural entropy (0..log(K)).
    h = 0.0
    for _, v in (p or {}).items():
        x = float(v)
        if x > 0:
            h -= x * math.log(x)
    return float(h)


@dataclass(frozen=True)
class RouterAggregateSignals:
    p_trend: float
    p_mean: float
    p_notrade: float
    confidence: float
    regime_entropy: float
    crowding_score: float = 0.0

    # For TREND ZERO LAW cross-symbol consistency checks
    key_symbol_trend_flags: Optional[Dict[str, bool]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "p_trend": float(self.p_trend),
            "p_mean": float(self.p_mean),
            "p_notrade": float(self.p_notrade),
            "confidence": float(self.confidence),
            "regime_entropy": float(self.regime_entropy),
            "crowding_score": float(self.crowding_score),
            "key_symbol_trend_flags": dict(self.key_symbol_trend_flags or {}),
        }


class SymbolModeLike(Protocol):
    symbol: str
    mode: str


def _get_symbol_mode(d: Any) -> Tuple[str, str]:
    """
    Extract (symbol, mode) from either:
    - attribute-style objects: d.symbol / d.mode
    - dict-like objects: d["symbol"] / d["mode"]
    """
    try:
        return (str(getattr(d, "symbol")), str(getattr(d, "mode")))
    except Exception:
        pass
    try:
        return (str(d.get("symbol")), str(d.get("mode")))
    except Exception:
        return ("", "NO_TRADE")


def aggregate_from_symbol_modes(
    *,
    decisions: Iterable[SymbolModeLike | Dict[str, Any] | Any],
    key_symbols: Optional[List[str]] = None,
) -> RouterAggregateSignals:
    """
    V1 aggregation (low-DOF):
    - p_trend/p_mean/p_notrade are computed from current symbol modes (equal-weight).
    - confidence is derived from entropy (lower entropy => higher confidence).
    - regime_entropy is normalized entropy in [0,1] by dividing log(3).
    - crowding_score is not available yet => 0.0 (conservative overlays will remain off).
    """
    counts = {"TREND": 0, "MEAN": 0, "NO_TRADE": 0}
    ds = list(decisions)
    for d in ds:
        _, mode = _get_symbol_mode(d)
        m = str(mode).upper()
        if m in {"TREND", "MEAN", "NO_TRADE"}:
            counts[m] += 1
        else:
            counts["NO_TRADE"] += 1
    n = max(1, int(len(ds)))
    p = {
        "TREND": float(counts["TREND"]) / float(n),
        "MEAN": float(counts["MEAN"]) / float(n),
        "NO_TRADE": float(counts["NO_TRADE"]) / float(n),
    }
    ent = _entropy(p)
    ent_norm = float(ent / max(1e-12, math.log(3.0)))
    conf = float(_clamp(1.0 - ent_norm, 0.0, 1.0))

    key_flags = None
    if key_symbols:
        key_flags = {}
        for sym in key_symbols:
            # if missing, treat as not trend (conservative)
            flag = False
            for d in ds:
                s2, m2 = _get_symbol_mode(d)
                if str(s2) == str(sym):
                    flag = str(m2).upper() == "TREND"
                    break
            key_flags[str(sym)] = bool(flag)

    return RouterAggregateSignals(
        p_trend=float(p["TREND"]),
        p_mean=float(p["MEAN"]),
        p_notrade=float(p["NO_TRADE"]),
        confidence=conf,
        regime_entropy=float(ent_norm),
        crowding_score=0.0,
        key_symbol_trend_flags=key_flags,
    )


@dataclass(frozen=True)
class PortfolioAssetsConfigV1:
    name: str
    assets: Dict[str, Dict[str, Any]]
    router_to_weights: Dict[str, Dict[str, Any]]
    trend_zero_law: Dict[str, Any]


def load_portfolio_assets_config(path: str | Path) -> PortfolioAssetsConfigV1:
    p = Path(path)
    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return PortfolioAssetsConfigV1(
        name=str(obj.get("name", "portfolio_assets_v1")),
        assets=dict(obj.get("assets") or {}),
        router_to_weights=dict(obj.get("router_to_weights") or {}),
        trend_zero_law=dict(obj.get("trend_zero_law") or {}),
    )


def _trend_zero_law_triggered(
    *,
    cfg: PortfolioAssetsConfigV1,
    sig: RouterAggregateSignals,
    gate_veto: bool,
    portfolio_drawdown: float,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    law = cfg.trend_zero_law or {}
    for r in law.get("rules") or []:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name", "rule")).strip() or "rule"
        if "regime_entropy_gt" in r:
            thr = float(r["regime_entropy_gt"])
            if float(sig.regime_entropy) > thr:
                reasons.append(name)
                continue
        if "portfolio_drawdown_gt" in r:
            thr = float(r["portfolio_drawdown_gt"])
            if float(portfolio_drawdown) > thr:
                reasons.append(name)
                continue
        if r.get("gate_veto_required") is True and bool(gate_veto):
            reasons.append(name)
            continue
        if "require_key_symbols" in r and isinstance(r["require_key_symbols"], list):
            keys = [str(x) for x in r["require_key_symbols"]]
            flags = sig.key_symbol_trend_flags or {}
            # Trigger when not all key symbols are TREND (conservative)
            if keys:
                if not all(bool(flags.get(k, False)) for k in keys):
                    reasons.append(name)
                    continue
    return (len(reasons) > 0), reasons


def compute_portfolio_asset_weights_v1(
    *,
    cfg: PortfolioAssetsConfigV1,
    sig: RouterAggregateSignals,
    gate_veto: bool = False,
    portfolio_drawdown: float = 0.0,
) -> Dict[str, float]:
    """
    Deterministic mapping from router signals to portfolio asset weights.
    Returns weights for the 5 portfolio assets; sums to 1 (with cash as residual).
    """

    # 0) Trend zero law
    trend_zero, _ = _trend_zero_law_triggered(
        cfg=cfg, sig=sig, gate_veto=gate_veto, portfolio_drawdown=portfolio_drawdown
    )

    w = {
        "GLOBAL_TREND": 0.0,
        "GLOBAL_MEAN": 0.0,
        "GLOBAL_CASH": 0.0,
        "HIGH_BETA_OVERLAY": 0.0,
        "DEFENSIVE_MEAN": 0.0,
    }

    # 1) GLOBAL_TREND (资格制)
    tr = cfg.router_to_weights.get("global_trend") or {}
    tr_min = float(tr.get("p_trend_min", 0.6))
    ent_max = float(tr.get("regime_entropy_max", 0.4))
    tr_max = float(
        tr.get(
            "max_weight",
            float(cfg.assets.get("GLOBAL_TREND", {}).get("max_weight", 0.4)),
        )
    )
    if (
        (not trend_zero)
        and float(sig.p_trend) >= tr_min
        and float(sig.regime_entropy) <= ent_max
    ):
        base = (float(sig.p_trend) - tr_min) / max(1e-12, 1.0 - tr_min)
        penalty = (
            float(sig.crowding_score) if bool(tr.get("crowding_penalty", True)) else 0.0
        )
        w["GLOBAL_TREND"] = _clamp(base * (1.0 - penalty), 0.0, tr_max)

    # 2) GLOBAL_MEAN (永远在场)
    mn = cfg.router_to_weights.get("global_mean") or {}
    floor = float(mn.get("base_floor", 0.2))
    mn_max = float(
        mn.get(
            "max_weight",
            float(cfg.assets.get("GLOBAL_MEAN", {}).get("max_weight", 0.35)),
        )
    )
    mean_w = float(floor + float(sig.p_mean) * (1.0 - float(sig.p_trend)))
    w["GLOBAL_MEAN"] = _clamp(
        mean_w, float(cfg.assets.get("GLOBAL_MEAN", {}).get("min_weight", 0.2)), mn_max
    )

    # 3) DEFENSIVE_MEAN (高不确定性时出现)
    dm = cfg.router_to_weights.get("defensive_mean") or {}
    ent_min = float(dm.get("regime_entropy_min", 0.5))
    dm_max = float(
        dm.get(
            "max_weight",
            float(cfg.assets.get("DEFENSIVE_MEAN", {}).get("max_weight", 0.25)),
        )
    )
    if float(sig.regime_entropy) >= ent_min:
        w["DEFENSIVE_MEAN"] = _clamp(float(sig.p_mean + sig.p_notrade), 0.0, dm_max)

    # 4) HIGH_BETA_OVERLAY (严格条件)
    hb = cfg.router_to_weights.get("high_beta_overlay") or {}
    hb_tr_min = float(hb.get("p_trend_min", 0.75))
    hb_crowd_max = float(hb.get("crowding_max", 0.3))
    hb_conf_min = float(hb.get("confidence_min", 0.7))
    hb_max = float(
        hb.get(
            "max_weight",
            float(cfg.assets.get("HIGH_BETA_OVERLAY", {}).get("max_weight", 0.1)),
        )
    )
    if (
        (not trend_zero)
        and float(sig.p_trend) > hb_tr_min
        and float(sig.crowding_score) < hb_crowd_max
        and float(sig.confidence) > hb_conf_min
    ):
        w["HIGH_BETA_OVERLAY"] = _clamp(
            (float(sig.p_trend) - hb_tr_min) * 0.4, 0.0, hb_max
        )

    # 5) GLOBAL_CASH as residual with min floor
    cash = cfg.router_to_weights.get("global_cash") or {}
    cash_min = float(
        cash.get(
            "min_weight",
            float(cfg.assets.get("GLOBAL_CASH", {}).get("min_weight", 0.1)),
        )
    )
    used = float(
        w["GLOBAL_TREND"]
        + w["GLOBAL_MEAN"]
        + w["DEFENSIVE_MEAN"]
        + w["HIGH_BETA_OVERLAY"]
    )
    w["GLOBAL_CASH"] = max(cash_min, 1.0 - used)

    # Renormalize if needed (due to cash floor)
    s = float(sum(w.values()))
    if s <= 1e-12:
        return {"GLOBAL_CASH": 1.0}
    return {k: float(v / s) for k, v in w.items()}


def trend_zero_law_status_v1(
    *,
    cfg: PortfolioAssetsConfigV1,
    sig: RouterAggregateSignals,
    gate_veto: bool = False,
    portfolio_drawdown: float = 0.0,
) -> Dict[str, Any]:
    """
    Public helper: check whether TREND ZERO LAW is triggered, and return reasons.
    Useful for reporting/attribution artifacts.
    """
    triggered, reasons = _trend_zero_law_triggered(
        cfg=cfg,
        sig=sig,
        gate_veto=bool(gate_veto),
        portfolio_drawdown=float(portfolio_drawdown),
    )
    return {"triggered": bool(triggered), "reasons": list(reasons)}
