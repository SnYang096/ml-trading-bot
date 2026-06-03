"""Prefilter / Gate time regions on Trade Map (archetype YAML + feature bus)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml

from mlbot_console.services.chop_grid_overlay import _parse_ts
from mlbot_console.services.feature_overlay import _resolve_feature_path
from time_series_model.archetype.loader import (
    PrefilterConfig,
    _evaluate_when_clause,
    load_strategy_archetype,
)
from time_series_model.live.feature_stage_taxonomy import (
    _MULTILEG_RUNTIME_ALIASES,
    extract_strategy_stage_columns,
)

# Trend strategies shown when B·Trend layer is on.
TREND_STAGE_STRATEGIES: Tuple[str, ...] = ("bpc", "tpc", "me", "srb")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw if isinstance(raw, dict) else {}


def _resolve_feature_value(row: pd.Series, feat: str) -> Optional[float]:
    """Read feature from parquet row with multileg aliases."""
    col = str(feat)
    candidates = [col, f"{col}_f"]
    candidates.extend(_MULTILEG_RUNTIME_ALIASES.get(col, []))
    for name in candidates:
        if name not in row.index:
            continue
        val = row[name]
        if val is None or (isinstance(val, float) and val != val):
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def _row_features(row: pd.Series, columns: Set[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in columns:
        val = _resolve_feature_value(row, col)
        if val is not None:
            out[col] = val
    return out


def _evaluate_prefilter_rules(rules: List[Dict[str, Any]], features: Dict[str, Any]) -> bool:
    if not rules:
        return True
    for rule in rules:
        if isinstance(rule, dict) and "all_of" in rule:
            subs = rule.get("all_of") or []
            if not subs:
                continue
            when = {"all_of": []}
            for sub in subs:
                if not isinstance(sub, dict) or "feature" not in sub:
                    continue
                feat = str(sub["feature"])
                op = str(sub.get("operator") or ">=").strip()
                try:
                    thr = float(sub["value"])
                except (TypeError, ValueError):
                    return False
                cond: Dict[str, Any] = {}
                if op in (">=", "ge"):
                    cond = {feat: {"value_gte": thr}}
                elif op in (">", "gt"):
                    cond = {feat: {"value_gt": thr}}
                elif op in ("<=", "le"):
                    cond = {feat: {"value_lte": thr}}
                elif op in ("<", "lt"):
                    cond = {feat: {"value_lt": thr}}
                else:
                    cond = {feat: {"value_gte": thr}}
                when["all_of"].append(cond)
            if not when["all_of"]:
                return False
            try:
                if not _evaluate_when_clause(when, features, None):
                    return False
            except ValueError:
                return False
            continue
        if not PrefilterConfig._check_single(rule, features):
            return False
    return True


def _bool_series_to_regions(
    times: List[int],
    active: List[bool],
    *,
    label: str,
    strategy: str,
    stage: str,
) -> List[Dict[str, Any]]:
    regions: List[Dict[str, Any]] = []
    in_region = False
    start: Optional[int] = None
    prev_ts: Optional[int] = None
    for ts, on in zip(times, active):
        if on and not in_region:
            in_region = True
            start = ts
        elif not on and in_region and start is not None:
            regions.append(
                {
                    "start": start,
                    "end": prev_ts if prev_ts is not None else ts,
                    "label": label,
                    "strategy": strategy,
                    "stage": stage,
                }
            )
            in_region = False
            start = None
        prev_ts = ts
    if in_region and start is not None and prev_ts is not None:
        regions.append(
            {
                "start": start,
                "end": prev_ts,
                "label": label,
                "strategy": strategy,
                "stage": stage,
            }
        )
    return regions


def _hysteresis_active(
    values: List[Optional[float]],
    *,
    entry_min: float,
    exit_below: float,
) -> List[bool]:
    active = False
    out: List[bool] = []
    for val in values:
        if val is None or val != val:
            out.append(active)
            continue
        if not active:
            active = val >= entry_min
        else:
            if val < exit_below:
                active = False
        out.append(active)
    return out


def _load_feature_frame(
    feature_bus_root: Path,
    symbol: str,
    timeframe: str,
    columns: Set[str],
    *,
    start: Optional[Any] = None,
    end: Optional[Any] = None,
) -> pd.DataFrame:
    path = _resolve_feature_path(feature_bus_root, symbol, timeframe)
    if path is None:
        return pd.DataFrame()
    read_cols = ["timestamp"]
    try:
        import pyarrow.parquet as pq

        names = set(pq.read_schema(path).names)
        for c in columns:
            if c in names:
                read_cols.append(c)
            if f"{c}_f" in names:
                read_cols.append(f"{c}_f")
        for aliases in _MULTILEG_RUNTIME_ALIASES.values():
            for a in aliases:
                if a in names and a not in read_cols:
                    read_cols.append(a)
    except Exception:
        read_cols = list({"timestamp", *columns})
    try:
        df = pd.read_parquet(path, columns=list(dict.fromkeys(read_cols)))
    except Exception:
        df = pd.read_parquet(path)
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
    df = df.sort_values("timestamp")
    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def load_trend_prefilter_regions(
    feature_bus_root: Path,
    symbol: str,
    timeframe: str,
    strategy: str,
    strategies_root: Path,
    *,
    start: Optional[Any] = None,
    end: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    arch = strategies_root / strategy / "archetypes"
    pre = PrefilterConfig.from_yaml(arch / "prefilter.yaml")
    if not pre.rules:
        return []
    columns = set(extract_strategy_stage_columns(arch).get("prefilter", []))
    df = _load_feature_frame(
        feature_bus_root, symbol, timeframe, columns, start=start, end=end
    )
    if df.empty:
        return []
    times: List[int] = []
    flags: List[bool] = []
    for _, row in df.iterrows():
        ts = _parse_ts(row["timestamp"])
        if ts is None:
            continue
        features = _row_features(row, columns)
        passed, _ = pre.evaluate(features)
        times.append(ts)
        flags.append(bool(passed))
    return _bool_series_to_regions(
        times,
        flags,
        label=f"{strategy} prefilter",
        strategy=strategy,
        stage="prefilter",
    )


def load_trend_gate_regions(
    feature_bus_root: Path,
    symbol: str,
    timeframe: str,
    strategy: str,
    strategies_root: Path,
    *,
    start: Optional[Any] = None,
    end: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    arch = strategies_root / strategy / "archetypes"
    gate_path = arch / "gate.yaml"
    if not gate_path.is_file():
        return []
    try:
        archetype = load_strategy_archetype(
            strategy, strategies_root=strategies_root, live_layout=True
        )
    except FileNotFoundError:
        return []
    columns: Set[str] = set()
    for stage in ("gate", "prefilter", "regime"):
        columns |= set(extract_strategy_stage_columns(arch).get(stage, []))
    for rule in archetype.gate.all_rules:
        for key in rule.when:
            if key not in ("all_of", "any_of", "min_matches"):
                columns.add(str(key))
    df = _load_feature_frame(
        feature_bus_root, symbol, timeframe, columns, start=start, end=end
    )
    if df.empty:
        return []
    times: List[int] = []
    flags: List[bool] = []
    for _, row in df.iterrows():
        ts = _parse_ts(row["timestamp"])
        if ts is None:
            continue
        features = _row_features(row, columns)
        try:
            passed, _, _ = archetype.apply_gate(features)
        except ValueError:
            passed = False
        times.append(ts)
        flags.append(bool(passed))
    return _bool_series_to_regions(
        times,
        flags,
        label=f"{strategy} gate",
        strategy=strategy,
        stage="gate",
    )


def load_chop_grid_prefilter_regions(
    feature_bus_root: Path,
    symbol: str,
    timeframe: str,
    strategies_root: Path,
    *,
    start: Optional[Any] = None,
    end: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Chop grid tradable window: regime chop hysteresis + prefilter rules (+ not box)."""
    pre_path = strategies_root / "chop_grid" / "archetypes" / "prefilter.yaml"
    raw = _load_yaml(pre_path)
    regime_cfg = raw.get("regime") if isinstance(raw.get("regime"), dict) else {}
    rules = list(raw.get("rules") or [])
    arch = strategies_root / "chop_grid" / "archetypes"
    columns: Set[str] = set()
    columns |= set(extract_strategy_stage_columns(arch).get("prefilter", []))
    columns |= set(extract_strategy_stage_columns(arch).get("regime", []))
    entry_feat = str(regime_cfg.get("entry_feature") or "bpc_semantic_chop")
    columns.add(entry_feat)
    columns.update(_MULTILEG_RUNTIME_ALIASES.get(entry_feat, []))
    columns.add("box_prefilter")

    df = _load_feature_frame(
        feature_bus_root, symbol, timeframe, columns, start=start, end=end
    )
    if df.empty:
        return []

    entry_min = float(regime_cfg.get("entry_min", regime_cfg.get("entry_chop_min", 0.50)))
    exit_below = float(regime_cfg.get("exit_below", regime_cfg.get("exit_chop_below", 0.32)))
    chop_vals: List[Optional[float]] = []
    times: List[int] = []
    feature_rows: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        ts = _parse_ts(row["timestamp"])
        if ts is None:
            continue
        times.append(ts)
        chop_vals.append(_resolve_feature_value(row, entry_feat))
        feature_rows.append(_row_features(row, columns))

    chop_on = _hysteresis_active(chop_vals, entry_min=entry_min, exit_below=exit_below)
    flags: List[bool] = []
    for i, features in enumerate(feature_rows):
        rules_ok = _evaluate_prefilter_rules(rules, features)
        is_box = bool(features.get("box_prefilter", False))
        chop_active = chop_on[i] if i < len(chop_on) else False
        flags.append(chop_active and rules_ok and not is_box)

    return _bool_series_to_regions(
        times,
        flags,
        label="chop_grid prefilter",
        strategy="chop_grid",
        stage="prefilter",
    )


def _chop_grid_regime_params(
    strategies_root: Path,
) -> Tuple[str, float, float]:
    from src.config.regime_layer import multileg_regime_section

    reg_path = strategies_root / "chop_grid" / "archetypes" / "regime.yaml"
    pre_path = strategies_root / "chop_grid" / "archetypes" / "prefilter.yaml"
    raw = _load_yaml(reg_path) if reg_path.is_file() else _load_yaml(pre_path)
    regime_cfg = multileg_regime_section(raw)
    entry_feat = str(regime_cfg.get("entry_feature") or "bpc_semantic_chop")
    entry_min = float(
        regime_cfg.get("entry_min", regime_cfg.get("entry_chop_min", 0.50))
    )
    exit_below = float(
        regime_cfg.get("exit_below", regime_cfg.get("exit_chop_below", 0.32))
    )
    return entry_feat, entry_min, exit_below


def load_chop_grid_regime_exit_markers(
    feature_bus_root: Path,
    symbol: str,
    timeframe: str,
    strategies_root: Path,
    *,
    start: Optional[Any] = None,
    end: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Synthetic exit markers when chop hysteresis turns off (chop < exit_chop_below).

    Matches live ``ChopGridLiveEngine`` regime flatten (``regime_or_risk_exit``), not
    per-leg TP fills. Shown when feature bus is available even if DB lacks market_exit rows.
    """
    entry_feat, entry_min, exit_below = _chop_grid_regime_params(strategies_root)
    columns: Set[str] = {entry_feat}
    columns.update(_MULTILEG_RUNTIME_ALIASES.get(entry_feat, []))
    df = _load_feature_frame(
        feature_bus_root, symbol, timeframe, columns, start=start, end=end
    )
    if df.empty:
        return []

    times: List[int] = []
    chop_vals: List[Optional[float]] = []
    for _, row in df.iterrows():
        ts = _parse_ts(row["timestamp"])
        if ts is None:
            continue
        times.append(ts)
        chop_vals.append(_resolve_feature_value(row, entry_feat))

    if not times:
        return []

    chop_on = _hysteresis_active(
        chop_vals, entry_min=entry_min, exit_below=exit_below
    )
    sym = symbol.upper()
    markers: List[Dict[str, Any]] = []
    for i in range(1, len(times)):
        if not (chop_on[i - 1] and not chop_on[i]):
            continue
        val = chop_vals[i]
        if val is None or val != val or val >= exit_below:
            continue
        t = times[i]
        markers.append(
            {
                "id": f"multi_leg:regime_exit:{sym}:{t}",
                "time": t,
                "symbol": sym,
                "scope": "multi_leg",
                "strategy": "chop_grid",
                "event": "exit",
                "side": "long",
                "price": None,
                "qty": None,
                "status": "filled",
                "color": "#ff7043",
                "detail": {
                    "exit_kind": "regime_or_risk_exit",
                    "exit_reason": "regime_or_risk_exit",
                    "chop": val,
                    "entry_min": entry_min,
                    "exit_below": exit_below,
                    "source": "feature_bus_hysteresis",
                },
            }
        )
    return markers


def load_bundle_stage_regions(
    feature_bus_root: Path,
    strategies_root: Path,
    symbol: str,
    timeframe: str,
    *,
    scopes: List[str],
    include_prefilter: bool = True,
    include_gate: bool = True,
    start: Optional[Any] = None,
    end: Optional[Any] = None,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Regions keyed by strategy -> stage -> spans (unix seconds)."""
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    scope_set = {s.strip().lower() for s in scopes}

    if include_prefilter and "multi_leg" in scope_set:
        spans = load_chop_grid_prefilter_regions(
            feature_bus_root,
            symbol,
            timeframe,
            strategies_root,
            start=start,
            end=end,
        )
        if spans:
            out.setdefault("chop_grid", {})["prefilter"] = spans

    if "trend" in scope_set:
        for strat in TREND_STAGE_STRATEGIES:
            if not (strategies_root / strat / "archetypes").is_dir():
                continue
            strat_out = out.setdefault(strat, {})
            if include_prefilter:
                pre = load_trend_prefilter_regions(
                    feature_bus_root,
                    symbol,
                    timeframe,
                    strat,
                    strategies_root,
                    start=start,
                    end=end,
                )
                if pre:
                    strat_out["prefilter"] = pre
            if include_gate:
                gate = load_trend_gate_regions(
                    feature_bus_root,
                    symbol,
                    timeframe,
                    strat,
                    strategies_root,
                    start=start,
                    end=end,
                )
                if gate:
                    strat_out["gate"] = gate
    return out