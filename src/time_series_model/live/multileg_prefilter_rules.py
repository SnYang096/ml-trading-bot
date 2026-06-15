"""Multileg prefilter rule evaluation (live + backtest shared)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

CHOP_GRID_PREFILTER_ALIASES: dict[str, str] = {
    "atr": "atr14",
    "bpc_semantic_chop": "semantic_chop",
    "bpc_semantic_chop_ts_q": "semantic_chop_ts_q",
}

_OP_ALIASES = {
    ">": ">",
    ">=": ">=",
    "gte": ">=",
    "ge": ">=",
    "<": "<",
    "<=": "<=",
    "lte": "<=",
    "le": "<=",
    "==": "==",
    "=": "==",
    "eq": "==",
    "!=": "!=",
    "ne": "!=",
}


def _to_numeric_series(df: pd.DataFrame, feature: str) -> pd.Series:
    if feature not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[feature], errors="coerce")


def _scalar_bool(df: pd.DataFrame, value: bool) -> pd.Series:
    return pd.Series([bool(value)] * len(df), index=df.index, dtype=bool)


def _eval_simple_rule(df: pd.DataFrame, rule: Mapping[str, Any]) -> pd.Series:
    feat = str(rule.get("feature", "") or "").strip()
    if not feat:
        return _scalar_bool(df, True)
    op_raw = str(rule.get("operator", "") or "").strip().lower()
    op = _OP_ALIASES.get(op_raw, op_raw)
    value = rule.get("value")
    if value is None:
        return _scalar_bool(df, False)
    s = _to_numeric_series(df, feat)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _scalar_bool(df, False)
    if op == ">":
        return (s > v).fillna(False)
    if op == ">=":
        return (s >= v).fillna(False)
    if op == "<":
        return (s < v).fillna(False)
    if op == "<=":
        return (s <= v).fillna(False)
    if op == "==":
        return (s == v).fillna(False)
    if op == "!=":
        return (s != v).fillna(False)
    return _scalar_bool(df, False)


def eval_prefilter_rule(df: pd.DataFrame, rule: Mapping[str, Any]) -> pd.Series:
    any_of = rule.get("any_of")
    if isinstance(any_of, list) and any_of:
        mask = _scalar_bool(df, False)
        for sub in any_of:
            if isinstance(sub, Mapping):
                mask = mask | eval_prefilter_rule(df, sub)
        return mask.fillna(False)

    all_of = rule.get("all_of")
    if isinstance(all_of, list) and all_of:
        mask = _scalar_bool(df, True)
        for sub in all_of:
            if isinstance(sub, Mapping):
                mask = mask & eval_prefilter_rule(df, sub)
        return mask.fillna(False)

    return _eval_simple_rule(df, rule)


def apply_prefilter_rules(
    df: pd.DataFrame,
    rules: list[Mapping[str, Any]] | None,
    *,
    feature_aliases: Mapping[str, str] | None = None,
) -> pd.Series:
    if df.empty:
        return pd.Series([], index=df.index, dtype=bool)
    raw_rules = [r for r in (rules or []) if isinstance(r, Mapping)]
    if not raw_rules:
        return _scalar_bool(df, True)

    aliases = dict(feature_aliases or {})
    use_df = df.copy()
    for target, src in aliases.items():
        if target in use_df.columns or src not in use_df.columns:
            continue
        use_df[target] = use_df[src]

    mask = _scalar_bool(use_df, True)
    for rule in raw_rules:
        mask = mask & eval_prefilter_rule(use_df, rule)
    return mask.fillna(False)


def features_pass_prefilter_rules(
    features: Mapping[str, Any] | None,
    rules: Sequence[Mapping[str, Any]] | None,
    *,
    feature_aliases: Mapping[str, str] | None = None,
) -> bool:
    """Single-bar prefilter pass (fail closed on missing numeric features)."""
    raw_rules = [r for r in (rules or []) if isinstance(r, Mapping)]
    if not raw_rules:
        return True
    row = dict(features or {})
    aliases = dict(feature_aliases or {})
    for target, src in aliases.items():
        if target not in row and src in row:
            row[target] = row[src]
    if not row:
        return False
    return bool(apply_prefilter_rules(pd.DataFrame([row]), raw_rules).iloc[0])
