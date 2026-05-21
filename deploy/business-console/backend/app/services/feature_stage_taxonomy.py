"""Archetype YAML scan for console feature taxonomy (no src/ dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml

CONSOLE_STRATEGIES: Tuple[Dict[str, str], ...] = (
    {"id": "tpc", "account_layer": "trend", "title": "TPC"},
    {"id": "bpc", "account_layer": "trend", "title": "BPC"},
    {"id": "me", "account_layer": "trend", "title": "ME"},
    {"id": "srb", "account_layer": "trend", "title": "SRB"},
    {"id": "spot_accum_simple", "account_layer": "spot", "title": "Spot"},
    {"id": "chop_grid", "account_layer": "multi_leg", "title": "Chop Grid"},
    {"id": "trend_scalp", "account_layer": "multi_leg", "title": "Trend Scalp"},
)

STAGE_ORDER: Tuple[str, ...] = (
    "prefilter",
    "direction",
    "gate",
    "entry",
    "evidence",
    "regime",
    "execution",
)

STAGE_LABELS: Dict[str, str] = {
    "prefilter": "Prefilter",
    "direction": "Direction",
    "gate": "Gate",
    "entry": "Entry",
    "evidence": "Evidence",
    "regime": "Regime",
    "execution": "Execution",
}

ACCOUNT_LAYER_LABELS: Dict[str, str] = {
    "trend": "B·Trend",
    "spot": "A·Spot",
    "multi_leg": "C·Multi-leg",
}

_MULTILEG_RUNTIME_ALIASES: Dict[str, List[str]] = {
    "bpc_semantic_chop": ["semantic_chop", "tpc_semantic_chop"],
    "trend_confidence": ["trend_confidence_f"],
}

_WHEN_RESERVED = frozenset(
    {"and", "or", "not", "all_of", "any_of", "min_matches", "min_matches_any"}
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    obj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return obj if isinstance(obj, dict) else {}


def _extract_features_from_when(when: Any) -> Set[str]:
    out: Set[str] = set()
    if isinstance(when, dict):
        for k, v in when.items():
            if k in _WHEN_RESERVED:
                if isinstance(v, list):
                    for item in v:
                        out |= _extract_features_from_when(item)
                elif isinstance(v, dict):
                    out |= _extract_features_from_when(v)
            else:
                out.add(str(k))
    return out


def _extract_features_from_gate(cfg: Dict[str, Any]) -> Set[str]:
    features: Set[str] = set()
    for section in ("hard_gates", "guardrails", "system_safety"):
        rules = cfg.get(section)
        if not isinstance(rules, list):
            continue
        for rule in rules:
            if isinstance(rule, dict) and rule.get("when"):
                features |= _extract_features_from_when(rule["when"])
    return features


def _extract_features_from_evidence(cfg: Dict[str, Any]) -> Set[str]:
    features: Set[str] = set()
    for item in cfg.get("evidence") or []:
        if isinstance(item, dict) and item.get("feature"):
            features.add(str(item["feature"]))
    return features


def _extract_features_from_entry_filters(cfg: Dict[str, Any]) -> Set[str]:
    features: Set[str] = set()
    for f in cfg.get("filters") or []:
        if not isinstance(f, dict) or not f.get("enabled", False):
            continue
        for cond in f.get("conditions") or []:
            if isinstance(cond, dict) and cond.get("feature"):
                features.add(str(cond["feature"]))
    for col in cfg.get("prefetch_columns") or []:
        if col:
            features.add(str(col))
    return features


def _extract_features_from_prefilter(cfg: Dict[str, Any]) -> Set[str]:
    features: Set[str] = set()
    for rule in cfg.get("rules") or []:
        if isinstance(rule, dict) and rule.get("feature"):
            features.add(str(rule["feature"]))
    return features


def _walk_dict_for_features(obj: Any, out: Set[str]) -> None:
    if isinstance(obj, dict):
        feat = obj.get("feature")
        if isinstance(feat, str) and feat.strip():
            out.add(feat.strip())
        raw = obj.get("features")
        if isinstance(raw, list):
            for f in raw:
                if f:
                    out.add(str(f))
        band = obj.get("band_feature")
        if isinstance(band, str) and band.strip():
            out.add(band.strip())
        for v in obj.values():
            _walk_dict_for_features(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _walk_dict_for_features(item, out)


def _extract_features_from_direction(cfg: Dict[str, Any]) -> Set[str]:
    features: Set[str] = set()
    _walk_dict_for_features(cfg.get("direction_rules"), features)
    return features


def _extract_features_from_execution(cfg: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    rf = cfg.get("regime_feature")
    if isinstance(rf, str) and rf.strip():
        out.add(rf.strip())
    return out


def _extract_features_from_rule_block(rule: Any) -> Set[str]:
    out: Set[str] = set()
    if isinstance(rule, dict):
        feat = rule.get("feature")
        if feat:
            out.add(str(feat))
        for key in ("all_of", "any_of", "rules"):
            block = rule.get(key)
            if isinstance(block, list):
                for sub in block:
                    out |= _extract_features_from_rule_block(sub)
    elif isinstance(rule, list):
        for sub in rule:
            out |= _extract_features_from_rule_block(sub)
    return out


def _extract_features_from_prefilter_full(cfg: Dict[str, Any]) -> Set[str]:
    out = _extract_features_from_prefilter(cfg)
    for rule in cfg.get("rules") or []:
        out |= _extract_features_from_rule_block(rule)
    return out


def _extract_features_from_multileg_regime(cfg: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    regime = cfg.get("regime")
    if not isinstance(regime, dict):
        return out
    entry_feature = regime.get("entry_feature")
    if entry_feature:
        col = str(entry_feature)
        out.add(col)
        out.update(_MULTILEG_RUNTIME_ALIASES.get(col, []))
    if (
        regime.get("max_semantic_chop_entry") is not None
        or regime.get("max_semantic_chop_hold") is not None
    ):
        out.add("semantic_chop")
        out.update(_MULTILEG_RUNTIME_ALIASES.get("bpc_semantic_chop", []))
    if not regime.get("exclude_box_prefilter", True):
        out.add("box_prefilter")
    return out


def extract_strategy_stage_columns(archetypes_dir: Path) -> Dict[str, List[str]]:
    d = Path(archetypes_dir)
    stages: Dict[str, Set[str]] = {s: set() for s in STAGE_ORDER}

    pre_path = d / "prefilter.yaml"
    if pre_path.is_file():
        pre = _load_yaml(pre_path)
        stages["prefilter"] |= _extract_features_from_prefilter_full(pre)
        stages["regime"] |= _extract_features_from_multileg_regime(pre)

    gate_path = d / "gate.yaml"
    if gate_path.is_file():
        stages["gate"] |= _extract_features_from_gate(_load_yaml(gate_path))

    dir_path = d / "direction.yaml"
    if dir_path.is_file():
        stages["direction"] |= _extract_features_from_direction(_load_yaml(dir_path))

    ef_path = d / "entry_filters.yaml"
    if ef_path.is_file():
        stages["entry"] |= _extract_features_from_entry_filters(_load_yaml(ef_path))

    ev_path = d / "evidence.yaml"
    if ev_path.is_file():
        stages["evidence"] |= _extract_features_from_evidence(_load_yaml(ev_path))

    ex_path = d / "execution.yaml"
    if ex_path.is_file():
        stages["execution"] |= _extract_features_from_execution(_load_yaml(ex_path))

    return {k: sorted(v) for k, v in stages.items() if v}


def build_console_feature_taxonomy(strategies_root: str | Path) -> Dict[str, Any]:
    root = Path(strategies_root)
    strategies_out: List[Dict[str, Any]] = []
    index: Dict[str, List[Dict[str, str]]] = {}

    for meta in CONSOLE_STRATEGIES:
        sid = meta["id"]
        arch = root / sid / "archetypes"
        if not arch.is_dir():
            continue
        stage_cols = extract_strategy_stage_columns(arch)
        if not any(stage_cols.values()):
            continue
        strategies_out.append(
            {
                "id": sid,
                "account_layer": meta["account_layer"],
                "account_layer_title": ACCOUNT_LAYER_LABELS.get(
                    meta["account_layer"], meta["account_layer"]
                ),
                "title": meta["title"],
                "stages": stage_cols,
            }
        )
        for stage, cols in stage_cols.items():
            stage_title = STAGE_LABELS.get(stage, stage)
            for col in cols:
                rec = {
                    "column": col,
                    "strategy": sid,
                    "strategy_title": meta["title"],
                    "account_layer": meta["account_layer"],
                    "account_layer_title": ACCOUNT_LAYER_LABELS.get(
                        meta["account_layer"], meta["account_layer"]
                    ),
                    "stage": stage,
                    "stage_title": stage_title,
                }
                index.setdefault(col, []).append(rec)

    return {
        "strategies": strategies_out,
        "index": index,
        "stage_order": list(STAGE_ORDER),
        "stage_labels": dict(STAGE_LABELS),
        "account_layer_labels": dict(ACCOUNT_LAYER_LABELS),
    }
