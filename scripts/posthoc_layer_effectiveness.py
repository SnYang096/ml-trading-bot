#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass
class EffectStat:
    strategy: str
    layer: str
    scope: str
    name: str
    expected_direction: str
    n_total: int
    n_true: int
    n_false: int
    true_rate: float
    success_true: float
    success_false: float
    effect: float
    z_score: float
    p_value: float


def _find_latest_predictions(results_root: Path, strategy: str) -> Path:
    base = results_root / strategy / "validate_static.constrained" / strategy
    if not base.exists():
        raise FileNotFoundError(f"missing strategy result base: {base}")
    candidates = sorted(base.glob("*/results/predictions.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no predictions.parquet under: {base}")
    return candidates[-1]


def _resolve_predictions_path(
    results_root: Path,
    strategy: str,
    override: Path | None,
) -> Path:
    if override is not None:
        path = override.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"predictions not found: {path}")
        return path
    return _find_latest_predictions(results_root, strategy)


def _to_numeric_series(df: pd.DataFrame, col: str, size: int) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index if len(df.index) == size else None)
    return pd.to_numeric(df[col], errors="coerce")


def _cmp(series: pd.Series, operator: str, value: Any) -> pd.Series:
    op = operator.strip()
    if op == ">":
        return series > value
    if op == ">=":
        return series >= value
    if op == "<":
        return series < value
    if op == "<=":
        return series <= value
    if op == "==":
        return series == value
    if op == "!=":
        return series != value
    raise ValueError(f"unsupported operator: {operator}")


def _eval_prefilter_predicate(pred: Dict[str, Any], df: pd.DataFrame) -> pd.Series:
    if "any_of" in pred:
        parts = [_eval_prefilter_predicate(p, df) for p in pred.get("any_of", [])]
        if not parts:
            return pd.Series(False, index=df.index)
        out = parts[0].copy()
        for part in parts[1:]:
            out = out | part
        return out
    if "all_of" in pred:
        parts = [_eval_prefilter_predicate(p, df) for p in pred.get("all_of", [])]
        if not parts:
            return pd.Series(True, index=df.index)
        out = parts[0].copy()
        for part in parts[1:]:
            out = out & part
        return out
    feature = str(pred.get("feature", "")).strip()
    operator = str(pred.get("operator", "")).strip()
    value = pred.get("value")
    if not feature or not operator:
        return pd.Series(False, index=df.index)
    s = _to_numeric_series(df, feature, len(df))
    return _cmp(s, operator, value).fillna(False)


def _eval_gate_when(when_cfg: Dict[str, Any], df: pd.DataFrame) -> pd.Series:
    if not isinstance(when_cfg, dict):
        return pd.Series(False, index=df.index)
    if "all_of" in when_cfg:
        parts = [_eval_gate_when(item, df) for item in when_cfg.get("all_of", [])]
        if not parts:
            return pd.Series(True, index=df.index)
        out = parts[0].copy()
        for part in parts[1:]:
            out = out & part
        return out
    if "any_of" in when_cfg:
        parts = [_eval_gate_when(item, df) for item in when_cfg.get("any_of", [])]
        if not parts:
            return pd.Series(False, index=df.index)
        out = parts[0].copy()
        for part in parts[1:]:
            out = out | part
        return out

    checks: List[pd.Series] = []
    for feature, cond in when_cfg.items():
        if not isinstance(cond, dict):
            continue
        s = _to_numeric_series(df, str(feature), len(df))
        local = pd.Series(True, index=df.index)
        for key, value in cond.items():
            if key == "value_gt":
                local &= s > value
            elif key == "value_gte":
                local &= s >= value
            elif key == "value_lt":
                local &= s < value
            elif key == "value_lte":
                local &= s <= value
            elif key == "value_eq":
                local &= s == value
            elif key == "value_ne":
                local &= s != value
        checks.append(local.fillna(False))
    if not checks:
        return pd.Series(False, index=df.index)
    out = checks[0].copy()
    for part in checks[1:]:
        out &= part
    return out


def _eval_entry_filter(filter_cfg: Dict[str, Any], df: pd.DataFrame) -> pd.Series:
    conditions = filter_cfg.get("conditions", []) or []
    if not conditions:
        return pd.Series(False, index=df.index)
    out = pd.Series(True, index=df.index)
    for cond in conditions:
        out &= _eval_prefilter_predicate(cond, df)
    return out.fillna(False)


def _safe_rate(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num / den)


def _ztest(
    success_true: int, n_true: int, success_false: int, n_false: int
) -> Tuple[float, float]:
    if n_true <= 0 or n_false <= 0:
        return float("nan"), float("nan")
    p1 = success_true / n_true
    p0 = success_false / n_false
    pooled = (success_true + success_false) / (n_true + n_false)
    se = math.sqrt(max(pooled * (1.0 - pooled) * (1.0 / n_true + 1.0 / n_false), 0.0))
    if se <= 0:
        return float("nan"), float("nan")
    z = (p1 - p0) / se
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return float(z), float(p)


def _build_stat(
    strategy: str,
    layer: str,
    scope: str,
    name: str,
    expected_direction: str,
    flag: pd.Series,
    success: pd.Series,
) -> EffectStat:
    f = flag.fillna(False).astype(bool)
    y = success.fillna(0).astype(int)
    n = int(len(y))
    n_true = int(f.sum())
    n_false = int(n - n_true)
    s_true = int(y[f].sum())
    s_false = int(y[~f].sum())
    r_true = _safe_rate(n_true, n)
    p_true = _safe_rate(s_true, n_true)
    p_false = _safe_rate(s_false, n_false)
    # effect > 0 means "aligned with expected direction"
    diff = p_true - p_false
    if expected_direction == "deny":
        diff = -diff
    z, pval = _ztest(s_true, n_true, s_false, n_false)
    if expected_direction == "deny" and not math.isnan(z):
        z = -z
    return EffectStat(
        strategy=strategy,
        layer=layer,
        scope=scope,
        name=name,
        expected_direction=expected_direction,
        n_total=n,
        n_true=n_true,
        n_false=n_false,
        true_rate=r_true,
        success_true=p_true,
        success_false=p_false,
        effect=diff,
        z_score=z,
        p_value=pval,
    )


def _scopes(df: pd.DataFrame) -> Dict[str, pd.Series]:
    if "ema_1200_position" not in df.columns:
        return {"all": pd.Series(True, index=df.index)}
    ema = pd.to_numeric(df["ema_1200_position"], errors="coerce")
    return {
        "all": pd.Series(True, index=df.index),
        "bull_ema1200": (ema >= 0).fillna(False),
        "bear_ema1200": (ema < 0).fillna(False),
    }


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw or {}


def analyze_strategy(
    results_root: Path,
    config_root: Path,
    strategy: str,
    predictions_path: Path | None = None,
) -> Dict[str, Any]:
    pred_path = _resolve_predictions_path(results_root, strategy, predictions_path)
    df = pd.read_parquet(pred_path)
    if "success_no_rr_extreme" not in df.columns:
        raise KeyError(
            f"{strategy}: missing label success_no_rr_extreme in {pred_path}"
        )
    success = (
        pd.to_numeric(df["success_no_rr_extreme"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    strat_cfg = config_root / strategy / "archetypes"
    regime_cfg = _load_yaml(strat_cfg / "regime.yaml")
    prefilter_cfg = _load_yaml(strat_cfg / "prefilter.yaml")
    gate_cfg = _load_yaml(strat_cfg / "gate.yaml")
    entry_cfg = _load_yaml(strat_cfg / "entry_filters.yaml")

    scopes = _scopes(df)
    stats: List[EffectStat] = []
    missing_features: Dict[str, Dict[str, List[str]]] = {
        "regime": {},
        "prefilter": {},
        "gate": {},
        "entry_filter": {},
    }
    locked_features_missing: Dict[str, Dict[str, List[str]]] = {
        "regime": {},
        "prefilter": {},
        "gate": {},
        "entry_filter": {},
    }

    # Regime — 与 Prefilter 同 rules schema，复用 evaluator
    regime_rules = regime_cfg.get("rules", []) or []
    rg_flags: List[pd.Series] = []
    for i, rule in enumerate(regime_rules):
        name = str(rule.get("id") or rule.get("feature") or f"regime_rule_{i+1}")
        feats = _collect_prefilter_features(rule)
        miss = sorted({f for f in feats if f not in df.columns})
        if miss:
            missing_features["regime"][name] = miss
            if bool(rule.get("locked", False)):
                locked_features_missing["regime"][name] = miss
        flag = _eval_prefilter_predicate(rule, df)
        rg_flags.append(flag)
        for scope_name, scope_mask in scopes.items():
            stats.append(
                _build_stat(
                    strategy,
                    "regime",
                    scope_name,
                    name,
                    "allow",
                    flag[scope_mask],
                    success[scope_mask],
                )
            )
    rg_layer_flag = pd.Series(True, index=df.index)
    for f in rg_flags:
        rg_layer_flag &= f
    for scope_name, scope_mask in scopes.items():
        stats.append(
            _build_stat(
                strategy,
                "regime",
                scope_name,
                "__layer_all_rules__",
                "allow",
                rg_layer_flag[scope_mask],
                success[scope_mask],
            )
        )

    # Prefilter
    pre_rules = prefilter_cfg.get("rules", []) or []
    pre_flags: List[pd.Series] = []
    for i, rule in enumerate(pre_rules):
        name = str(rule.get("id") or rule.get("feature") or f"prefilter_rule_{i+1}")
        feats = _collect_prefilter_features(rule)
        miss = sorted({f for f in feats if f not in df.columns})
        if miss:
            missing_features["prefilter"][name] = miss
            if bool(rule.get("locked", False)):
                locked_features_missing["prefilter"][name] = miss
        flag = _eval_prefilter_predicate(rule, df)
        pre_flags.append(flag)
        for scope_name, scope_mask in scopes.items():
            stats.append(
                _build_stat(
                    strategy,
                    "prefilter",
                    scope_name,
                    name,
                    "allow",
                    flag[scope_mask],
                    success[scope_mask],
                )
            )
    pre_layer_flag = pd.Series(True, index=df.index)
    for f in pre_flags:
        pre_layer_flag &= f
    for scope_name, scope_mask in scopes.items():
        stats.append(
            _build_stat(
                strategy,
                "prefilter",
                scope_name,
                "__layer_all_rules__",
                "allow",
                pre_layer_flag[scope_mask],
                success[scope_mask],
            )
        )

    # Gate
    gate_rules: List[Tuple[str, Dict[str, Any]]] = []
    for section in ("system_safety", "hard_gates", "guardrails"):
        for rule in gate_cfg.get(section, []) or []:
            if bool(rule.get("disabled", False)):
                continue
            if (
                str((rule.get("then") or {}).get("action", "")).strip().lower()
                != "deny"
            ):
                continue
            gate_rules.append((str(rule.get("id") or f"{section}_rule"), rule))
    gate_deny_flags: List[pd.Series] = []
    for name, rule in gate_rules:
        feats = _collect_gate_features(rule.get("when", {}) or {})
        miss = sorted({f for f in feats if f not in df.columns})
        if miss:
            missing_features["gate"][name] = miss
            if bool(rule.get("locked", False)):
                locked_features_missing["gate"][name] = miss
        deny = _eval_gate_when(rule.get("when", {}) or {}, df)
        gate_deny_flags.append(deny)
        for scope_name, scope_mask in scopes.items():
            stats.append(
                _build_stat(
                    strategy,
                    "gate",
                    scope_name,
                    name,
                    "deny",
                    deny[scope_mask],
                    success[scope_mask],
                )
            )
    gate_deny_any = pd.Series(False, index=df.index)
    for d in gate_deny_flags:
        gate_deny_any |= d
    gate_pass = ~gate_deny_any
    for scope_name, scope_mask in scopes.items():
        stats.append(
            _build_stat(
                strategy,
                "gate",
                scope_name,
                "__layer_gate_pass__",
                "allow",
                gate_pass[scope_mask],
                success[scope_mask],
            )
        )

    # Entry filter
    entry_filters = [
        f for f in (entry_cfg.get("filters", []) or []) if bool(f.get("enabled", True))
    ]
    entry_pass_any = pd.Series(False, index=df.index)
    for filt in entry_filters:
        name = str(filt.get("id") or "entry_filter")
        feats = _collect_entry_features(filt)
        miss = sorted({f for f in feats if f not in df.columns})
        if miss:
            missing_features["entry_filter"][name] = miss
            if bool(filt.get("locked", False)):
                locked_features_missing["entry_filter"][name] = miss
        fpass = _eval_entry_filter(filt, df)
        entry_pass_any |= fpass
        for scope_name, scope_mask in scopes.items():
            stats.append(
                _build_stat(
                    strategy,
                    "entry_filter",
                    scope_name,
                    name,
                    "allow",
                    fpass[scope_mask],
                    success[scope_mask],
                )
            )
    for scope_name, scope_mask in scopes.items():
        stats.append(
            _build_stat(
                strategy,
                "entry_filter",
                scope_name,
                "__layer_entry_pass__",
                "allow",
                entry_pass_any[scope_mask],
                success[scope_mask],
            )
        )

    return {
        "strategy": strategy,
        "predictions_path": str(pred_path),
        "n_rows": int(len(df)),
        "missing_features": missing_features,
        "locked_features_missing": locked_features_missing,
        "stats": [asdict(s) for s in stats],
    }


def _collect_prefilter_features(rule: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if not isinstance(rule, dict):
        return out
    if "feature" in rule:
        out.append(str(rule["feature"]))
    for key in ("any_of", "all_of"):
        for sub in rule.get(key, []) or []:
            out.extend(_collect_prefilter_features(sub))
    return out


def _collect_gate_features(when_cfg: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if not isinstance(when_cfg, dict):
        return out
    for key, value in when_cfg.items():
        if key in ("all_of", "any_of"):
            for sub in value or []:
                out.extend(_collect_gate_features(sub))
        elif isinstance(value, dict):
            out.append(str(key))
    return out


def _collect_entry_features(filter_cfg: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for cond in filter_cfg.get("conditions", []) or []:
        if isinstance(cond, dict) and "feature" in cond:
            out.append(str(cond["feature"]))
    return out


def _fmt_pct(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    return f"{v*100:.2f}%"


def _top_lines(
    stats: Iterable[Dict[str, Any]], scope: str, layer: str, n: int = 3
) -> List[str]:
    rows = [
        r
        for r in stats
        if r["scope"] == scope
        and r["layer"] == layer
        and not r["name"].startswith("__")
    ]
    rows = [r for r in rows if r["n_true"] >= 200]
    rows.sort(
        key=lambda r: (
            r["effect"],
            -r["p_value"] if not math.isnan(r["p_value"]) else 1.0,
        )
    )
    bad = rows[:n]
    out: List[str] = []
    for r in bad:
        out.append(
            f"- {r['name']}: effect={_fmt_pct(r['effect'])}, p={r['p_value']:.4g}, "
            f"hit={_fmt_pct(r['true_rate'])}, succ_true={_fmt_pct(r['success_true'])}, succ_false={_fmt_pct(r['success_false'])}"
        )
    return out


def _write_markdown(report: List[Dict[str, Any]], out_md: Path) -> None:
    lines: List[str] = []
    lines.append("# Post-hoc Layer Effectiveness Report")
    lines.append("")
    lines.append(
        "Method: evaluate prefilter/gate/entry independently on the same base predictions set, "
        "then compare success label (`success_no_rr_extreme`) under EMA1200 bull/bear splits."
    )
    lines.append("")
    for strat in report:
        stats = strat["stats"]
        lines.append(f"## {strat['strategy']}")
        lines.append(f"- predictions: `{strat['predictions_path']}`")
        lines.append(f"- rows: `{strat['n_rows']}`")
        for layer in ("regime", "prefilter", "gate", "entry_filter"):
            missing = strat.get("missing_features", {}).get(layer, {})
            if missing:
                lines.append(f"- missing {layer} features:")
                for name, feats in missing.items():
                    lines.append(f"  - {name}: {', '.join(feats)}")
        for layer in ("regime", "prefilter", "gate", "entry_filter"):
            locked_missing = strat.get("locked_features_missing", {}).get(layer, {})
            if locked_missing:
                lines.append(f"- **locked** {layer} rules with missing features:")
                for name, feats in locked_missing.items():
                    lines.append(f"  - **{name}**: {', '.join(feats)}")
        for scope in ("all", "bull_ema1200", "bear_ema1200"):
            lines.append(f"- {scope}:")
            for layer, layer_key in (
                ("regime", "__layer_all_rules__"),
                ("prefilter", "__layer_all_rules__"),
                ("gate", "__layer_gate_pass__"),
                ("entry_filter", "__layer_entry_pass__"),
            ):
                row = next(
                    r
                    for r in stats
                    if r["scope"] == scope
                    and r["layer"] == layer
                    and r["name"] == layer_key
                )
                lines.append(
                    f"  - {layer}: effect={_fmt_pct(row['effect'])}, p={row['p_value']:.4g}, "
                    f"pass_rate={_fmt_pct(row['true_rate'])}, succ_pass={_fmt_pct(row['success_true'])}, succ_fail={_fmt_pct(row['success_false'])}"
                )
            for layer in ("regime", "prefilter", "gate", "entry_filter"):
                bad = _top_lines(stats, scope, layer, n=2)
                if bad:
                    lines.append(f"  - weakest {layer} rules:")
                    lines.extend([f"    {b}" for b in bad])
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-hoc layer effectiveness diagnostics by EMA1200 regime."
    )
    parser.add_argument("--strategies", default="bpc,me,tpc,srb")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--config-root", default="config/strategies")
    parser.add_argument("--out-dir", default="")
    parser.add_argument(
        "--predictions",
        default="",
        help="Override predictions.parquet path (single strategy or shared file)",
    )
    parser.add_argument(
        "--strict-locked-features",
        action="store_true",
        help=(
            "Pre-deploy contract: 任何 locked 规则缺特征则 BLOCKED（非零退出码）。"
            "regime/prefilter/gate/entry 任一层有 locked rule missing → 阻断。"
        ),
    )
    args = parser.parse_args()

    strategies = [s.strip() for s in str(args.strategies).split(",") if s.strip()]
    results_root = Path(args.results_root).resolve()
    config_root = Path(args.config_root).resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir).resolve()
        if str(args.out_dir).strip()
        else results_root / "posthoc_layer_effectiveness" / ts
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_override = (
        Path(args.predictions).resolve() if str(args.predictions).strip() else None
    )

    report: List[Dict[str, Any]] = []
    for strategy in strategies:
        report.append(
            analyze_strategy(
                results_root,
                config_root,
                strategy,
                predictions_path=pred_override,
            )
        )

    out_json = out_dir / "report.json"
    out_md = out_dir / "report.md"
    out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(report, out_md)
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")

    if args.strict_locked_features:
        blockers: List[str] = []
        for strat in report:
            locked_missing = strat.get("locked_features_missing", {}) or {}
            for layer, by_rule in locked_missing.items():
                if not by_rule:
                    continue
                for rule_name, feats in by_rule.items():
                    blockers.append(
                        f"{strat['strategy']}/{layer}/{rule_name}: missing {feats}"
                    )
        if blockers:
            blocked_path = out_dir / "BLOCKED.txt"
            blocked_path.write_text(
                "Pre-deploy contract failed (locked features missing):\n\n"
                + "\n".join(f"- {b}" for b in blockers)
                + "\n",
                encoding="utf-8",
            )
            print("BLOCKED — locked-feature contract violated:")
            for b in blockers:
                print(f"  - {b}")
            print(f"  see {blocked_path}")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
