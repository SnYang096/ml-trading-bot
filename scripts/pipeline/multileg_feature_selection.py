from __future__ import annotations

from html import escape
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from src.features.normalization.raw_scale_columns import load_raw_scale_columns


PROJECT_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_REQUIRED_NODES_BY_TYPE = {
    "grid": {"bpc_soft_phase_f", "atr_f"},
    "dual_add_trend": {"trend_confidence_f", "bpc_soft_phase_f", "atr_f"},
}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _as_str_set(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {str(v).strip() for v in value if str(v).strip()}
    if isinstance(value, str) and value.strip():
        return {value.strip()}
    return set()


def _safe_name(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "candidate"
    return "".join(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in raw)


def _load_feature_yaml(path: Path) -> tuple[dict[str, Any], list[str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        data = {}
    fp = data.get("feature_pipeline") or {}
    requested = fp.get("requested_features") or []
    if not isinstance(requested, list):
        requested = []
    return data, [str(v) for v in requested if str(v).strip()]


def _selected_in_original_order(
    original: list[str], selected_nodes: set[str]
) -> list[str]:
    return [n for n in original if n in selected_nodes]


def _write_feature_subset_yaml(
    *,
    path: Path,
    data: Mapping[str, Any],
    selected: list[str],
    removed: list[str],
    strategy_type: str,
    candidate_name: str,
    tuned_candidate: Mapping[str, Any],
) -> None:
    obj = dict(data or {})
    fp = obj.get("feature_pipeline") or {}
    if not isinstance(fp, dict):
        fp = {}
    fp["requested_features"] = list(selected)
    obj["feature_pipeline"] = fp
    obj["_multileg_feature_selection"] = {
        "timestamp": datetime.now().isoformat(),
        "strategy_type": strategy_type,
        "source": "slow_snapshot",
        "selected_nodes": list(selected),
        "removed_nodes": list(removed),
        "candidate_name": candidate_name,
        "tuned_candidate": dict(tuned_candidate or {}),
    }
    path.write_text(
        yaml.safe_dump(obj, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _load_prefilter_yaml(config_dir: Path) -> dict[str, Any]:
    prefilter_path = config_dir / "archetypes" / "prefilter.yaml"
    if not prefilter_path.exists():
        return {}
    raw = yaml.safe_load(prefilter_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _load_feature_dependencies() -> dict[str, Any]:
    path = PROJECT_ROOT / "config" / "feature_dependencies.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    features = raw.get("features", {}) if isinstance(raw, Mapping) else {}
    return features if isinstance(features, dict) else {}


def _load_semantic_polarity(config_dir: Path, strategy: str) -> dict[str, str]:
    paths = [config_dir / "semantic_polarity.yaml"]
    if strategy:
        paths.append(
            PROJECT_ROOT
            / "config"
            / "strategies"
            / str(strategy)
            / "semantic_polarity.yaml"
        )
    for path in paths:
        if not path.exists():
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        polarity = raw.get("polarity", {}) if isinstance(raw, Mapping) else {}
        if not isinstance(polarity, Mapping):
            continue
        return {
            str(k).strip(): str(v).strip().lower()
            for k, v in polarity.items()
            if str(k).strip()
            and str(v).strip().lower()
            in {"higher_is_better", "lower_is_better", "unknown"}
        }
    return {}


def _as_float_list(value: Any, default: list[float]) -> list[float]:
    if not isinstance(value, list):
        return list(default)
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out or list(default)


def _as_float_pair_list(
    value: Any, default: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return list(default)
    out: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue
        try:
            lo = float(item[0])
            hi = float(item[1])
        except (TypeError, ValueError):
            continue
        if lo <= hi:
            out.append((lo, hi))
    return out or list(default)


def _threshold_token(value: float) -> str:
    return str(float(value)).rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _default_unknown_ranges_for_column(
    col: str, output_normalization_map: Mapping[str, Any]
) -> list[tuple[float, float]]:
    norm = str(output_normalization_map.get(col, "") or "").strip().lower()
    if norm == "bounded_-1_1":
        return [(-0.50, 0.50), (-0.25, 0.25)]
    return [(0.25, 0.75), (0.35, 0.65)]


def _polarity_for_column(col: str, polarity: Mapping[str, str]) -> str:
    if col in polarity:
        return str(polarity[col])
    base = str(col).rsplit("_", 1)[0] if str(col).rsplit("_", 1)[-1].isdigit() else col
    return str(polarity.get(base, "unknown"))


def _is_auto_rule_column_allowed(
    col: str,
    norm_type: str,
    raw_scale_columns: set[str],
) -> bool:
    c = str(col or "").strip()
    nt = str(norm_type or "").strip().lower()
    if not c:
        return False
    if c in raw_scale_columns:
        return False
    if nt in {
        "price_unit",
        "raw",
        "usd",
        "identity",
        "passthrough",
        "log1p_robust_rolling",
        "categorical",
        "count",
    }:
        return False
    return True


def _generic_threshold_variants_for_node(
    *,
    node: str,
    strategy: str,
    config_dir: Path,
    fs_cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    deps = _load_feature_dependencies()
    node_def = deps.get(str(node), {}) or {}
    output_columns = node_def.get("output_columns", []) or []
    if not isinstance(output_columns, list):
        return []
    polarity = _load_semantic_polarity(config_dir, strategy)
    compute_params = node_def.get("compute_params", {}) or {}
    output_normalization_map = (
        compute_params.get("output_normalization_map", {})
        if isinstance(compute_params, Mapping)
        else {}
    )
    if not isinstance(output_normalization_map, Mapping):
        output_normalization_map = {}
    threshold_cfg = fs_cfg.get("auto_rule_thresholds", {})
    if not isinstance(threshold_cfg, Mapping):
        threshold_cfg = {}
    higher_values = _as_float_list(
        threshold_cfg.get("higher_is_better"), [0.55, 0.65, 0.75]
    )
    lower_values = _as_float_list(
        threshold_cfg.get("lower_is_better"), [0.45, 0.35, 0.25]
    )
    unknown_ranges = _as_float_pair_list(threshold_cfg.get("unknown_range"), [])
    max_variants = int(fs_cfg.get("max_auto_rule_variants_per_node", 8) or 8)
    raw_scale_columns = load_raw_scale_columns()

    variants: list[dict[str, Any]] = []
    for col_raw in output_columns:
        col = str(col_raw).strip()
        norm_type = str(output_normalization_map.get(col, "") or "")
        if not _is_auto_rule_column_allowed(col, norm_type, raw_scale_columns):
            continue
        pol = _polarity_for_column(col, polarity)
        if pol == "higher_is_better":
            for value in higher_values:
                variants.append(
                    {
                        "suffix": f"{_safe_name(col)}_gte_{_threshold_token(value)}",
                        "rules": [
                            {"feature": col, "operator": ">=", "value": float(value)}
                        ],
                        "source": "semantic_polarity_threshold_scan",
                    }
                )
        elif pol == "lower_is_better":
            for value in lower_values:
                variants.append(
                    {
                        "suffix": f"{_safe_name(col)}_lte_{_threshold_token(value)}",
                        "rules": [
                            {"feature": col, "operator": "<=", "value": float(value)}
                        ],
                        "source": "semantic_polarity_threshold_scan",
                    }
                )
        elif pol == "unknown":
            ranges = unknown_ranges or _default_unknown_ranges_for_column(
                col, output_normalization_map
            )
            for lo, hi in ranges:
                variants.append(
                    {
                        "suffix": (
                            f"{_safe_name(col)}_range_"
                            f"{_threshold_token(lo)}_{_threshold_token(hi)}"
                        ),
                        "rules": [
                            {
                                "all_of": [
                                    {
                                        "feature": col,
                                        "operator": ">=",
                                        "value": float(lo),
                                    },
                                    {
                                        "feature": col,
                                        "operator": "<=",
                                        "value": float(hi),
                                    },
                                ]
                            }
                        ],
                        "source": "semantic_polarity_range_scan",
                    }
                )
    return variants[:max_variants]


def _write_prefilter_yaml(
    *,
    candidate_cfg_dir: Path,
    base_prefilter: Mapping[str, Any],
    rules: list[dict[str, Any]],
    strategy_type: str,
    candidate_name: str,
) -> None:
    prefilter_path = candidate_cfg_dir / "archetypes" / "prefilter.yaml"
    if not prefilter_path.exists():
        return
    obj = dict(base_prefilter or {})
    obj["rules"] = list(rules)
    obj["_multileg_feature_selection"] = {
        "timestamp": datetime.now().isoformat(),
        "strategy_type": strategy_type,
        "source": "slow_snapshot",
        "candidate_name": candidate_name,
        "rule_count": len(rules),
    }
    prefilter_path.write_text(
        yaml.safe_dump(obj, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _rule_variants_for_node(
    *,
    node: str,
    strategy: str,
    strategy_type: str,
    config_dir: Path,
    fs_cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    generic = _generic_threshold_variants_for_node(
        node=node, strategy=strategy, config_dir=config_dir, fs_cfg=fs_cfg
    )
    if generic:
        return generic
    return [{"suffix": "", "rules": [], "source": "no_semantic_rule"}]


def _build_candidate_sets(
    *,
    fs_cfg: Mapping[str, Any],
    strategy_type: str,
    all_available_nodes: set[str],
    candidate_pool_nodes: list[str],
) -> list[dict[str, Any]]:
    required = set(_DEFAULT_REQUIRED_NODES_BY_TYPE.get(strategy_type, set()))
    protected = required | _as_str_set(fs_cfg.get("protected_nodes"))
    global_keep = _as_str_set(fs_cfg.get("keep_nodes"))
    global_drop = _as_str_set(fs_cfg.get("drop_nodes"))
    raw_sets = fs_cfg.get("candidate_sets") or []
    if not isinstance(raw_sets, list):
        raw_sets = []

    out: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for idx, raw in enumerate(raw_sets, start=1):
        if not isinstance(raw, Mapping):
            continue
        name = _safe_name(str(raw.get("name", "") or f"candidate_{idx:02d}"))
        if name in seen_names:
            name = _safe_name(f"{name}_{idx:02d}")
        seen_names.add(name)

        mode = str(raw.get("mode", "") or "").strip().lower()
        if mode == "all_requested":
            nodes = set(all_available_nodes)
        else:
            nodes = _as_str_set(raw.get("nodes"))
        nodes |= protected | global_keep | _as_str_set(raw.get("keep_nodes"))
        nodes -= global_drop | _as_str_set(raw.get("drop_nodes"))
        nodes |= protected
        nodes &= all_available_nodes
        out.append(
            {
                "name": name,
                "mode": mode or "explicit",
                "nodes": sorted(nodes),
                "focus_node": str(raw.get("focus_node", "") or "").strip() or None,
            }
        )

    if out:
        return out

    base_nodes = ((protected | global_keep) - global_drop) & all_available_nodes
    pool = [
        n
        for n in candidate_pool_nodes
        if n in all_available_nodes and n not in global_drop
    ]
    candidate_rows: list[dict[str, Any]] = []

    def _ordered(nodes: set[str]) -> list[str]:
        ordered = [n for n in pool if n in nodes]
        ordered.extend(sorted(n for n in nodes if n not in ordered))
        return ordered

    candidate_rows.append(
        {
            "name": "core",
            "mode": "auto_core",
            "nodes": _ordered(base_nodes),
            "focus_node": None,
        }
    )
    for node in pool:
        if node in base_nodes:
            continue
        candidate_rows.append(
            {
                "name": f"core_plus_{_safe_name(node)}",
                "mode": "auto_single_add",
                "nodes": _ordered(base_nodes | {node}),
                "focus_node": node,
            }
        )
    candidate_rows.append(
        {
            "name": "full_default",
            "mode": "all_requested",
            "nodes": _ordered(all_available_nodes - global_drop),
            "focus_node": None,
        }
    )
    return candidate_rows


def _is_better_candidate(
    *,
    score: float,
    node_count: int,
    rule_count: int,
    best_score: float | None,
    best_node_count: int | None,
    best_rule_count: int | None,
) -> bool:
    if best_score is None:
        return True
    if score > best_score:
        return True
    if abs(score - best_score) <= 1e-12:
        if best_node_count is None or node_count < best_node_count:
            return True
        if node_count == best_node_count:
            return best_rule_count is None or rule_count < best_rule_count
        return False
    return False


def _metric(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    evaluation = row.get("evaluation") if isinstance(row, Mapping) else {}
    metrics = evaluation.get("metrics") if isinstance(evaluation, Mapping) else {}
    if not isinstance(metrics, Mapping):
        return default
    try:
        return float(metrics.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _semantic_note(node: str | None, strategy_type: str) -> str:
    n = str(node or "").strip()
    st = str(strategy_type or "").strip().lower()
    notes = {
        "atr_percentile_f": "volatility capacity filter: avoids extreme ATR regimes.",
        "volatility_regime_f": "regime width filter: keeps medium realized volatility.",
        "bb_width_normalized_pct_f": "compression filter: tests whether narrow bands improve entries.",
        "box_structure_f": "range-quality filter: tests stable, tradable boxes for inventory grids.",
        "ema_1200_position_f": "slow trend location filter: separates extended trend states.",
        "ema_1200_slope_f": "slow trend slope filter: tests directional regime exclusion.",
        "trend_confidence_f": "trend-strength filter for add-on trend inventory.",
    }
    if n in notes:
        return notes[n]
    if not n:
        if st == "grid":
            return "Core semantic chop + ATR baseline."
        if st == "dual_add_trend":
            return "Core trend confidence + chop/ATR baseline."
        return "Core required strategy features."
    return "Candidate feature has no dedicated semantic note yet."


def _format_rule(rule: Mapping[str, Any]) -> str:
    if "all_of" in rule and isinstance(rule.get("all_of"), list):
        return " AND ".join(
            _format_rule(r) for r in rule["all_of"] if isinstance(r, Mapping)
        )
    if "any_of" in rule and isinstance(rule.get("any_of"), list):
        return " OR ".join(
            _format_rule(r) for r in rule["any_of"] if isinstance(r, Mapping)
        )
    feature = escape(str(rule.get("feature", "")))
    operator = escape(str(rule.get("operator", "")))
    value = escape(str(rule.get("value", "")))
    return f"{feature} {operator} {value}".strip()


def _candidate_evidence(row: Mapping[str, Any], selected_name: str) -> str:
    if not row.get("valid", False):
        return "Skipped: invalid candidate configuration."
    score = row.get("score")
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        score_f = 0.0
    if score_f <= -1e5:
        return "Rejected by KPI penalty, usually too few trades or risk gate failure."
    if str(row.get("name", "")) == selected_name:
        return "Selected: best score after KPI constraints and tie-break."
    return "Rejected: weaker constrained score than selected candidate."


def _write_feature_selection_html(result: Mapping[str, Any], output_dir: Path) -> str:
    candidates = [c for c in result.get("candidates", []) if isinstance(c, Mapping)]
    selected = str(result.get("selected_candidate", "") or "")
    selected_score = result.get("selected_score")
    sorted_candidates = sorted(
        candidates,
        key=lambda c: float(c.get("score", -1e18) or -1e18),
        reverse=True,
    )
    runner_up = next(
        (c for c in sorted_candidates if str(c.get("name", "")) != selected), None
    )
    winner = next((c for c in candidates if str(c.get("name", "")) == selected), None)
    runner_score = runner_up.get("score") if isinstance(runner_up, Mapping) else None
    try:
        margin = float(selected_score or 0.0) - float(runner_score or 0.0)
    except (TypeError, ValueError):
        margin = 0.0

    rows: list[str] = []
    for row in sorted_candidates:
        rules = row.get("prefilter_rules") or []
        rule_text = (
            "<br/>".join(
                escape(_format_rule(rule))
                for rule in rules
                if isinstance(rule, Mapping)
            )
            or "-"
        )
        nodes = ", ".join(str(n) for n in (row.get("nodes") or []))
        rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('name', '')))}</td>"
            f"<td>{escape(_semantic_note(row.get('focus_node'), str(result.get('strategy_type', ''))))}</td>"
            f"<td>{escape(nodes)}</td>"
            f"<td>{rule_text}</td>"
            f"<td>{float(row.get('score', 0.0) or 0.0):.6f}</td>"
            f"<td>{_metric(row, 'n_trades'):.0f}</td>"
            f"<td>{_metric(row, 'total_r'):.6f}</td>"
            f"<td>{_metric(row, 'gross_bps_per_trade'):.2f}</td>"
            f"<td>{_metric(row, 'cost_bps_per_trade'):.2f}</td>"
            f"<td>{_metric(row, 'cost_coverage_ratio'):.3f}</td>"
            f"<td>{_metric(row, 'median_grid_per_side_span_to_1std'):.2f}</td>"
            f"<td>{_metric(row, 'median_grid_full_span_to_range'):.2f}</td>"
            f"<td>{_metric(row, 'win_rate'):.3f}</td>"
            f"<td>{_metric(row, 'max_drawdown_r'):.6f}</td>"
            f"<td>{_metric(row, 'worst_segment'):.6f}</td>"
            f"<td>{_metric(row, 'forced_rate'):.3f}</td>"
            f"<td>{escape(_candidate_evidence(row, selected))}</td>"
            "</tr>"
        )

    selected_rules = []
    if isinstance(winner, Mapping):
        selected_rules = [
            _format_rule(rule)
            for rule in (winner.get("prefilter_rules") or [])
            if isinstance(rule, Mapping)
        ]
    selected_rule_text = (
        "<br/>".join(escape(r) for r in selected_rules) or "No extra prefilter rule."
    )
    automation_gap = (
        "This report proves candidate quality by strategy-specific ablation backtests, "
        "but it is not yet a full BPC-style statistical report: it does not compute SHAP, "
        "bootstrap confidence intervals, repeated walk-forward folds, or out-of-sample "
        "adoption-vs-baseline degradation statistics for every candidate."
    )

    html = f"""<html>
<head>
  <meta charset="utf-8" />
  <title>Multi-leg Feature Selection Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: right; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    td:first-child, th:first-child, td:nth-child(2), th:nth-child(2), td:nth-child(3), th:nth-child(3), td:nth-child(4), th:nth-child(4), td:last-child, th:last-child {{ text-align: left; }}
    .note {{ max-width: 980px; color: #4b5563; }}
    .ok {{ color: #065f46; font-weight: 700; }}
    .warn {{ color: #92400e; font-weight: 700; }}
    code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>Multi-leg Feature Selection Report</h1>
  <p class="note">
    Strategy: <code>{escape(str(result.get("strategy", "")))}</code>;
    selector: <code>{escape(str(result.get("selector", "")))}</code>;
    tie-breaker: <code>{escape(str(result.get("tie_breaker", "")))}</code>.
  </p>
  <h2>Conclusion</h2>
  <p>
    Selected candidate: <span class="ok">{escape(selected or "none")}</span>,
    score={float(selected_score or 0.0):.6f},
    runner-up margin={margin:.6f},
    selected rule count={int(result.get("selected_rule_count", 0) or 0)}.
  </p>
  <p class="note">Selected prefilter rules: {selected_rule_text}</p>
  <h2>Candidate Comparison</h2>
  <table>
    <thead>
      <tr>
        <th>Candidate</th><th>Semantic Fit</th><th>Feature Nodes</th><th>Rules</th>
        <th>Score</th><th>Trades</th><th>Total R</th><th>Gross bps/trade</th>
        <th>Cost bps/trade</th><th>Cost Coverage</th><th>Side Span / 1σ</th>
        <th>Full Span / Range</th><th>Win Rate</th>
        <th>Max DD</th><th>Worst Segment</th><th>Forced</th><th>Evidence</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  <h2>Automation Gap</h2>
  <p class="warn">{automation_gap}</p>
</body>
</html>
"""
    html_path = output_dir / "multileg_feature_selection.html"
    html_path.write_text(html, encoding="utf-8")
    return str(html_path)


def select_multileg_feature_subset(
    *,
    strategy: str,
    strategy_type: str,
    config_dir: Path,
    output_dir: Path,
    strategy_cfg: Mapping[str, Any] | None,
    best_calibration: Mapping[str, Any] | None,
    metrics: Mapping[str, Any] | None,
    evaluate_candidate: (
        Callable[[str, Path, set[str]], Mapping[str, Any]] | None
    ) = None,
) -> dict[str, Any]:
    """Evaluate config-driven multi-leg feature subsets and adopt a winner."""
    scfg = strategy_cfg or {}
    fs_cfg = scfg.get("multileg_feature_selection") or {}
    if not isinstance(fs_cfg, Mapping):
        fs_cfg = {}
    if not _as_bool(fs_cfg.get("enabled"), True):
        return {
            "strategy": strategy,
            "strategy_type": strategy_type,
            "enabled": False,
            "reason": "disabled_by_config",
        }

    best = best_calibration or {}
    tuned_candidate = best.get("tuned_candidate") or best.get("candidate") or {}
    if not isinstance(tuned_candidate, Mapping):
        tuned_candidate = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    originals: dict[str, dict[str, Any]] = {}
    requested_by_file: dict[str, list[str]] = {}
    for yaml_name in ("features.yaml", "features_prefilter.yaml"):
        path = config_dir / yaml_name
        if not path.exists():
            continue
        data, requested = _load_feature_yaml(path)
        if not requested:
            continue
        originals[yaml_name] = data
        requested_by_file[yaml_name] = requested

    if not requested_by_file:
        result = {
            "strategy": strategy,
            "strategy_type": strategy_type,
            "enabled": True,
            "selector": "config_polymorphic_multileg_ablation",
            "reason": "no_feature_yaml",
            "best_calibration": best,
            "metrics": dict(metrics or {}),
            "candidates": [],
            "files": [],
        }
        artifact_path = output_dir / "multileg_feature_selection.json"
        artifact_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        result["artifact_path"] = str(artifact_path)
        return result

    all_available_nodes = {n for nodes in requested_by_file.values() for n in nodes}
    candidate_pool_nodes = (
        requested_by_file.get("features_prefilter.yaml")
        or requested_by_file.get("features.yaml")
        or sorted(all_available_nodes)
    )
    candidate_sets = _build_candidate_sets(
        fs_cfg=fs_cfg,
        strategy_type=str(strategy_type or "").strip().lower(),
        all_available_nodes=all_available_nodes,
        candidate_pool_nodes=candidate_pool_nodes,
    )
    base_prefilter = _load_prefilter_yaml(config_dir)
    base_rules = [
        r for r in (base_prefilter.get("rules") or []) if isinstance(r, Mapping)
    ]

    candidates_root = output_dir / "_feature_candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict[str, Any]] = []
    winner_idx: int | None = None
    winner_score: float | None = None
    winner_node_count: int | None = None
    winner_rule_count: int | None = None

    for idx, cset in enumerate(candidate_sets):
        base_name = str(cset.get("name", f"candidate_{idx+1:02d}"))
        selected_node_list = [str(n) for n in (cset.get("nodes") or [])]
        selected_nodes = set(selected_node_list)
        focus_node = str(cset.get("focus_node", "") or "").strip()
        rule_variants = (
            _rule_variants_for_node(
                node=focus_node,
                strategy=strategy,
                strategy_type=strategy_type,
                config_dir=config_dir,
                fs_cfg=fs_cfg,
            )
            if focus_node
            else [{"suffix": "", "rules": [], "source": "baseline"}]
        )

        for variant in rule_variants:
            suffix = str(variant.get("suffix", "") or "").strip()
            name = base_name if not suffix else f"{base_name}__{suffix}"
            candidate_rules = list(base_rules)
            candidate_rules.extend(
                [
                    dict(r)
                    for r in (variant.get("rules") or [])
                    if isinstance(r, Mapping)
                ]
            )
            candidate_cfg_dir = candidates_root / name / "strategy_config"
            shutil.rmtree(candidate_cfg_dir, ignore_errors=True)
            from src.config.strategy_layout import copy_strategy_package

            copy_strategy_package(config_dir, candidate_cfg_dir, dirs_exist_ok=True)
            _write_prefilter_yaml(
                candidate_cfg_dir=candidate_cfg_dir,
                base_prefilter=base_prefilter,
                rules=[dict(r) for r in candidate_rules if isinstance(r, Mapping)],
                strategy_type=strategy_type,
                candidate_name=name,
            )

            file_rows: list[dict[str, Any]] = []
            invalid = False
            for yaml_name, original in requested_by_file.items():
                selected = _selected_in_original_order(original, selected_nodes)
                if not selected:
                    invalid = True
                removed = [n for n in original if n not in selected]
                file_rows.append(
                    {
                        "file": yaml_name,
                        "original": list(original),
                        "selected": list(selected),
                        "removed": list(removed),
                    }
                )
                if selected:
                    _write_feature_subset_yaml(
                        path=candidate_cfg_dir / yaml_name,
                        data=originals[yaml_name],
                        selected=selected,
                        removed=removed,
                        strategy_type=strategy_type,
                        candidate_name=name,
                        tuned_candidate=tuned_candidate,
                    )

            row: dict[str, Any] = {
                "name": name,
                "mode": cset.get("mode", "explicit"),
                "nodes": selected_node_list,
                "focus_node": focus_node or None,
                "rule_source": str(variant.get("source", "") or ""),
                "rule_count": len(candidate_rules),
                "prefilter_rules": [
                    dict(r) for r in candidate_rules if isinstance(r, Mapping)
                ],
                "files": file_rows,
                "config_dir": str(candidate_cfg_dir),
            }
            if invalid:
                row["valid"] = False
                row["skipped"] = "empty_selection_in_file"
                candidate_rows.append(row)
                continue

            row["valid"] = True
            if evaluate_candidate is not None:
                try:
                    ev = dict(
                        evaluate_candidate(name, candidate_cfg_dir, selected_nodes)
                        or {}
                    )
                    row["evaluation"] = ev
                    if ev.get("score") is not None:
                        score = float(ev.get("score") or 0.0)
                        row["score"] = score
                        if _is_better_candidate(
                            score=score,
                            node_count=len(selected_nodes),
                            rule_count=len(candidate_rules),
                            best_score=winner_score,
                            best_node_count=winner_node_count,
                            best_rule_count=winner_rule_count,
                        ):
                            winner_score = score
                            winner_node_count = len(selected_nodes)
                            winner_rule_count = len(candidate_rules)
                            winner_idx = len(candidate_rows)
                except Exception as exc:  # pragma: no cover - runtime guard
                    row["valid"] = False
                    row["error"] = str(exc)
            else:
                row["score"] = 0.0
                if winner_idx is None:
                    winner_idx = len(candidate_rows)
                    winner_score = 0.0
                    winner_node_count = len(selected_nodes)
                    winner_rule_count = len(candidate_rules)
            candidate_rows.append(row)

    if winner_idx is None:
        for i, row in enumerate(candidate_rows):
            if row.get("valid"):
                winner_idx = i
                break

    selected_name = ""
    adopted_files: list[dict[str, Any]] = []
    winner_row: dict[str, Any] = {}
    if winner_idx is not None and 0 <= winner_idx < len(candidate_rows):
        winner_row = candidate_rows[winner_idx]
        selected_name = str(winner_row.get("name", ""))
        winner_cfg_dir = Path(str(winner_row.get("config_dir", "") or ""))
        winner_nodes = set(winner_row.get("nodes") or [])
        for yaml_name, original in requested_by_file.items():
            src = winner_cfg_dir / yaml_name
            if not src.exists():
                continue
            dst = config_dir / yaml_name
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            after = _selected_in_original_order(original, winner_nodes)
            adopted_files.append(
                {
                    "file": yaml_name,
                    "original": list(original),
                    "selected": list(after),
                    "removed": [n for n in original if n not in after],
                }
            )
        src_prefilter = winner_cfg_dir / "archetypes" / "prefilter.yaml"
        dst_prefilter = config_dir / "archetypes" / "prefilter.yaml"
        if src_prefilter.exists():
            dst_prefilter.parent.mkdir(parents=True, exist_ok=True)
            dst_prefilter.write_text(
                src_prefilter.read_text(encoding="utf-8"), encoding="utf-8"
            )
            adopted_files.append(
                {
                    "file": "archetypes/prefilter.yaml",
                    "rule_count": int(winner_row.get("rule_count", 0) or 0),
                }
            )

    result = {
        "strategy": strategy,
        "strategy_type": strategy_type,
        "enabled": True,
        "selector": "config_polymorphic_multileg_ablation",
        "selected_candidate": selected_name,
        "selected_score": winner_score,
        "selected_rule_count": winner_rule_count,
        "tie_breaker": "higher_score_then_fewer_features_then_fewer_rules",
        "best_calibration": best,
        "metrics": dict(metrics or {}),
        "candidates": candidate_rows,
        "files": adopted_files,
    }
    if winner_row:
        result["winner"] = winner_row
    html_path = _write_feature_selection_html(result, output_dir)
    result["html_report_path"] = html_path
    artifact_path = output_dir / "multileg_feature_selection.json"
    artifact_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    result["artifact_path"] = str(artifact_path)
    return result
