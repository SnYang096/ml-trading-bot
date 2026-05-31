"""Holdout IC screen: prune feature nodes by |IC| and best_lag (forward_rr target)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from src.research.stat_kernels.ic import rank_ic, resolve_target_col, shift_target_by_horizon
from src.time_series_model.strategies.models.feature_direction import expand_invert_features

DEFAULT_TARGET = "forward_rr"
DEFAULT_FEATURE_DEPS = Path("config/feature_dependencies.yaml")

_BASE_SKIP = {
    "datetime",
    "timestamp",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "_symbol",
    "label",
    "forward_rr",
    "split",
    "sample_weight",
    "atr",
    "atr14",
    "signal",
    "binary_signal",
    "pred",
    "score",
}

InvertMode = Literal["none", "auto"]


def load_feature_deps(path: Path | None = None) -> dict[str, Any]:
    deps_path = path or DEFAULT_FEATURE_DEPS
    data = yaml.safe_load(deps_path.read_text(encoding="utf-8")) or {}
    return data.get("features", data)


def column_to_nodes(deps: dict[str, Any]) -> dict[str, set[str]]:
    col_to_nodes: dict[str, set[str]] = {}
    for node, info in deps.items():
        if not isinstance(info, dict):
            continue
        for col in info.get("output_columns", []) or []:
            col_to_nodes.setdefault(str(col), set()).add(str(node))
    return col_to_nodes


def holdout_mask(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    if "datetime" in df.columns:
        ts = pd.to_datetime(df["datetime"])
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = pd.Series(df.index, index=df.index)
    else:
        raise ValueError("Need datetime column or DatetimeIndex")
    return (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))


def target_at_horizon(
    sub: pd.DataFrame, target: str, horizon: int
) -> pd.Series | None:
    tcol, need_shift = resolve_target_col(sub, target, horizon)
    if tcol is None:
        return None
    y = pd.to_numeric(sub[tcol], errors="coerce")
    if need_shift and horizon > 1:
        y = shift_target_by_horizon(y, horizon, sub)
    return y


def parse_horizons(raw: str | list[int]) -> list[int]:
    if isinstance(raw, list):
        return [int(x) for x in raw]
    return [int(x) for x in str(raw).split(",") if str(x).strip()]


def trim_nodes(
    requested_nodes: list[str],
    *,
    top_n: int | None,
    pool_nodes: set[str] | None,
    always_include: list[str] | None,
) -> list[str]:
    always = list(always_include or [])
    nodes = [n for n in requested_nodes if n not in always]
    if pool_nodes:
        nodes = [n for n in nodes if n in pool_nodes]
    if top_n is not None and top_n > 0:
        cap = max(0, top_n - len(always))
        nodes = nodes[:cap]
    return always + nodes


def ic_sign(rank_ic_val: float) -> str:
    if not np.isfinite(rank_ic_val):
        return "?"
    return "+" if rank_ic_val >= 0 else "-"


def screen_features(
    df: pd.DataFrame,
    *,
    holdout_start: str,
    holdout_end: str,
    horizons: list[int],
    min_ic: float,
    max_lag: int,
    min_n: int,
    target: str = DEFAULT_TARGET,
    feature_deps: dict[str, Any] | None = None,
) -> tuple[list[dict], list[dict], list[str]]:
    """Return (column_rows, node_summaries, requested_nodes)."""
    if target not in df.columns:
        raise KeyError(
            f"Target column {target!r} missing from parquet; run prepare-only first"
        )

    mask = holdout_mask(df, holdout_start, holdout_end)
    sub = df.loc[mask].copy()
    feature_cols = [
        c
        for c in sub.columns
        if c not in _BASE_SKIP
        and not str(c).startswith(("binary_signal", "signal_"))
        and pd.api.types.is_numeric_dtype(sub[c])
    ]

    rows: list[dict] = []
    for feat in feature_cols:
        x = pd.to_numeric(sub[feat], errors="coerce")
        best_lag: int | None = None
        best_ic = 0.0
        best_p = float("nan")
        best_n = 0
        per_h: dict[str, float] = {}
        for h in horizons:
            y = target_at_horizon(sub, target, h)
            if y is None:
                per_h[str(h)] = float("nan")
                continue
            rho, p, n = rank_ic(x, y, min_n=min_n)
            per_h[str(h)] = rho
            if np.isfinite(rho) and abs(rho) > abs(best_ic):
                best_ic = float(rho)
                best_lag = h
                best_p = p
                best_n = n
        if best_lag is None or best_lag > max_lag or abs(best_ic) < min_ic:
            continue
        rows.append(
            {
                "feature": feat,
                "best_lag": best_lag,
                "rank_ic": best_ic,
                "ic_sign": ic_sign(best_ic),
                "p_value": best_p,
                "n": best_n,
                "ic_by_horizon": per_h,
            }
        )

    rows.sort(key=lambda r: abs(r["rank_ic"]), reverse=True)
    deps = feature_deps if feature_deps is not None else load_feature_deps()
    col_to_nodes = column_to_nodes(deps)
    node_best: dict[str, dict] = {}
    for row in rows:
        for node in col_to_nodes.get(row["feature"], set()):
            prev = node_best.get(node)
            if prev is None or abs(row["rank_ic"]) > abs(prev["rank_ic"]):
                node_best[node] = {
                    "node": node,
                    "via_column": row["feature"],
                    "best_lag": row["best_lag"],
                    "rank_ic": row["rank_ic"],
                    "ic_sign": row["ic_sign"],
                }
    node_summaries = sorted(
        node_best.values(),
        key=lambda r: abs(r["rank_ic"]),
        reverse=True,
    )
    requested = sorted({n["node"] for n in node_summaries})
    return rows, node_summaries, requested


def invert_columns_for_nodes(
    node_summaries: list[dict],
    requested_nodes: list[str],
) -> list[str]:
    """Output column names to invert (auto mode): negative IC at best lag."""
    keep = set(requested_nodes)
    inv: list[str] = []
    for ns in node_summaries:
        if ns["node"] not in keep:
            continue
        if ns.get("rank_ic", 0) < 0:
            inv.append(str(ns["via_column"]))
    return sorted(set(inv))


def build_monotone_payload(
    requested_nodes: list[str],
    node_summaries: list[dict],
    *,
    feature_deps: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Review-only monotone constraint hints per expanded output column."""
    deps = feature_deps if feature_deps is not None else load_feature_deps()
    sign_by_node = {ns["node"]: ns.get("rank_ic", 0.0) for ns in node_summaries}
    expanded: list[dict[str, Any]] = []
    for node in requested_nodes:
        cols = (deps.get(node) or {}).get("output_columns") or [node]
        rank_ic_val = sign_by_node.get(node)
        if rank_ic_val is None or not np.isfinite(rank_ic_val):
            constraint = 0
        elif rank_ic_val >= 0:
            constraint = 1
        else:
            constraint = -1
        for col in cols:
            expanded.append(
                {
                    "column": str(col),
                    "node": node,
                    "monotone_constraint": constraint,
                }
            )
    return {
        "requested_features": requested_nodes,
        "expanded_columns": expanded,
        "monotone_constraints": [e["monotone_constraint"] for e in expanded],
        "note": (
            "Review-only. Align column order with trainer numeric feature columns "
            "before pasting into model_params.monotone_constraints."
        ),
    }


def _leading_comment_block(raw: str) -> str:
    lines = raw.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or stripped == "":
            out.append(line)
        else:
            break
    block = "\n".join(out).rstrip()
    return block + "\n\n" if block else ""


def _writeback_features_yaml(
    feat_yaml: Path,
    *,
    requested_nodes: list[str],
    invert_cols: list[str],
    max_lag: int,
    min_ic: float,
    target: str,
) -> None:
    """Update requested_features / invert_features, preserving other keys.

    - Preserves the leading comment block.
    - Preserves existing feature_pipeline sub-keys (exclude_columns, selector,
      post_processors, ensure_signal_column, ...).
    - Replaces (not appends) the description; no duplicate-key accumulation.
    """
    raw = feat_yaml.read_text(encoding="utf-8")
    doc = yaml.safe_load(raw) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"features yaml is not a mapping: {feat_yaml}")

    fp = doc.get("feature_pipeline")
    if not isinstance(fp, dict):
        fp = {}
    fp["requested_features"] = requested_nodes
    if invert_cols:
        fp["invert_features"] = expand_invert_features(invert_cols)
    else:
        fp.pop("invert_features", None)
    doc["feature_pipeline"] = fp
    doc["description"] = (
        f"IC-pruned @ holdout best_lag<={max_lag}, |IC|>={min_ic}, target={target}.\n"
    )

    comment = _leading_comment_block(raw)
    body = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
    feat_yaml.write_text(comment + body, encoding="utf-8")


def run_ic_prune(
    *,
    parquet: str | Path,
    output_dir: str | Path,
    holdout_start: str = "2025-10-01",
    holdout_end: str = "2026-04-01",
    horizons: str | list[int] = "1,2,3,4,5",
    max_lag: int = 5,
    min_ic: float = 0.02,
    min_n: int = 200,
    target: str = DEFAULT_TARGET,
    write_features_yaml: str | Path | None = None,
    top_n_nodes: int | None = None,
    intersect_features_yaml: str | Path | None = None,
    always_include: list[str] | None = None,
    invert_mode: InvertMode = "none",
    emit_monotone_constraints: str | Path | None = None,
    project_root: Path | None = None,
    feature_deps_path: Path | None = None,
) -> dict[str, Path | None]:
    """Run holdout IC screen; return paths to artifacts."""
    root = project_root or Path(__file__).resolve().parents[3]
    pq = Path(parquet)
    if not pq.is_absolute():
        pq = (root / pq).resolve()
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    deps_path = feature_deps_path
    if deps_path and not deps_path.is_absolute():
        deps_path = (root / deps_path).resolve()
    feature_deps = load_feature_deps(deps_path)

    df = pd.read_parquet(pq)
    horizon_list = parse_horizons(horizons)
    rows, node_summaries, requested_nodes = screen_features(
        df,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        horizons=horizon_list,
        min_ic=min_ic,
        max_lag=max_lag,
        min_n=min_n,
        target=target,
        feature_deps=feature_deps,
    )

    pool_nodes: set[str] | None = None
    if intersect_features_yaml:
        pool_path = Path(intersect_features_yaml)
        if not pool_path.is_absolute():
            pool_path = (root / pool_path).resolve()
        pool_data = yaml.safe_load(pool_path.read_text(encoding="utf-8")) or {}
        pool_nodes = set(
            pool_data.get("feature_pipeline", {}).get("requested_features") or []
        )

    requested_nodes = trim_nodes(
        requested_nodes,
        top_n=top_n_nodes,
        pool_nodes=pool_nodes,
        always_include=always_include or ["atr_f"],
    )
    keep_nodes = set(requested_nodes)
    node_summaries = [ns for ns in node_summaries if ns["node"] in keep_nodes]

    invert_cols: list[str] = []
    if invert_mode == "auto":
        invert_cols = invert_columns_for_nodes(node_summaries, requested_nodes)

    payload: dict[str, Any] = {
        "holdout": [holdout_start, holdout_end],
        "target": target,
        "horizons": horizon_list,
        "max_lag": max_lag,
        "min_ic": min_ic,
        "invert_mode": invert_mode,
        "n_pass_columns": len(rows),
        "n_pass_nodes": len(requested_nodes),
        "columns": rows,
        "nodes": node_summaries,
        "requested_features": requested_nodes,
    }
    if invert_cols:
        payload["invert_features"] = invert_cols

    json_path = out_dir / "ic_prune_holdout.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        f"# holdout IC prune (lag ≤ {max_lag}, target={target})",
        "",
        f"- parquet: `{pq}`",
        f"- holdout: {holdout_start} → {holdout_end}",
        f"- pass: |IC|≥{min_ic}, best_lag≤{max_lag}",
        f"- columns: {len(rows)}, nodes: {len(requested_nodes)}",
        f"- invert_mode: {invert_mode}",
        "",
        "| feature | best_lag | rank_ic | sign | n |",
        "|---|---:|---:|:--:|---:|",
    ]
    for r in rows[:60]:
        md.append(
            f"| {r['feature']} | {r['best_lag']} | {r['rank_ic']:.4f} | "
            f"{r['ic_sign']} | {r['n']} |"
        )
    if len(rows) > 60:
        md.append(f"| ... | | | | ({len(rows) - 60} more) |")
    md.extend(["", "## nodes (_f)", ""])
    md.append("| node | via_column | best_lag | rank_ic | sign |")
    md.append("|---|---|---:|---:|:--:|")
    for ns in node_summaries:
        md.append(
            f"| {ns['node']} | {ns['via_column']} | {ns['best_lag']} | "
            f"{ns['rank_ic']:.4f} | {ns['ic_sign']} |"
        )
    if invert_cols:
        md.extend(["", "## invert_features (output columns)", ""])
        for col in invert_cols:
            md.append(f"- {col}")
    md_path = out_dir / "ic_prune_holdout.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    mono_path: Path | None = None
    if emit_monotone_constraints not in (None, False, "false", "False"):
        mono_path = Path(str(emit_monotone_constraints))
        if not mono_path.is_absolute():
            mono_path = (root / mono_path).resolve()
        mono_path.parent.mkdir(parents=True, exist_ok=True)
        mono_payload = build_monotone_payload(
            requested_nodes, node_summaries, feature_deps=feature_deps
        )
        mono_path.write_text(
            yaml.safe_dump(mono_payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(f"wrote {mono_path}")

    feat_out: Path | None = None
    if write_features_yaml not in (None, False, "false", "False"):
        feat_yaml = Path(str(write_features_yaml))
        if not feat_yaml.is_absolute():
            feat_yaml = root / feat_yaml
        _writeback_features_yaml(
            feat_yaml,
            requested_nodes=requested_nodes,
            invert_cols=invert_cols,
            max_lag=max_lag,
            min_ic=min_ic,
            target=target,
        )
        feat_out = feat_yaml
        print(f"updated {feat_yaml} ({len(requested_nodes)} nodes)")

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return {
        "json": json_path,
        "md": md_path,
        "features_yaml": feat_out,
        "monotone_constraints": mono_path,
    }
