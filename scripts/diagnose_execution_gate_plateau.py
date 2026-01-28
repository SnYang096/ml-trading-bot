#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec
from src.time_series_model.core.constitution.execution_evidence import (
    load_evidence_quantiles,
)
from src.time_series_model.diagnostics.kpi_gate import check_kpi_gate
from src.time_series_model.live.meta_router_config import load_meta_router_live_config
from src.time_series_model.live.tree_gate import apply_gate_rules
from src.time_series_model.nnmultihead.strategy_profile import (
    ExecutionArchetype,
    load_execution_archetypes_registry,
)


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _ensure_timestamp_col(df: pd.DataFrame, *, col: str = "timestamp") -> pd.DataFrame:
    if col in df.columns:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out[col] = out.index
        return out
    return df


def _month_range(start: str, end: str) -> List[str]:
    s = pd.Timestamp(start).to_period("M").to_timestamp()
    e = pd.Timestamp(end).to_period("M").to_timestamp()
    months = []
    cur = s
    while cur <= e:
        months.append(cur.strftime("%Y-%m"))
        cur = (cur + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp()
    return months


def _read_feature_store_df(
    *,
    root: Path,
    layer: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    cols: List[str],
) -> pd.DataFrame:
    store = FeatureStore(root)
    spec = FeatureStoreSpec(layer=layer, symbol=symbol, timeframe=timeframe)
    months = _month_range(start_date, end_date)
    parts = []
    for m in months:
        if not store.has_month(spec, m):
            continue
        df_m = store.read_month(spec, m)
        keep = [c for c in cols if c in df_m.columns]
        if keep:
            parts.append(df_m[keep])
    if not parts:
        return pd.DataFrame(columns=cols)
    df = pd.concat(parts, axis=0).sort_index()
    return df


def _build_quantiles(
    *,
    df: pd.DataFrame,
    key: str,
    qs: List[float],
) -> Dict[str, float]:
    if key not in df.columns:
        return {}
    vals = pd.to_numeric(df[key], errors="coerce").dropna()
    if vals.empty:
        return {}
    return {str(q): float(vals.quantile(q)) for q in qs}


def _collect_gate_rule_keys(arches: Dict[str, ExecutionArchetype]) -> List[str]:
    keys: List[str] = []
    for arch in arches.values():
        rules = (arch.gate_rules or {}).get("rules") or []
        for r in rules:
            k = str(r.get("key") or "").strip()
            if k:
                keys.append(k)
    return sorted(set(keys))


def _override_gate_rules(
    gate_rules: Dict[str, Any], *, sweep_key: str, q: float
) -> Dict[str, Any]:
    out = dict(gate_rules or {})
    rules = list(out.get("rules") or [])
    new_rules = []
    for r in rules:
        r2 = dict(r)
        if str(r2.get("key") or "") == sweep_key and str(
            r2.get("kind") or ""
        ).startswith("quantile_"):
            r2["quantile"] = float(q)
        new_rules.append(r2)
    out["rules"] = new_rules
    return out


def _enabled_archetypes(
    *, db_path: str, archetypes: Dict[str, ExecutionArchetype]
) -> List[str]:
    cfg = load_meta_router_live_config(db_path=db_path)
    xs = cfg.enabled_archetypes or []
    return [x for x in xs if x in archetypes]


def _compute_returns_from_archetype(
    df: pd.DataFrame,
    *,
    archetype_col: str = "gate_archetype",
    ret_mean_col: str = "ret_mean",
    ret_trend_col: str = "ret_trend",
) -> np.ndarray:
    """
    Select ret_mean or ret_trend based on archetype.
    - TC/TE → ret_trend
    - FR/ET → ret_mean
    """
    if len(df) == 0:
        return np.array([], dtype=float)

    ret = np.zeros(len(df), dtype=float)
    ret_mean = (
        pd.to_numeric(df.get(ret_mean_col), errors="coerce").fillna(0.0).to_numpy()
    )
    ret_trend = (
        pd.to_numeric(df.get(ret_trend_col), errors="coerce").fillna(0.0).to_numpy()
    )

    archetype = df.get(archetype_col)
    if archetype is None:
        # Fallback to mode for backward compatibility
        mode = df.get("mode")
        if mode is not None:
            mode_str = mode.astype(str).str.upper().to_numpy()
            ret[mode_str == "MEAN"] = ret_mean[mode_str == "MEAN"]
            ret[mode_str == "TREND"] = ret_trend[mode_str == "TREND"]
        return ret

    archetype_str = archetype.astype(str).str.upper().to_numpy()

    # TC/TE → ret_trend
    trend_mask = np.array([("TC" in a or "TE" in a) for a in archetype_str])
    ret[trend_mask] = ret_trend[trend_mask]

    # FR/ET → ret_mean
    mean_mask = np.array([("FR" in a or "ET" in a) for a in archetype_str])
    ret[mean_mask] = ret_mean[mean_mask]

    return ret


def _compute_returns_from_mode(df: pd.DataFrame) -> np.ndarray:
    """Legacy function for backward compatibility. Use _compute_returns_from_archetype instead."""
    return _compute_returns_from_archetype(df)


def _max_dd(arr: np.ndarray) -> float:
    if not arr.size:
        return 0.0
    eq = np.cumsum(arr)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    return float(dd.max()) if dd.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Execution-layer plateau sweep (joint gate + execution KPIs)."
    )
    ap.add_argument("--feature-store-root", default="feature_store")
    ap.add_argument("--layer", required=True)
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols")
    ap.add_argument("--timeframe", default="240T")
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--mode", required=True, help="mode_3action parquet/csv")
    ap.add_argument("--logs", required=True, help="logs_3action parquet/csv")
    ap.add_argument(
        "--registry",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="Execution archetypes registry yaml",
    )
    ap.add_argument(
        "--db-path",
        default=os.getenv("MLBOT_ORDER_MANAGEMENT_DB_PATH", "data/order_management.db"),
        help="Order management DB path (live_config stored here)",
    )
    ap.add_argument("--sweep-key", default="vpin")
    ap.add_argument("--q-grid", default="0.55,0.60,0.65,0.70,0.75,0.80")
    ap.add_argument("--quantiles", default="0.1,0.5,0.9")
    ap.add_argument(
        "--evidence-quantiles",
        default=None,
        help="Optional evidence_quantiles.json (per-symbol or GLOBAL).",
    )
    ap.add_argument(
        "--gate-yaml",
        default="config/kpi_gates/nnmh_execution_layer.yaml",
        help="KPI gate yaml for auto selection",
    )
    ap.add_argument(
        "--require-gate",
        action="store_true",
        help="Only select q from gate-passing candidates.",
    )
    ap.add_argument(
        "--plateau-frac",
        type=float,
        default=0.05,
        help="Plateau cutoff as fraction of |best_score|.",
    )
    ap.add_argument(
        "--score-key",
        default="gate_exec_score",
        help="Column to optimize (default: gate_exec_score).",
    )
    ap.add_argument("--out", required=True, help="Output directory for report")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_grid = [float(x) for x in str(args.q_grid).split(",") if x.strip()]
    qs = [float(x) for x in str(args.quantiles).split(",") if x.strip()]
    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    sweep_key = str(args.sweep_key).strip()

    arches = load_execution_archetypes_registry(args.registry)
    rule_keys = _collect_gate_rule_keys(arches)
    if sweep_key and sweep_key not in rule_keys:
        rule_keys.append(sweep_key)

    quantiles_by_sym: Dict[str, Dict[str, Dict[str, float]]] = {}
    if args.evidence_quantiles:
        quantiles_by_sym = load_evidence_quantiles(args.evidence_quantiles) or {}
    else:
        root = Path(args.feature_store_root).resolve()
        for sym in symbols:
            df = _read_feature_store_df(
                root=root,
                layer=args.layer,
                symbol=sym,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date,
                cols=rule_keys,
            )
            if df.empty:
                continue
            quantiles_by_sym[sym] = {
                k: _build_quantiles(df=df, key=k, qs=qs) for k in rule_keys
            }

    mode_df = _ensure_timestamp_col(_read_any(Path(args.mode)))
    logs_df = _ensure_timestamp_col(_read_any(Path(args.logs)))
    mode_df["symbol"] = mode_df["symbol"].astype(str)
    logs_df["symbol"] = logs_df["symbol"].astype(str)
    mode_df["timestamp"] = pd.to_datetime(mode_df["timestamp"], errors="coerce")
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"], errors="coerce")

    feats = []
    root = Path(args.feature_store_root).resolve()
    for sym in symbols:
        df = _read_feature_store_df(
            root=root,
            layer=args.layer,
            symbol=sym,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            cols=rule_keys,
        )
        if df.empty:
            continue
        df = df.copy()
        if "timestamp" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df["timestamp"] = pd.to_datetime(df.index, errors="coerce")
        df["symbol"] = sym
        feats.append(df)
    feats_df = pd.concat(feats, axis=0) if feats else pd.DataFrame()
    if not feats_df.empty:
        if getattr(feats_df.index, "name", None) == "timestamp":
            feats_df = feats_df.reset_index(drop=True)
        feats_df["timestamp"] = pd.to_datetime(feats_df["timestamp"], errors="coerce")
        feats_df["symbol"] = feats_df["symbol"].astype(str)

    merged = mode_df.merge(
        feats_df, on=["symbol", "timestamp"], how="left", suffixes=("", "_feat")
    )

    gate = None
    try:
        gate = yaml.safe_load(Path(args.gate_yaml).read_text(encoding="utf-8")) or {}
    except Exception:
        gate = None

    rows = []
    for q in q_grid:
        arches_q: Dict[str, ExecutionArchetype] = {}
        for name, arch in arches.items():
            arches_q[name] = ExecutionArchetype(
                name=arch.name,
                regime=arch.regime,
                required_conditions=list(arch.required_conditions),
                required_evidence=list(arch.required_evidence),
                evidence_rules=list(arch.evidence_rules),
                gate_rules=_override_gate_rules(
                    arch.gate_rules, sweep_key=sweep_key, q=q
                ),
                execution_constraints=dict(arch.execution_constraints),
            )

        gate_ok: List[bool] = []
        gate_decision: List[str] = []
        gate_arch: List[str] = []

        for _, row in merged.iterrows():
            regime = str(row.get("mode") or "NO_TRADE").upper()
            if regime == "NO_TRADE":
                gate_ok.append(True)
                gate_decision.append("no_trade")
                gate_arch.append("")
                continue
            candidates = _enabled_archetypes(
                db_path=str(args.db_path),
                archetypes=arches_q,
            )
            if not candidates:
                gate_ok.append(True)
                gate_decision.append("no_archetype")
                gate_arch.append("")
                continue

            quantiles = None
            if isinstance(quantiles_by_sym, dict):
                sym_q = quantiles_by_sym.get(str(row.get("symbol")))
                quantiles = sym_q if isinstance(sym_q, dict) else quantiles_by_sym

            chosen = None
            for arch_name in candidates:
                arch = arches_q.get(arch_name)
                if not arch:
                    continue
                if not arch.gate_rules:
                    chosen = arch_name
                    break
                ok, _ = apply_gate_rules(
                    gate_rules=arch.gate_rules,
                    features=row.to_dict(),
                    quantiles=quantiles,
                )
                if ok:
                    chosen = arch_name
                    break

            if chosen:
                gate_ok.append(True)
                gate_decision.append("allow")
                gate_arch.append(chosen)
            else:
                gate_ok.append(False)
                gate_decision.append("veto")
                gate_arch.append(candidates[0] if candidates else "")

        gated_mode = merged[["symbol", "timestamp", "mode"]].copy()
        gated_mode["gate_ok"] = gate_ok
        gated_mode["gate_decision"] = gate_decision
        gated_mode["gate_archetype"] = gate_arch
        gated_mode.loc[~gated_mode["gate_ok"].astype(bool), "mode"] = "NO_TRADE"

        gated_mode = gated_mode.rename(columns={"mode": "gate_mode"})
        joined = logs_df.merge(
            gated_mode[["symbol", "timestamp", "gate_mode"]],
            on=["symbol", "timestamp"],
            how="left",
        )
        joined["mode"] = joined["gate_mode"].fillna("NO_TRADE").astype(str)
        ret_base = _compute_returns_from_mode(logs_df)
        ret = _compute_returns_from_mode(joined)
        trade_mask = joined["mode"].str.upper() != "NO_TRADE"
        trade_rets = ret[trade_mask.to_numpy()]
        trade_rate = float(trade_mask.mean())
        trade_win_rate = float((trade_rets > 0).mean()) if trade_rets.size else 0.0
        trade_avg_ret = float(trade_rets.mean()) if trade_rets.size else 0.0
        avg_ret = float(ret.mean()) if ret.size else 0.0
        ret_std = float(ret.std(ddof=1)) if ret.size > 1 else 0.0
        sharpe = float(avg_ret / ret_std) if ret_std > 0 else 0.0
        max_dd_base = _max_dd(ret_base)
        max_dd = _max_dd(ret)
        tail_loss_reduction = (
            (max_dd_base - max_dd) / max_dd_base if max_dd_base > 0 else 0.0
        )
        pos_mask = ret_base > 0
        denom = float(pos_mask.sum())
        false_reject_rate = (
            float((~trade_mask.to_numpy() & pos_mask).sum()) / denom
            if denom > 0
            else 0.0
        )

        gate_ok_flag = None
        gate_failures = []
        if gate:
            metrics = {
                "router_diag__trade_rate": trade_rate,
                "router_diag__trade_win_rate": trade_win_rate,
                "router_diag__trade_avg_ret": trade_avg_ret,
                "rule_avg_max_dd": max_dd,
                "gate__tail_loss_reduction": tail_loss_reduction,
                "gate__false_reject_rate": false_reject_rate,
            }
            res = check_kpi_gate(metrics=metrics, gate=gate)
            gate_ok_flag = bool(res.ok)
            gate_failures = list(res.hard_failures)

        gate_exec_score = (
            trade_avg_ret
            + 0.1 * trade_win_rate
            - max_dd
            + 0.2 * tail_loss_reduction
            - 0.2 * false_reject_rate
        )

        rows.append(
            {
                "q": q,
                "trade_rate": trade_rate,
                "trade_win_rate": trade_win_rate,
                "trade_avg_ret": trade_avg_ret,
                "avg_return": avg_ret,
                "sharpe": sharpe,
                "max_dd": max_dd,
                "max_dd_base": max_dd_base,
                "tail_loss_reduction": tail_loss_reduction,
                "false_reject_rate": false_reject_rate,
                "score_sharpe_minus_dd": sharpe - max_dd,
                "exec_score": trade_avg_ret + 0.1 * trade_win_rate - max_dd,
                "gate_exec_score": gate_exec_score,
                "gate_ok": gate_ok_flag,
                "gate_failures": ";".join(gate_failures) if gate_failures else "",
                "trade_n": int(trade_mask.sum()),
            }
        )

    df_out = pd.DataFrame(rows).sort_values("q")
    df_out.to_csv(out_dir / "plateau.csv", index=False)

    score_key = str(args.score_key)
    if score_key not in df_out.columns:
        score_key = "sharpe"
    df_sel = df_out.copy()
    if args.require_gate and "gate_ok" in df_sel.columns:
        df_sel = df_sel[df_sel["gate_ok"] == True]  # noqa: E712
    best_score = (
        float(df_sel[score_key].max())
        if len(df_sel)
        else float(df_out[score_key].max())
    )
    cutoff = best_score - abs(best_score) * float(args.plateau_frac)
    plateau = (
        df_sel[df_sel[score_key] >= cutoff]
        if len(df_sel)
        else df_out[df_out[score_key] >= cutoff]
    )
    plateau_qs = plateau["q"].tolist()
    selected_q = (
        float(plateau["q"].median())
        if plateau_qs
        else float(df_out.iloc[df_out[score_key].idxmax()]["q"])
    )

    summary = {
        "score_key": score_key,
        "best_score": best_score,
        "cutoff": cutoff,
        "plateau_qs": plateau_qs,
        "selected_q": selected_q,
        "gate_yaml": str(args.gate_yaml),
        "require_gate": bool(args.require_gate),
        "sweep_key": sweep_key,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(
        "# Execution Gate Plateau\n\n"
        + df_out.to_markdown(index=False)
        + "\n\n## Summary\n\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(f"✅ Wrote: {out_dir / 'plateau.csv'}")
    print(f"✅ Wrote: {out_dir / 'report.md'}")
    print(f"✅ Wrote: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
