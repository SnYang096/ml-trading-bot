#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from src.feature_store.feature_store import FeatureStore, FeatureStoreSpec
from src.time_series_model.core.constitution.execution_evidence import (
    compute_execution_evidence,
)
from src.time_series_model.diagnostics.kpi_gate import check_kpi_gate


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


def _load_archetype_rules(path: str, archetype: str) -> List[Dict]:
    obj = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    regimes = obj.get("regimes") or {}
    for rr in regimes.values():
        arch = (rr or {}).get("archetypes") or {}
        if archetype in arch:
            return list((arch.get(archetype) or {}).get("evidence_rules") or [])
    overlays = obj.get("overlays") or {}
    if archetype in overlays:
        return list((overlays.get(archetype) or {}).get("evidence_rules") or [])
    raise ValueError(f"Archetype not found in registry: {archetype}")


def _extract_rule_keys(rules: List[Dict]) -> List[str]:
    keys = []
    for r in rules:
        k = str(r.get("key") or "").strip()
        if k:
            keys.append(k)
    return sorted(set(keys))


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


def _apply_quantile_override(
    rules: List[Dict], *, sweep_key: str, q: float
) -> List[Dict]:
    out = []
    for r in rules:
        r2 = dict(r)
        if str(r2.get("key") or "") == sweep_key and str(
            r2.get("kind") or ""
        ).startswith("quantile_"):
            r2["quantile"] = float(q)
        out.append(r2)
    return out


def _compute_returns_from_mode(df: pd.DataFrame) -> np.ndarray:
    mode = df.get("mode")
    if mode is None:
        return np.array([], dtype=float)
    ret = np.zeros(len(df), dtype=float)
    ret_mean = pd.to_numeric(df.get("ret_mean"), errors="coerce").fillna(0.0).to_numpy()
    ret_trend = (
        pd.to_numeric(df.get("ret_trend"), errors="coerce").fillna(0.0).to_numpy()
    )
    mode_str = mode.astype(str).str.upper().to_numpy()
    ret[mode_str == "MEAN"] = ret_mean[mode_str == "MEAN"]
    ret[mode_str == "TREND"] = ret_trend[mode_str == "TREND"]
    return ret


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plateau sweep for quantile-based evidence rules."
    )
    ap.add_argument("--feature-store-root", default="feature_store")
    ap.add_argument("--layer", required=True)
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols")
    ap.add_argument("--timeframe", default="240T")
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument(
        "--registry",
        default="config/nnmultihead/execution_archetypes.yaml",
        help="Execution archetypes registry yaml",
    )
    ap.add_argument("--archetype", required=True, help="Archetype to evaluate")
    ap.add_argument(
        "--sweep-key", default="vpin", help="Feature key whose quantile to sweep"
    )
    ap.add_argument(
        "--q-grid",
        default="0.55,0.60,0.65,0.70,0.75,0.80",
        help="Comma-separated quantiles to sweep",
    )
    ap.add_argument(
        "--quantiles",
        default="0.1,0.5,0.9",
        help="Comma-separated quantiles to compute for keys",
    )
    ap.add_argument(
        "--quantiles-json",
        default=None,
        help="Optional evidence_quantiles.json (per-symbol or GLOBAL).",
    )
    ap.add_argument("--logs", default=None, help="logs_3action parquet/csv (optional)")
    ap.add_argument("--out", required=True, help="Output directory for report")
    ap.add_argument(
        "--gate-yaml",
        default="config/kpi_gates/nnmh_execution_layer.yaml",
        help="KPI gate yaml for auto selection",
    )
    ap.add_argument(
        "--require-gate",
        action="store_true",
        help="Only select q from gate-passing candidates (if logs present).",
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
    ap.add_argument(
        "--global",
        dest="global_pool",
        action="store_true",
        help="Pool all symbols into GLOBAL quantiles.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_grid = [float(x) for x in str(args.q_grid).split(",") if x.strip()]
    qs = [float(x) for x in str(args.quantiles).split(",") if x.strip()]
    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]

    rules = _load_archetype_rules(args.registry, args.archetype)
    rule_keys = _extract_rule_keys(rules)
    sweep_key = str(args.sweep_key).strip()
    if sweep_key and sweep_key not in rule_keys:
        rule_keys.append(sweep_key)

    # Load quantiles
    quantiles_by_sym: Dict[str, Dict[str, Dict[str, float]]] = {}
    if args.quantiles_json:
        quantiles_by_sym = json.loads(
            Path(args.quantiles_json).read_text(encoding="utf-8")
        )
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
        if args.global_pool:
            frames = []
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
                if not df.empty:
                    frames.append(df)
            if frames:
                df = pd.concat(frames, axis=0)
                quantiles_by_sym["GLOBAL"] = {
                    k: _build_quantiles(df=df, key=k, qs=qs) for k in rule_keys
                }

    # Load logs (optional)
    logs = None
    if args.logs:
        p = Path(args.logs)
        if p.suffix.lower() == ".parquet":
            logs = pd.read_parquet(p)
        else:
            logs = pd.read_csv(p)
        logs["timestamp"] = pd.to_datetime(logs["timestamp"], utc=True, errors="coerce")

    gate = None
    try:
        gate = yaml.safe_load(Path(args.gate_yaml).read_text(encoding="utf-8")) or {}
    except Exception:
        gate = None

    rows = []
    for q in q_grid:
        q_rules = _apply_quantile_override(rules, sweep_key=sweep_key, q=q)
        evidence_true = []
        gated_rates = []
        returns = []
        trade_win_rates = []
        trade_avg_rets = []
        max_dd_base_list = []
        max_dd_gated_list = []
        tail_loss_reduction_list = []
        false_reject_rate_list = []

        for sym in symbols:
            qmap = quantiles_by_sym.get(sym) or quantiles_by_sym.get("GLOBAL") or {}
            df = _read_feature_store_df(
                root=Path(args.feature_store_root).resolve(),
                layer=args.layer,
                symbol=sym,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date,
                cols=rule_keys,
            )
            if df.empty:
                continue
            # Compute evidence per row
            ev = []
            for _, r in df.iterrows():
                feats = {k: r.get(k) for k in rule_keys}
                ev_flag = compute_execution_evidence(
                    features=feats, rules=q_rules, quantiles=qmap
                )
                ev.append(all(ev_flag.values()) if ev_flag else False)
            ev = np.asarray(ev, dtype=bool)
            evidence_true.append(ev.mean() if len(ev) else 0.0)

            if logs is not None:
                lg = logs[logs["symbol"] == sym].copy()
                lg = lg.dropna(subset=["timestamp"])
                if lg.empty:
                    continue
                lg = lg.set_index("timestamp")
                df2 = df.copy()
                df2.index = pd.to_datetime(df2.index, utc=True, errors="coerce")
                df2["evidence_ok"] = ev
                merged = lg.join(df2[["evidence_ok"]], how="left")
                merged["evidence_ok"] = merged["evidence_ok"].fillna(False)
                merged["mode"] = merged["mode"].astype(str)
                ret_base = _compute_returns_from_mode(merged)
                merged.loc[~merged["evidence_ok"], "mode"] = "NO_TRADE"
                ret = _compute_returns_from_mode(merged)
                trade_mask = merged["mode"].str.upper() != "NO_TRADE"
                trade_rate = float(trade_mask.mean())
                trade_rets = ret[trade_mask.to_numpy()]
                win_rate = float((trade_rets > 0).mean()) if trade_rets.size else 0.0
                avg_ret = float(trade_rets.mean()) if trade_rets.size else 0.0
                gated_rates.append(trade_rate)
                trade_win_rates.append(win_rate)
                trade_avg_rets.append(avg_ret)
                returns.append(ret)

                # Gate KPIs: tail loss reduction + false reject rate (proxy)
                def _max_dd(arr: np.ndarray) -> float:
                    if not arr.size:
                        return 0.0
                    eq = np.cumsum(arr)
                    peak = np.maximum.accumulate(eq)
                    dd = peak - eq
                    return float(dd.max()) if dd.size else 0.0

                max_dd_base = _max_dd(ret_base)
                max_dd_gated = _max_dd(ret)
                max_dd_base_list.append(max_dd_base)
                max_dd_gated_list.append(max_dd_gated)
                if max_dd_base > 0:
                    tail_loss_reduction_list.append(
                        (max_dd_base - max_dd_gated) / max_dd_base
                    )
                else:
                    tail_loss_reduction_list.append(0.0)
                pos_mask = ret_base > 0
                denom = float(pos_mask.sum())
                if denom > 0:
                    false_reject_rate_list.append(
                        float((~merged["evidence_ok"].to_numpy() & pos_mask).sum())
                        / denom
                    )
                else:
                    false_reject_rate_list.append(0.0)

        evidence_rate = float(np.mean(evidence_true)) if evidence_true else 0.0
        trade_rate = float(np.mean(gated_rates)) if gated_rates else 0.0
        trade_win_rate = float(np.mean(trade_win_rates)) if trade_win_rates else 0.0
        trade_avg_ret = float(np.mean(trade_avg_rets)) if trade_avg_rets else 0.0
        ret_arr = np.concatenate(returns) if returns else np.array([], dtype=float)
        avg_ret = float(ret_arr.mean()) if ret_arr.size else 0.0
        ret_std = float(ret_arr.std(ddof=1)) if ret_arr.size > 1 else 0.0
        sharpe = float(avg_ret / ret_std) if ret_std > 0 else 0.0
        max_dd = float(np.mean(max_dd_gated_list)) if max_dd_gated_list else 0.0
        max_dd_base = float(np.mean(max_dd_base_list)) if max_dd_base_list else 0.0
        tail_loss_reduction = (
            float(np.mean(tail_loss_reduction_list))
            if tail_loss_reduction_list
            else 0.0
        )
        false_reject_rate = (
            float(np.mean(false_reject_rate_list)) if false_reject_rate_list else 0.0
        )
        gate_ok = None
        gate_failures = []
        if gate and logs is not None:
            metrics = {
                "router_diag__trade_rate": trade_rate,
                "router_diag__trade_win_rate": trade_win_rate,
                "router_diag__trade_avg_ret": trade_avg_ret,
                "rule_avg_max_dd": max_dd,
                "gate__tail_loss_reduction": tail_loss_reduction,
                "gate__false_reject_rate": false_reject_rate,
            }
            res = check_kpi_gate(metrics=metrics, gate=gate)
            gate_ok = bool(res.ok)
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
                "evidence_true_rate": evidence_rate,
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
                "gate_ok": gate_ok,
                "gate_failures": ";".join(gate_failures) if gate_failures else "",
                "n": int(ret_arr.size),
            }
        )

    df_out = pd.DataFrame(rows).sort_values("q")
    df_out.to_csv(out_dir / "plateau.csv", index=False)
    # Auto-select plateau q
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
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(
        "# Evidence Quantile Plateau\n\n"
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
