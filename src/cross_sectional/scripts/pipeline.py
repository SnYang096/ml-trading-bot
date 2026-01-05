#!/usr/bin/env python3
"""
YAML-driven cross-sectional pipeline:
  - build/load panel (FeatureStore or existing parquet)
  - factor eval (IC + long/short)
  - factor select (existing auto_select_factors logic)
  - backtest selected (already embedded in factor eval outputs)

This is intentionally a thin orchestration wrapper around existing CS scripts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import pandas as pd
import datetime as _dt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _month_keys(start_date: str, end_date: str) -> List[str]:
    start = pd.Timestamp(start_date, tz="UTC").normalize().replace(day=1)
    end = pd.Timestamp(end_date, tz="UTC").normalize().replace(day=1)
    months = pd.date_range(start=start, end=end, freq="MS")
    return [f"{d.year:04d}-{d.month:02d}" for d in months]


def _parse_factor_set_names(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    # support "a,b,c"
    return [x.strip() for x in str(val).split(",") if x.strip()]


def _maybe_autobuild_feature_store(cfg: Dict[str, Any], *, out_root: Path) -> None:
    """
    Optional convenience: if panel.source=feature_store and requested months are missing in the layer,
    automatically run the CS build-store builder to fill missing month partitions.
    """
    panel_cfg = cfg.get("panel", {}) or {}
    if str(panel_cfg.get("source", "")).strip().lower() != "feature_store":
        return
    fs = panel_cfg.get("feature_store", {}) or {}

    auto = panel_cfg.get("auto_build_store", None)
    enabled = False
    auto_cfg: Dict[str, Any] = {}
    if isinstance(auto, bool):
        enabled = bool(auto)
    elif isinstance(auto, dict):
        auto_cfg = auto
        enabled = bool(auto_cfg.get("enabled", False))
    if not enabled:
        return

    root = Path(str(fs.get("root", "feature_store")))
    layer = str(fs.get("layer", "")).strip()
    timeframe = str(fs.get("timeframe", "240T")).strip()
    symbols = [
        s.strip().upper() for s in str(fs.get("symbols", "")).split(",") if s.strip()
    ]
    start_date = str(fs.get("start_date", "")).strip()
    end_date = str(fs.get("end_date", "")).strip()
    if not (layer and symbols and start_date and end_date):
        return

    months = _month_keys(start_date, end_date)
    # quick missing check (filesystem only)
    missing: List[str] = []
    for sym in symbols:
        base = root / layer / sym / timeframe
        for mk in months:
            if not (base / f"{mk}.parquet").exists():
                missing.append(f"{sym}/{mk}")
                break  # one miss per symbol is enough to trigger build
    if not missing:
        return

    print(
        f"ℹ️ FeatureStore missing months detected for {len(missing)} symbols (e.g. {missing[:3]}). Auto-building..."
    )

    # Resolve desired outputs from factor_set (use factor_eval defaults if not provided)
    eval_cfg = cfg.get("factor_eval", {}) or {}
    factor_set_yaml = str(
        auto_cfg.get("factor_set_yaml") or eval_cfg.get("factor_set_yaml") or ""
    ).strip()
    factor_set_val = auto_cfg.get("factor_set") or eval_cfg.get("factor_set")
    factor_sets = _parse_factor_set_names(factor_set_val)
    if not factor_set_yaml or not factor_sets:
        raise ValueError(
            "panel.auto_build_store is enabled but factor_set_yaml/factor_set is missing. "
            "Set panel.auto_build_store.factor_set_yaml + factor_set (or provide them under factor_eval)."
        )

    from cross_sectional.feature_store_builder import (
        CSFeatureStoreBuildConfig,
        build_feature_store_for_symbols,
        load_factor_set,
    )

    desired: List[str] = []
    for name in factor_sets:
        desired.extend(
            load_factor_set(factor_set_yaml=factor_set_yaml, factor_set=name)
        )
    desired = list(dict.fromkeys(desired))

    feature_deps = str(
        auto_cfg.get("feature_deps") or "config/feature_dependencies.yaml"
    )
    data_path = str(auto_cfg.get("data_path") or "data/parquet_data")
    warmup_bars = int(auto_cfg.get("warmup_bars", 600))
    include_ohlcv = bool(auto_cfg.get("include_ohlcv", True))
    overwrite = bool(auto_cfg.get("overwrite", False))

    build_cfg = CSFeatureStoreBuildConfig(
        data_path=data_path,
        features_store_root=str(root),
        features_store_layer=layer,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        warmup_bars=warmup_bars,
        include_ohlcv=include_ohlcv,
        overwrite=overwrite,
    )
    resolved_layer = build_feature_store_for_symbols(
        symbols=symbols,
        desired_output_cols=desired,
        feature_deps_path=feature_deps,
        cfg=build_cfg,
    )
    print(f"✅ Auto build-store finished. layer={resolved_layer}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CS pipeline from a YAML config.")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument(
        "--no-docker",
        action="store_true",
        help="(reserved) kept for CLI parity; no-op here",
    )
    return p.parse_args()


def _read_yaml(path: str) -> Dict[str, Any]:
    obj = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(obj, dict):
        raise ValueError("pipeline config must be a YAML mapping")
    return obj


def _run_py(script: str, args: List[str]) -> None:
    cmd = [sys.executable, script] + args
    print("▶", " ".join(cmd))
    subprocess.check_call(cmd)


def _maybe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _write_html_index(
    *,
    out_root: Path,
    title: str,
    config: Dict[str, Any],
    panel_path: Optional[Path] = None,
    summary_csv: Path,
    selected_factors_path: Path,
    report_md_path: Optional[Path],
    train_dir: Optional[Path],
) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    html_path = out_root / "index.html"

    summary_html = "<p>(summary.csv not found)</p>"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        summary_html = df.to_html(index=False, float_format=lambda x: f"{x:.6g}")

    selected_html = "<p>(selected_factors.txt not found)</p>"
    if selected_factors_path.exists():
        lines = [
            x.strip()
            for x in selected_factors_path.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        selected_html = "<pre>" + "\n".join(lines) + "</pre>"

    report_link = ""
    if report_md_path and report_md_path.exists():
        rel = (
            report_md_path.relative_to(out_root)
            if report_md_path.is_relative_to(out_root)
            else report_md_path
        )
        report_link = f"<p><a href='{rel}'>Fama-MacBeth report (markdown)</a></p>"

    train_link = ""
    if train_dir and train_dir.exists():
        rel = (
            train_dir.relative_to(out_root)
            if train_dir.is_relative_to(out_root)
            else train_dir
        )
        train_link = f"<p>Train artifacts: <code>{rel}</code></p>"

    def _fmt(x: Any) -> str:
        try:
            if x is None:
                return "NA"
            if isinstance(x, (int, str)):
                return str(x)
            if isinstance(x, float):
                if pd.isna(x):
                    return "NA"
                return f"{x:.6g}"
            return str(x)
        except Exception:
            return str(x)

    def _details(title: str, inner_html: str, *, open_default: bool = False) -> str:
        return (
            f"<details {'open' if open_default else ''}>"
            f"<summary><b>{title}</b></summary>"
            f"{inner_html}"
            f"</details>"
        )

    def _csv_preview(path: Path, *, nrows: int = 200) -> str:
        if not path.exists():
            return "<p>(missing)</p>"
        try:
            df = pd.read_csv(path, nrows=int(nrows))
            return df.to_html(index=False, float_format=lambda x: f"{x:.6g}")
        except Exception as e:
            return f"<p>(failed to read csv: {e})</p>"

    def _bullet_list(items: List[str]) -> str:
        if not items:
            return "<p>(no conclusion)</p>"
        li = "".join([f"<li>{x}</li>" for x in items])
        return f"<ul>{li}</ul>"

    # -----------------------------------------------------------------------------
    # Conclusions helpers (derived from outputs)
    # -----------------------------------------------------------------------------
    eval_cfg = config.get("factor_eval", {}) or {}
    sel_cfg = config.get("select", {}) or {}
    tr_cfg = config.get("train", {}) or {}
    bt_cfg = tr_cfg.get("backtest_cfg", {}) or {}
    wf_cfg = tr_cfg.get("walk_forward", {}) or {}

    # panel time range (prefer actual timestamps from panel file; fallback to YAML dates)
    panel_range_html = ""
    try:
        ts_min = None
        ts_max = None
        if panel_path and panel_path.exists():
            if panel_path.suffix.lower() in [".parquet", ".pq"]:
                try:
                    df_ts = pd.read_parquet(panel_path, columns=["timestamp"])
                except Exception:
                    df_ts = pd.read_parquet(panel_path)
                if "timestamp" in df_ts.columns:
                    s = pd.to_datetime(
                        df_ts["timestamp"], utc=True, errors="coerce"
                    ).dropna()
                    if not s.empty:
                        ts_min = s.min()
                        ts_max = s.max()
            else:
                df_ts = pd.read_csv(panel_path, usecols=["timestamp"])
                s = pd.to_datetime(
                    df_ts["timestamp"], utc=True, errors="coerce"
                ).dropna()
                if not s.empty:
                    ts_min = s.min()
                    ts_max = s.max()

        if ts_min is None or ts_max is None:
            # fallback to config dates if provided (feature_store)
            fs = (config.get("panel", {}) or {}).get("feature_store", {}) or {}
            sd = fs.get("start_date")
            ed = fs.get("end_date")
            if sd and ed:
                ts_min = str(sd)
                ts_max = str(ed)
        if ts_min is not None and ts_max is not None:
            panel_range_html = f"<li><b>Panel range</b>: <code>{ts_min}</code> → <code>{ts_max}</code></li>"
    except Exception:
        panel_range_html = ""

    # OOS backtest time range (from aggregated timeseries CSVs)
    oos_range_items: List[str] = []
    try:
        if train_dir and train_dir.exists():
            for mode in ["long_only", "market_neutral"]:
                fp = train_dir / f"model_bt_timeseries__{mode}.csv"
                if not fp.exists():
                    continue
                df = pd.read_csv(fp)
                # Prefer "timestamp" column, else "index" (older variants)
                col = (
                    "timestamp"
                    if "timestamp" in df.columns
                    else (df.columns[0] if len(df.columns) else None)
                )
                if not col:
                    continue
                s = pd.to_datetime(df[col], utc=True, errors="coerce").dropna()
                if s.empty:
                    continue
                oos_range_items.append(
                    f"<li><b>OOS backtest range</b> (<code>{mode}</code>): <code>{s.min()}</code> → <code>{s.max()}</code> "
                    f"(n_periods={len(s)})</li>"
                )
    except Exception:
        oos_range_items = []

    test_window_note = ""
    if panel_range_html and oos_range_items:
        test_window_note = (
            "<p><b>Why these differ?</b> Panel range is the available data snapshot. "
            "OOS backtest range is only the out-of-sample test segments after train split / walk-forward embargo, "
            "and it is further thinned by rebalance frequency (holding_period_bars) and label/return alignment. "
            "To extend OOS: increase data end_date (and ensure FeatureStore months exist), reduce train_ratio, "
            "or reduce holding_period_bars (more frequent rebalances), and ensure min_assets filter doesn’t drop early timestamps.</p>"
        )

    test_window_html = ""
    if panel_range_html or oos_range_items:
        test_window_html = (
            "<h2>Test window</h2><ul>"
            + (panel_range_html if panel_range_html else "")
            + "".join(oos_range_items)
            + "</ul>"
            + test_window_note
        )

    # selection summary (if exists)
    selection_summary = out_root / "selection_summary.json"
    sj: Dict[str, Any] = {}
    if selection_summary.exists():
        try:
            sj = json.loads(selection_summary.read_text(encoding="utf-8"))
        except Exception:
            sj = {}

    # selected factors
    selected_factors: List[str] = []
    if selected_factors_path.exists():
        selected_factors = [
            x.strip()
            for x in selected_factors_path.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]

    # walk-forward summary
    wf_rows: List[Dict[str, Any]] = []
    if train_dir and train_dir.exists():
        wf_sum = train_dir / "walk_forward_summary.json"
        if wf_sum.exists():
            try:
                wf_rows = json.loads(wf_sum.read_text(encoding="utf-8"))
                if not isinstance(wf_rows, list):
                    wf_rows = []
            except Exception:
                wf_rows = []

    # model backtest metrics
    bt_conc: List[str] = []
    bt_table_html = ""
    bt_links_html = ""
    if train_dir and train_dir.exists():

        def _sharpe_grade(sh: float) -> str:
            # Heuristic buckets for Sharpe(net) on OOS backtests
            if sh >= 2.0:
                return "EXCELLENT / 很强"
            if sh >= 1.0:
                return "GOOD / 不错"
            if sh >= 0.0:
                return "MARGINAL / 一般"
            return "BAD / 差"

        rows = []
        for mode in ["long_only", "market_neutral"]:
            met_m = train_dir / f"model_bt_metrics__{mode}.json"
            met_f = train_dir / f"factor_combo_bt_metrics__{mode}.json"
            if not met_m.exists() and not met_f.exists():
                continue
            try:
                m = (
                    json.loads(met_m.read_text(encoding="utf-8"))
                    if met_m.exists()
                    else {}
                )
                f = (
                    json.loads(met_f.read_text(encoding="utf-8"))
                    if met_f.exists()
                    else {}
                )
                sharpe_m = m.get("sharpe_net")
                sharpe_f = f.get("sharpe_net")
                rows.append(
                    {
                        "mode": mode,
                        "sharpe_net_model": sharpe_m,
                        "sharpe_net_factor_combo": sharpe_f,
                        "avg_turnover_model": m.get("avg_turnover"),
                        "avg_turnover_factor_combo": f.get("avg_turnover"),
                        "fee_bps": m.get("fee_bps"),
                        "slippage_bps": m.get("slippage_bps"),
                        "total_return_net_model": m.get("total_return_net"),
                        "total_return_net_factor_combo": f.get("total_return_net"),
                        "max_drawdown_model": m.get("max_drawdown"),
                        "max_drawdown_factor_combo": f.get("max_drawdown"),
                        "n_timestamps": m.get("n_timestamps") or f.get("n_timestamps"),
                        "periods_per_year": m.get("periods_per_year")
                        or f.get("periods_per_year"),
                    }
                )
            except Exception:
                continue

        if rows:
            df_bt = pd.DataFrame(rows)
            # Always show sample size caveat if available
            try:
                n_ts = df_bt["n_timestamps"].dropna().astype(float)
                if not n_ts.empty:
                    bt_conc.append(
                        f"<b>Sample size</b>: n_timestamps={_fmt(float(n_ts.iloc[0]))} "
                        f"(this is the number of OOS rebalance periods after holding/lag; roughly ≈ total_OOS_bars / holding_period_bars. "
                        f"Small = conclusions are preliminary; prefer 200+ by extending the date range or reducing holding_period_bars)."
                    )
            except Exception:
                pass
            # conclusions: pick best mode by model sharpe, compare vs factor combo
            best_mode = None
            try:
                tmp = df_bt.dropna(subset=["sharpe_net_model"]).copy()
                if not tmp.empty:
                    best_row = tmp.iloc[tmp["sharpe_net_model"].astype(float).idxmax()]
                    best_mode = str(best_row["mode"])
                    bt_conc.append(
                        f"<b>Best mode (model)</b>: <code>{best_mode}</code> with Sharpe(net)={_fmt(best_row['sharpe_net_model'])}."
                    )
            except Exception:
                pass

            # per mode compare
            for _, r in df_bt.iterrows():
                mode = str(r.get("mode"))
                sm = r.get("sharpe_net_model")
                sf = r.get("sharpe_net_factor_combo")
                if sm is None or pd.isna(sm) or sf is None or pd.isna(sf):
                    continue
                diff = float(sm) - float(sf)
                grade = _sharpe_grade(float(sm))
                ddm = r.get("max_drawdown_model")
                trm = r.get("total_return_net_model")
                bt_conc.append(
                    f"<code>{mode}</code>: <b>Sharpe(net)</b>={_fmt(sm)} → <b>{grade}</b>; "
                    f"total_return_net={_fmt(trm)}, max_drawdown={_fmt(ddm)}; "
                    f"vs factor-combo Sharpe(net)={_fmt(sf)} (Δ={_fmt(diff)})."
                )

            bt_table_html = df_bt.to_html(
                index=False, float_format=lambda x: f"{x:.6g}"
            )

            # Add links to detailed trade/audit artifacts if present
            link_rows: List[str] = []
            for mode in ["long_only", "market_neutral"]:
                rep = train_dir / f"backtest_report__{mode}.html"
                m_tr = train_dir / f"model_trades__{mode}.csv"
                f_tr = train_dir / f"factor_combo_trades__{mode}.csv"
                m_rb = train_dir / f"model_rebalance_log__{mode}.csv"
                f_rb = train_dir / f"factor_combo_rebalance_log__{mode}.csv"
                if not (
                    rep.exists()
                    or m_tr.exists()
                    or f_tr.exists()
                    or m_rb.exists()
                    or f_rb.exists()
                ):
                    continue
                parts: List[str] = []
                if rep.exists():
                    parts.append(f"<a href='train/{rep.name}'>detail report</a>")
                if m_tr.exists():
                    parts.append(f"<a href='train/{m_tr.name}'>model trades</a>")
                if f_tr.exists():
                    parts.append(f"<a href='train/{f_tr.name}'>factor-combo trades</a>")
                if m_rb.exists():
                    parts.append(f"<a href='train/{m_rb.name}'>model rebalance log</a>")
                if f_rb.exists():
                    parts.append(
                        f"<a href='train/{f_rb.name}'>factor rebalance log</a>"
                    )
                link_rows.append(
                    f"<li><code>{mode}</code>: " + " / ".join(parts) + "</li>"
                )
            if link_rows:
                # Inline previews (avoid download-only UX)
                preview_blocks: List[str] = []
                for mode in ["long_only", "market_neutral"]:
                    rep = train_dir / f"backtest_report__{mode}.html"
                    m_tr = train_dir / f"model_trades__{mode}.csv"
                    f_tr = train_dir / f"factor_combo_trades__{mode}.csv"
                    m_rb = train_dir / f"model_rebalance_log__{mode}.csv"
                    f_rb = train_dir / f"factor_combo_rebalance_log__{mode}.csv"
                    if not (
                        m_rb.exists()
                        or f_rb.exists()
                        or m_tr.exists()
                        or f_tr.exists()
                        or rep.exists()
                    ):
                        continue

                    inner = ""
                    if rep.exists():
                        inner += f"<p><a href='train/{rep.name}'>Open detail report (HTML)</a></p>"
                    if m_rb.exists():
                        inner += _details(
                            f"{mode} / model rebalance log (preview)",
                            _csv_preview(m_rb, nrows=200),
                            open_default=False,
                        )
                    if f_rb.exists():
                        inner += _details(
                            f"{mode} / factor rebalance log (preview)",
                            _csv_preview(f_rb, nrows=200),
                            open_default=False,
                        )
                    if m_tr.exists():
                        inner += _details(
                            f"{mode} / model trades (preview)",
                            _csv_preview(m_tr, nrows=200),
                            open_default=False,
                        )
                    if f_tr.exists():
                        inner += _details(
                            f"{mode} / factor-combo trades (preview)",
                            _csv_preview(f_tr, nrows=200),
                            open_default=False,
                        )
                    preview_blocks.append(
                        _details(
                            f"{mode}: view logs & trades (inline)",
                            inner,
                            open_default=False,
                        )
                    )

                bt_links_html = (
                    "<h3>Detailed trades & audit (inline preview)</h3>"
                    + "".join(preview_blocks)
                    + "<p>Direct file links (fallback):</p><ul>"
                    + "".join(link_rows)
                    + "</ul>"
                )

    # factor eval summary df
    fe_conc: List[str] = []
    fe_table_html = summary_html
    if summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            dfv = df.copy()
            # filter to valid numeric rows
            for col in ["ic_mean", "ic_ir", "sharpe_net", "avg_turnover", "ic_count"]:
                if col in dfv.columns:
                    dfv[col] = pd.to_numeric(dfv[col], errors="coerce")
            df_ok = dfv[dfv.get("error").isna()] if "error" in dfv.columns else dfv
            df_ok = (
                df_ok.dropna(subset=["ic_mean"], how="any")
                if "ic_mean" in df_ok.columns
                else df_ok
            )
            n_all = len(dfv)
            n_ok = len(df_ok)
            if n_all:
                fe_conc.append(
                    f"<b>Evaluated</b>: {n_ok}/{n_all} factors produced valid stats."
                )
            if "ic_mean" in df_ok.columns and not df_ok.empty:
                n_pos = int((df_ok["ic_mean"] > 0).sum())
                fe_conc.append(
                    f"<b>IC sign</b>: {n_pos}/{n_ok} factors have positive IC_mean."
                )
            # top by ic_ir and sharpe_net
            if "ic_ir" in df_ok.columns and "factor" in df_ok.columns:
                top_ir = df_ok.sort_values("ic_ir", ascending=False).head(5)
                tops = ", ".join(
                    [
                        f"<code>{r['factor']}</code>({_fmt(r['ic_ir'])})"
                        for _, r in top_ir.iterrows()
                    ]
                )
                fe_conc.append(f"<b>Top IC/IR</b>: {tops}.")
            if "sharpe_net" in df_ok.columns and "factor" in df_ok.columns:
                top_sh = df_ok.sort_values("sharpe_net", ascending=False).head(5)
                tops = ", ".join(
                    [
                        f"<code>{r['factor']}</code>({_fmt(r['sharpe_net'])})"
                        for _, r in top_sh.iterrows()
                    ]
                )
                fe_conc.append(f"<b>Top Sharpe(net)</b>: {tops}.")
            if "avg_turnover" in df_ok.columns and not df_ok.empty:
                med_to = float(df_ok["avg_turnover"].median())
                fe_conc.append(
                    f"<b>Turnover (median)</b>: {_fmt(med_to)} per rebalance."
                )
        except Exception:
            pass

    # selection output conclusions
    sel_conc: List[str] = []
    if selected_factors:
        sel_conc.append(f"<b>Selected</b>: {len(selected_factors)} factors.")
        sel_conc.append(
            f"<b>List</b>: "
            + ", ".join([f"<code>{x}</code>" for x in selected_factors[:30]])
            + ("..." if len(selected_factors) > 30 else "")
            + "."
        )
    else:
        sel_conc.append(
            "<b>Selected</b>: 0 factors (check thresholds / min_assets / factor pool)."
        )
    if sj:
        sel_conc.append(
            "Criteria: "
            + ", ".join(
                [
                    f"target=<code>{_fmt(sj.get('target'))}</code>",
                    f"min_assets={_fmt(sj.get('min_assets'))}",
                    f"per_category_top={_fmt(sj.get('per_category_top'))}",
                    f"global_top={_fmt(sj.get('global_top'))}",
                    f"ranking_stat=<code>{_fmt(sj.get('ranking_stat'))}</code>",
                    f"ic_threshold={_fmt(sj.get('ic_threshold'))}",
                ]
            )
            + "."
        )

    # walk-forward conclusions
    wf_conc: List[str] = []
    wf_table_html = ""
    if wf_rows:
        try:
            dff = pd.DataFrame(wf_rows)
            for col in ["ic_mean", "rank_ic_mean"]:
                if col in dff.columns:
                    dff[col] = pd.to_numeric(dff[col], errors="coerce")
            if "ic_mean" in dff.columns and dff["ic_mean"].notna().any():
                mean_ic = float(dff["ic_mean"].mean())
                std_ic = float(dff["ic_mean"].std(ddof=0))
                n_folds = int(len(dff))
                n_pos = int((dff["ic_mean"] > 0).sum())
                min_ic = float(dff["ic_mean"].min())
                max_ic = float(dff["ic_mean"].max())
                cv = (std_ic / abs(mean_ic)) if mean_ic != 0 else float("inf")

                # Simple, explicit verdict for humans (heuristics)
                # Typical CS IC: ~0.01-0.03 is weak, 0.03-0.06 is decent, >0.06 is strong.
                # Stability: positive across folds and not dominated by one fold; min fold shouldn't collapse.
                verdict = "WEAK"
                verdict_cn = "偏弱"
                if mean_ic >= 0.06 and n_pos == n_folds and min_ic >= 0.01:
                    verdict = "STRONG"
                    verdict_cn = "很好（强且稳定）"
                elif (
                    mean_ic >= 0.03
                    and n_pos >= max(1, int(0.75 * n_folds))
                    and min_ic >= 0.0
                ):
                    verdict = "GOOD"
                    verdict_cn = "不错（有一致性）"
                elif mean_ic >= 0.01 and n_pos >= max(1, int(0.6 * n_folds)):
                    verdict = "MARGINAL"
                    verdict_cn = "一般（有信号但偏弱）"
                else:
                    verdict = "WEAK"
                    verdict_cn = "偏弱/不稳定"

                # fold coverage note (if configured folds != realized folds)
                cfg_folds = wf_cfg.get("folds")
                if cfg_folds is not None:
                    try:
                        cfg_folds_i = int(cfg_folds)
                        if cfg_folds_i > 0 and n_folds != cfg_folds_i:
                            wf_conc.append(
                                f"<b>Fold coverage</b>: got {n_folds}/{cfg_folds_i} folds (missing folds usually means not enough usable timestamps after embargo/filters)."
                            )
                    except Exception:
                        pass

                wf_conc.append(
                    f"<b>Verdict</b>: <code>{verdict}</code> / {verdict_cn}. "
                    f"IC_mean={_fmt(mean_ic)} (typical: 0.01–0.03 weak, 0.03–0.06 ok, >0.06 strong), "
                    f"min={_fmt(min_ic)}, max={_fmt(max_ic)}, CV(std/|mean|)={_fmt(cv)}."
                )

                wf_conc.append(
                    f"<b>Walk-forward IC_mean</b>: mean={_fmt(mean_ic)}, std={_fmt(std_ic)} over {n_folds} folds."
                )
                wf_conc.append(
                    f"<b>Sign stability</b>: {n_pos}/{n_folds} folds have positive IC_mean."
                )
                try:
                    best = dff.iloc[dff["ic_mean"].idxmax()]
                    worst = dff.iloc[dff["ic_mean"].idxmin()]
                    wf_conc.append(
                        f"<b>Best fold</b>: fold={_fmt(best.get('fold'))}, IC_mean={_fmt(best.get('ic_mean'))}."
                    )
                    wf_conc.append(
                        f"<b>Worst fold</b>: fold={_fmt(worst.get('fold'))}, IC_mean={_fmt(worst.get('ic_mean'))}."
                    )
                except Exception:
                    pass
            wf_table_html = dff.to_html(index=False, float_format=lambda x: f"{x:.6g}")
        except Exception:
            wf_table_html = ""
    else:
        wf_conc.append("<b>Walk-forward</b>: (no fold summary found).")

    # -----------------------------------------------------------------------------
    # Separate interpretation page (keep the long “how to read” content off index.html)
    # -----------------------------------------------------------------------------
    interpretation_path = out_root / "interpretation.html"
    interpretation_body = f"""
    <h1>{title}: Interpretation Guide</h1>
    <p><a href="index.html">← Back to conclusions</a></p>

    <h2>How to read this report</h2>
    <ul>
      <li><b>Factor evaluation summary</b> answers: does a single factor work as a CS long/short signal?</li>
      <li><b>Selected factors</b> is produced by <code>select</code> for <i>this</i> window/universe/cost settings (a run artifact).</li>
      <li><b>OOS backtest</b> compares <code>model</code> vs <code>factor-combo</code> under the same execution assumptions.</li>
    </ul>

    <h3>What is factor-combo?</h3>
    <p>
      A simple baseline: for factors in <code>selected_factors.txt</code>, we z-score each factor cross-sectionally per timestamp and average into one signal.
      It answers whether the model adds value beyond a linear multi-factor blend.
    </p>

    <h3>Selection criteria (this run)</h3>
    <ul>
      <li><b>target</b>: <code>{_fmt(sj.get('target') or sel_cfg.get('target') or eval_cfg.get('target'))}</code></li>
      <li><b>min_assets</b>: {_fmt(sj.get('min_assets') or sel_cfg.get('min_assets') or eval_cfg.get('min_assets'))}</li>
      <li><b>per_category_top</b>: {_fmt(sj.get('per_category_top') or sel_cfg.get('per_category_top'))}</li>
      <li><b>global_top</b>: {_fmt(sj.get('global_top') or sel_cfg.get('global_top'))}</li>
      <li><b>ranking_stat</b>: <code>{_fmt(sj.get('ranking_stat') or sel_cfg.get('ranking_stat') or 'ic')}</code></li>
      <li><b>ic_threshold</b>: {_fmt(sj.get('ic_threshold') or sel_cfg.get('ic_threshold'))}</li>
      <li><b>ir_threshold</b>: {_fmt(sj.get('ir_threshold') or sel_cfg.get('ir_threshold'))}</li>
    </ul>

    <h3>Execution assumptions (this run)</h3>
    <ul>
      <li><b>factor_eval</b>: horizon={_fmt(eval_cfg.get('horizon'))}, target=<code>{_fmt(eval_cfg.get('target'))}</code>, min_assets={_fmt(eval_cfg.get('min_assets'))}, quantiles={_fmt(eval_cfg.get('quantiles'))}, fee_bps={_fmt(eval_cfg.get('fee_bps'))}</li>
      <li><b>train.backtest_cfg</b>: holding={_fmt(bt_cfg.get('holding_period_bars'))}, lag={_fmt(bt_cfg.get('execution_lag_bars'))}, mode=<code>{_fmt(bt_cfg.get('mode'))}</code>, top_k={_fmt(bt_cfg.get('top_k'))}, bottom_k={_fmt(bt_cfg.get('bottom_k'))}, max_weight={_fmt(bt_cfg.get('max_weight'))}, cash_buffer={_fmt(bt_cfg.get('cash_buffer'))}, fee_bps={_fmt(bt_cfg.get('fee_bps'))}, slippage_bps={_fmt(bt_cfg.get('slippage_bps'))}</li>
      <li><b>walk_forward</b>: folds={_fmt(wf_cfg.get('folds'))}, embargo_bars={_fmt(wf_cfg.get('embargo_bars'))}</li>
    </ul>
    """

    interpretation_html = f"""\
<html>
  <head>
    <meta charset="utf-8"/>
    <title>{title} - Interpretation</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif; padding: 20px; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; }}
      th {{ background: #f6f6f6; position: sticky; top: 0; }}
      pre {{ background: #f6f8fa; padding: 12px; overflow-x: auto; }}
      code {{ background: #f6f8fa; padding: 2px 4px; }}
    </style>
  </head>
  <body>
    {interpretation_body}
  </body>
</html>
"""
    interpretation_path.write_text(interpretation_html, encoding="utf-8")

    # -----------------------------------------------------------------------------
    # Index page: conclusions per table + raw tables collapsed
    # -----------------------------------------------------------------------------
    link_interp = (
        "<p><a href='interpretation.html'>Interpretation (how to read)</a></p>"
    )

    oos_section = (
        "<h2>OOS backtest (model vs factor-combo) — conclusion</h2>"
        + _bullet_list(bt_conc)
        + (bt_links_html if bt_links_html else "")
        + (
            _details("Raw table", bt_table_html)
            if bt_table_html
            else "<p>(no backtest table)</p>"
        )
    )

    wf_section = (
        "<h2>Walk-forward stability — conclusion</h2>"
        + _bullet_list(wf_conc)
        + (_details("Raw table", wf_table_html) if wf_table_html else "")
    )

    sel_section = (
        "<h2>Selected factors — conclusion</h2>"
        + _bullet_list(sel_conc)
        + _details("Raw list", selected_html)
    )

    fe_section = (
        "<h2>Factor evaluation summary — conclusion</h2>"
        + _bullet_list(fe_conc)
        + _details("Raw table", fe_table_html)
    )

    html = f"""\
<html>
  <head>
    <meta charset="utf-8"/>
    <title>{title}</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif; padding: 20px; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; }}
      th {{ background: #f6f6f6; position: sticky; top: 0; }}
      pre {{ background: #f6f8fa; padding: 12px; overflow-x: auto; }}
      code {{ background: #f6f8fa; padding: 2px 4px; }}
      summary {{ cursor: pointer; }}
    </style>
  </head>
  <body>
    <h1>{title}</h1>
    <p>Output root: <code>{out_root}</code></p>
    {report_link}
    {train_link}
    {link_interp}
    {test_window_html}
    {oos_section}
    {wf_section}
    {sel_section}
    {fe_section}
  </body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main() -> None:
    args = _parse_args()
    cfg = _read_yaml(args.config)

    out_root = Path(cfg.get("output_root", "results/cross_sectional/pipeline"))
    out_root.mkdir(parents=True, exist_ok=True)

    # Optional: auto build FeatureStore partitions if missing
    _maybe_autobuild_feature_store(cfg, out_root=out_root)

    # 1) Panel source
    panel_cfg = cfg.get("panel", {}) or {}
    source = str(panel_cfg.get("source", "parquet")).strip().lower()
    panel_path = panel_cfg.get("path")

    if source == "parquet":
        if not panel_path:
            raise ValueError("panel.source=parquet requires panel.path")
        panel_path = str(panel_path)
    elif source == "feature_store":
        # Build and persist a panel parquet for reproducibility and for downstream `select/train`.
        fs = panel_cfg.get("feature_store", {}) or {}
        required = ["layer", "symbols", "start_date", "end_date"]
        for k in required:
            if not fs.get(k):
                raise ValueError(
                    f"panel.feature_store.{k} is required when source=feature_store"
                )
        from cross_sectional.feature_store_panel import (
            FeatureStorePanelConfig,
            load_feature_store_frames,
        )

        fs_cfg = FeatureStorePanelConfig(
            root=str(fs.get("root", "feature_store")),
            layer=str(fs["layer"]),
            timeframe=str(fs.get("timeframe", "240T")),
            timestamp_col="timestamp",
            symbol_col="symbol",
        )
        df = load_feature_store_frames(
            symbols=[
                s.strip().upper() for s in str(fs["symbols"]).split(",") if s.strip()
            ],
            cfg=fs_cfg,
            start_date=str(fs["start_date"]),
            end_date=str(fs["end_date"]),
            columns=None,
        )
        # Pre-compute a forward-return target for downstream selection/training.
        # This matches the convention used across CS scripts: future_return_<horizon>.
        eval_cfg = cfg.get("factor_eval", {}) or {}
        horizon = int(eval_cfg.get("horizon", 12))
        target_col = f"future_return_{horizon}"
        if "close" in df.columns and target_col not in df.columns:
            df = df.sort_values(["symbol", "timestamp"]).copy()
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            future = df.groupby("symbol")["close"].shift(-horizon)
            df[target_col] = (future - df["close"]) / df["close"]

        panel_path = str(out_root / "panel_from_feature_store.parquet")
        Path(panel_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(panel_path, index=False)
    else:
        raise ValueError("panel.source must be one of: parquet, feature_store")

    # 2) Factor eval
    eval_cfg = cfg.get("factor_eval", {}) or {}
    eval_out = out_root / "factor_eval"
    fe_args: List[str] = ["--output-dir", str(eval_out)]

    # factor list sources
    if eval_cfg.get("factors"):
        fe_args += ["--factors", str(eval_cfg["factors"])]
    if eval_cfg.get("factors_file"):
        fe_args += ["--factors-file", str(eval_cfg["factors_file"])]
    if eval_cfg.get("factor_set_yaml"):
        fe_args += ["--factor-set-yaml", str(eval_cfg["factor_set_yaml"])]
    if eval_cfg.get("factor_set"):
        fs_val = eval_cfg.get("factor_set")
        if isinstance(fs_val, list):
            # allow YAML list: [setA, setB] -> "setA,setB"
            fs_val = ",".join([str(x).strip() for x in fs_val if str(x).strip()])
        fe_args += ["--factor-set", str(fs_val)]

    if panel_path:
        fe_args += ["--input", str(panel_path)]
    else:
        fs = panel_cfg.get("feature_store", {}) or {}
        required = ["layer", "symbols", "start_date", "end_date"]
        for k in required:
            if not fs.get(k):
                raise ValueError(
                    f"panel.feature_store.{k} is required when source=feature_store"
                )
        fe_args += [
            "--features-store-root",
            str(fs.get("root", "feature_store")),
            "--features-store-layer",
            str(fs["layer"]),
            "--timeframe",
            str(fs.get("timeframe", "240T")),
            "--symbols",
            str(fs["symbols"]),
            "--start-date",
            str(fs["start_date"]),
            "--end-date",
            str(fs["end_date"]),
        ]
        if fs.get("columns"):
            fe_args += ["--columns", str(fs["columns"])]

    if eval_cfg.get("horizon") is not None:
        fe_args += ["--horizon", str(int(eval_cfg["horizon"]))]
    if eval_cfg.get("target"):
        fe_args += ["--target", str(eval_cfg["target"])]
    if eval_cfg.get("min_assets") is not None:
        fe_args += ["--min-assets", str(int(eval_cfg["min_assets"]))]
    if eval_cfg.get("quantiles") is not None:
        fe_args += ["--quantiles", str(int(eval_cfg["quantiles"]))]
    if eval_cfg.get("fee_bps") is not None:
        fe_args += ["--fee-bps", str(float(eval_cfg["fee_bps"]))]

    _run_py("src/cross_sectional/scripts/factor_eval.py", fe_args)

    # 3) Factor select (optional)
    select_cfg = cfg.get("select", {}) or {}
    selected_path = out_root / "selected_factors.txt"
    selection_json = out_root / "selection_summary.json"
    if bool(select_cfg.get("enabled", True)):
        if panel_path:
            sel_args = [
                "--input",
                str(panel_path),
                "--output",
                str(selected_path),
                "--output-json",
                str(selection_json),
            ]
            if select_cfg.get("target"):
                sel_args += ["--target", str(select_cfg["target"])]
            if select_cfg.get("min_assets") is not None:
                sel_args += ["--min-assets", str(int(select_cfg["min_assets"]))]
            if select_cfg.get("per_category_top") is not None:
                sel_args += [
                    "--per-category-top",
                    str(int(select_cfg["per_category_top"])),
                ]
            if select_cfg.get("global_top") is not None:
                sel_args += ["--global-top", str(int(select_cfg["global_top"]))]
            if select_cfg.get("ic_threshold") is not None:
                sel_args += ["--ic-threshold", str(float(select_cfg["ic_threshold"]))]
            if select_cfg.get("ir_threshold") is not None:
                sel_args += ["--ir-threshold", str(float(select_cfg["ir_threshold"]))]
            if select_cfg.get("ranking_stat"):
                sel_args += ["--ranking-stat", str(select_cfg["ranking_stat"])]
            if select_cfg.get("include_categories"):
                # argparse expects nargs="*"
                cats = select_cfg.get("include_categories")
                if isinstance(cats, str):
                    cats = [c.strip() for c in cats.split(",") if c.strip()]
                sel_args += ["--include-categories"] + list(cats or [])

            _run_py("src/cross_sectional/scripts/auto_select_factors.py", sel_args)
    else:
        print("ℹ️  Selection disabled by config.")

    # 3.5) Optional: report (Fama-MacBeth markdown)
    report_cfg = cfg.get("report", {}) or {}
    report_md: Optional[Path] = None
    if bool(report_cfg.get("enabled", False)):
        report_md = out_root / str(report_cfg.get("output", "fama_macbeth_report.md"))
        report_args: List[str] = [
            "--input",
            str(panel_path),
            "--output",
            str(report_md),
        ]
        # prefer selected factors if available, else allow explicit feature_file/cols
        if selected_path.exists():
            report_args += ["--feature-file", str(selected_path)]
        elif report_cfg.get("feature_file"):
            report_args += ["--feature-file", str(report_cfg["feature_file"])]
        elif report_cfg.get("feature_cols"):
            report_args += ["--feature-cols", str(report_cfg["feature_cols"])]
        if report_cfg.get("symbols"):
            report_args += ["--symbols", str(report_cfg["symbols"])]
        if report_cfg.get("horizon") is not None:
            report_args += ["--horizon", str(int(report_cfg["horizon"]))]
        if report_cfg.get("winsor") is not None:
            report_args += ["--winsor", str(float(report_cfg["winsor"]))]
        if report_cfg.get("max_lag") is not None:
            report_args += ["--max-lag", str(int(report_cfg["max_lag"]))]
        if report_cfg.get("periods_per_year") is not None:
            report_args += ["--periods-per-year", str(report_cfg["periods_per_year"])]
        if report_cfg.get("zscore") is False:
            report_args += ["--no-zscore"]
        if report_cfg.get("skip_na_drop") is True:
            report_args += ["--skip-na-drop"]
        if report_cfg.get("crypto_factors") is False:
            report_args += ["--no-crypto-factors"]
        try:
            _run_py("src/cross_sectional/scripts/run_famacbeth_report.py", report_args)
        except subprocess.CalledProcessError as e:
            # Keep the pipeline usable even if Fama-MacBeth fails on sparse panels.
            print(f"⚠️  Report step failed (continuing): {e}")
            report_md = None

    # 3.6) Optional: train
    train_cfg = cfg.get("train", {}) or {}
    train_dir: Optional[Path] = None
    if bool(train_cfg.get("enabled", False)):
        train_dir = out_root / str(train_cfg.get("output_dir", "train"))
        train_args: List[str] = [
            "--input",
            str(panel_path),
            "--output-dir",
            str(train_dir),
        ]
        if selected_path.exists():
            train_args += ["--feature-file", str(selected_path)]
        elif train_cfg.get("feature_file"):
            train_args += ["--feature-file", str(train_cfg["feature_file"])]
        elif train_cfg.get("feature_cols"):
            train_args += ["--feature-cols", str(train_cfg["feature_cols"])]
        if train_cfg.get("symbols"):
            train_args += ["--symbols", str(train_cfg["symbols"])]
        if train_cfg.get("horizon") is not None:
            train_args += ["--horizon", str(int(train_cfg["horizon"]))]
        if train_cfg.get("model"):
            train_args += ["--model", str(train_cfg["model"])]
        if train_cfg.get("winsor") is not None:
            train_args += ["--winsor", str(float(train_cfg["winsor"]))]
        if train_cfg.get("zscore") is False:
            train_args += ["--no-zscore"]
        if train_cfg.get("crypto_factors") is False:
            train_args += ["--no-crypto-factors"]
        if train_cfg.get("periods_per_year") is not None:
            train_args += ["--periods-per-year", str(train_cfg["periods_per_year"])]
        # Backtest settings for model predictions
        if train_cfg.get("backtest") is False:
            train_args += ["--no-backtest"]
        split_cfg = train_cfg.get("split", {}) or {}
        if split_cfg.get("train_ratio") is not None:
            train_args += ["--train-ratio", str(float(split_cfg["train_ratio"]))]

        wf_cfg = train_cfg.get("walk_forward", {}) or {}
        if wf_cfg.get("folds") is not None:
            train_args += ["--wf-folds", str(int(wf_cfg["folds"]))]
        if wf_cfg.get("embargo_bars") is not None:
            train_args += ["--wf-embargo-bars", str(int(wf_cfg["embargo_bars"]))]

        bt_cfg = train_cfg.get("backtest_cfg", {}) or {}
        if bt_cfg.get("mode"):
            train_args += ["--bt-mode", str(bt_cfg["mode"])]
        if bt_cfg.get("holding_period_bars") is not None:
            train_args += ["--bt-holding", str(int(bt_cfg["holding_period_bars"]))]
        if bt_cfg.get("execution_lag_bars") is not None:
            train_args += ["--bt-lag", str(int(bt_cfg["execution_lag_bars"]))]
        if bt_cfg.get("top_k") is not None:
            train_args += ["--bt-topk", str(int(bt_cfg["top_k"]))]
        if bt_cfg.get("bottom_k") is not None:
            train_args += ["--bt-bottomk", str(int(bt_cfg["bottom_k"]))]
        if bt_cfg.get("gross_leverage") is not None:
            train_args += ["--bt-gross-leverage", str(float(bt_cfg["gross_leverage"]))]
        if bt_cfg.get("max_weight") is not None:
            train_args += ["--bt-max-weight", str(float(bt_cfg["max_weight"]))]
        if bt_cfg.get("turnover_limit") is not None:
            train_args += ["--bt-turnover-limit", str(float(bt_cfg["turnover_limit"]))]
        if bt_cfg.get("cash_buffer") is not None:
            train_args += ["--bt-cash-buffer", str(float(bt_cfg["cash_buffer"]))]
        if bt_cfg.get("equity_mode") is not None:
            train_args += ["--bt-equity-mode", str(bt_cfg["equity_mode"])]
        bt_fee = bt_cfg.get("fee_bps")
        if bt_fee is None:
            bt_fee = (cfg.get("factor_eval", {}) or {}).get("fee_bps")
        if bt_fee is not None:
            train_args += ["--bt-fee-bps", str(float(bt_fee))]
        bt_slip = bt_cfg.get("slippage_bps")
        if bt_slip is not None:
            train_args += ["--bt-slippage-bps", str(float(bt_slip))]
        bt_funding = bt_cfg.get("funding_bps_per_bar")
        if bt_funding is not None:
            train_args += ["--bt-funding-bps-per-bar", str(float(bt_funding))]
        bt_borrow = bt_cfg.get("borrow_bps_per_bar")
        if bt_borrow is not None:
            train_args += ["--bt-borrow-bps-per-bar", str(float(bt_borrow))]
        bt_min = bt_cfg.get("min_assets")
        if bt_min is None:
            bt_min = (cfg.get("factor_eval", {}) or {}).get("min_assets")
        if bt_min is not None:
            train_args += ["--bt-min-assets", str(int(bt_min))]
        try:
            _run_py(
                "src/cross_sectional/scripts/train_cross_sectional_model.py", train_args
            )
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Train step failed (continuing): {e}")
            train_dir = None

    # 3.7) HTML index report
    summary_csv = eval_out / "summary.csv"
    # Best-effort fallback: if train step failed/was skipped but artifacts exist from a previous run,
    # still include them in the HTML report to avoid "(no backtest table)".
    if train_dir is None:
        candidate = out_root / str(
            (cfg.get("train", {}) or {}).get("output_dir", "train")
        )
        if candidate.exists():
            train_dir = candidate
    html_path = _write_html_index(
        out_root=out_root,
        title=str(cfg.get("title", "CS Pipeline Report")),
        config=cfg,
        panel_path=Path(panel_path) if panel_path else None,
        summary_csv=summary_csv,
        selected_factors_path=selected_path,
        report_md_path=report_md,
        train_dir=train_dir,
    )
    print(f"✅ HTML report: {html_path}")

    # 4) Save pipeline manifest
    _maybe_write_json(
        out_root / "pipeline_manifest.json",
        {
            "config": str(Path(args.config).resolve()),
            "output_root": str(out_root),
            "factor_eval_dir": str(eval_out),
            "selected_factors": str(selected_path) if selected_path.exists() else None,
            "html_report": str(html_path),
            "report_md": str(report_md) if report_md else None,
            "train_dir": str(train_dir) if train_dir else None,
        },
    )
    print(f"✅ CS pipeline finished. Output: {out_root}")


if __name__ == "__main__":
    main()
