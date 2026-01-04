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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


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

    model_bt_html = ""
    if train_dir and train_dir.exists():
        rows = []
        for mode in ["long_only", "market_neutral"]:
            met = train_dir / f"model_bt_metrics__{mode}.json"
            if not met.exists():
                continue
            try:
                m = json.loads(met.read_text(encoding="utf-8"))
                rows.append(
                    f"<tr><td>{mode}</td>"
                    f"<td>{m.get('sharpe_net', float('nan')):.6g}</td>"
                    f"<td>{m.get('sharpe_gross', float('nan')):.6g}</td>"
                    f"<td>{m.get('avg_turnover', float('nan')):.6g}</td>"
                    f"<td>{m.get('fee_bps', float('nan')):.6g}</td>"
                    f"<td>{m.get('slippage_bps', float('nan')):.6g}</td>"
                    f"<td><a href='train/model_bt_timeseries__{mode}.csv'>csv</a> / "
                    f"<a href='train/model_bt_metrics__{mode}.json'>json</a></td></tr>"
                )
            except Exception:
                continue
        if rows:
            model_bt_html = (
                "<h2>Model backtest (OOS, realistic execution)</h2>"
                "<p>Modes: long-only and market-neutral long/short.</p>"
                "<table><thead><tr>"
                "<th>mode</th><th>Sharpe(net)</th><th>Sharpe(gross)</th><th>avg_turnover</th>"
                "<th>fee_bps</th><th>slippage_bps</th><th>artifacts</th>"
                "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
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
    </style>
  </head>
  <body>
    <h1>{title}</h1>
    <p>Output root: <code>{out_root}</code></p>
    {report_link}
    {train_link}
    {model_bt_html}
    <h2>Selected factors</h2>
    {selected_html}
    <h2>Factor evaluation summary</h2>
    {summary_html}
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
        fe_args += ["--factor-set", str(eval_cfg["factor_set"])]

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
    html_path = _write_html_index(
        out_root=out_root,
        title=str(cfg.get("title", "CS Pipeline Report")),
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
