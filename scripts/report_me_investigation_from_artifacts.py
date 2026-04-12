#!/usr/bin/env python3
"""
ME 大波动 / fast_month 漏斗调研：从 event JSON + parquet + rolling calibrated YAML 汇总一页表。

默认对齐 run:
  results/me/slow-rolling-sim/_rolling_sim/20260411_195721/fast_month_2024-11/me/event_backtest_me.json

示例:
  python scripts/report_me_investigation_from_artifacts.py \\
    --out results/me/slow-rolling-sim/_rolling_sim/20260411_195721/fast_month_2024-11/me/investigation_me_202411_big_move.txt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.time_series_model.archetype.loader import (
    EvidenceConfig,
    ExecutionConfig,
    GateConfig,
    PrefilterConfig,
    StrategyArchetype,
)
from src.time_series_model.live.generic_live_strategy import (
    DerivedEntryFeatureState,
    DirectionEvaluator,
    GateEvaluator,
)
from src.time_series_model.execution.entry_filter import (
    check_entry_filters_or_single,
    load_entry_filters_config,
)


def _stats(rows: list) -> dict:
    n = len(rows)
    if n == 0:
        return {"bars": 0}
    pf_false = sum(1 for r in rows if r.get("prefilter") is False)
    pf_true = sum(1 for r in rows if r.get("prefilter") is True)
    gate_false = sum(1 for r in rows if r.get("gate") is False)
    gate_true = sum(1 for r in rows if r.get("gate") is True)
    gate_null = sum(1 for r in rows if r.get("gate") is None)
    dir_any = sum(1 for r in rows if r.get("direction_value") is not None)
    dir_long = sum(1 for r in rows if r.get("direction_value") == 1)
    dir_short = sum(1 for r in rows if r.get("direction_value") == -1)
    full = sum(
        1
        for r in rows
        if r.get("prefilter") is True
        and r.get("direction_value") is not None
        and r.get("gate") is True
        and r.get("entry_filter") is True
    )
    return {
        "bars": n,
        "prefilter_false": pf_false,
        "prefilter_false_pct": pf_false / n,
        "prefilter_true": pf_true,
        "gate_false": gate_false,
        "gate_true": gate_true,
        "gate_null": gate_null,
        "direction_any": dir_any,
        "direction_long": dir_long,
        "direction_short": dir_short,
        "full_chain_pf_dir_gate_ef": full,
    }


def _filter_rows(
    rows: list,
    *,
    symbols: set[str],
    t0: pd.Timestamp,
    t1: pd.Timestamp,
) -> list:
    out = []
    for r in rows:
        if str(r.get("symbol") or "") not in symbols:
            continue
        ts = pd.Timestamp(r["timestamp"])
        if not (t0 <= ts < t1):
            continue
        out.append(r)
    return out


def _high_vol_chain_report(
    df: pd.DataFrame,
    *,
    rolling_root: Path,
    month_token: str,
) -> str:
    pf_p = (
        rolling_root
        / f"fast_month_{month_token}"
        / "strategies_calibrated/me/archetypes/prefilter.yaml"
    )
    gate_p = (
        rolling_root
        / f"fast_month_{month_token}"
        / "strategies_calibrated/me/archetypes/gate.yaml"
    )
    exec_p = gate_p.parent / "execution.yaml"
    dir_p = gate_p.parent / "direction.yaml"
    entry_p = gate_p.parent / "entry_filters.yaml"
    pf_cfg = PrefilterConfig.from_yaml(pf_p)
    raw_dir = yaml.safe_load(dir_p.read_text(encoding="utf-8")) or {}
    de = DirectionEvaluator(raw_dir)
    arche = StrategyArchetype(
        name="me",
        gate=GateConfig.from_yaml(gate_p),
        evidence=EvidenceConfig(),
        execution=ExecutionConfig.from_yaml(exec_p),
        prefilter=pf_cfg,
    )
    ge = GateEvaluator(arche)
    entry_cfg = load_entry_filters_config(
        "me", strategies_root=entry_p.parent.parent.parent
    )
    ef_state = DerivedEntryFeatureState()

    def eval_row(row: pd.Series) -> tuple:
        feats = {}
        for k, v in row.items():
            if isinstance(v, float) and pd.isna(v):
                continue
            feats[k] = v
        feats["timestamp"] = row["datetime"]
        ok, _ = pf_cfg.evaluate(feats)
        if not ok:
            return ("pf_fail",)
        direction, _ = de.evaluate(feats)
        if direction == 0:
            return ("pf_ok", "no_dir")
        g_ok, _, _ = ge.evaluate(feats, quantiles=None)
        if not g_ok:
            return ("pf_ok", "dir_ok", "gate_fail")
        merged = {**feats, **ef_state.update(feats)}
        if not check_entry_filters_or_single(merged, entry_cfg):
            return ("pf_ok", "dir_ok", "gate_ok", "ef_fail")
        return ("pf_ok", "dir_ok", "gate_ok", "ef_ok")

    lines: list[str] = []
    for sym in sorted(df["symbol"].astype(str).unique()):
        d = df[df["symbol"].astype(str) == sym].reset_index(drop=True)
        if d.empty or "me_atr_pct" not in d.columns:
            continue
        atr = pd.to_numeric(d["me_atr_pct"], errors="coerce")
        thr = float(atr.quantile(0.75))
        high = atr >= thr
        lines.append(f"\n=== {sym} high-vol: me_atr_pct >= month p75 = {thr:.4f} ===")
        for label, mask in [
            ("high_vol_top25", high.fillna(False)),
            ("rest_bottom75", ~(high.fillna(False))),
        ]:
            sub = d.loc[mask].reset_index(drop=True)
            n = len(sub)
            if n == 0:
                lines.append(f"  {label}: n=0")
                continue
            hist = Counter()
            pf_ok = full = 0
            for i in range(n):
                out = eval_row(sub.iloc[i])
                hist[out[0]] += 1
                if out[0] == "pf_ok":
                    pf_ok += 1
                if out == ("pf_ok", "dir_ok", "gate_ok", "ef_ok"):
                    full += 1
            lines.append(f"  {label}: n={n}")
            lines.append(f"    prefilter_pass: {pf_ok}/{n} ({pf_ok / n:.1%})")
            lines.append(f"    full_chain: {full}/{n} ({full / n:.1%})")
            lines.append(f"    outcome_hist: {dict(hist)}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--event-json",
        type=Path,
        default=_ROOT
        / "results/me/slow-rolling-sim/_rolling_sim/20260411_195721/fast_month_2024-11/me/event_backtest_me.json",
    )
    p.add_argument(
        "--parquet",
        type=Path,
        default=_ROOT
        / "results/train_final_20260411_173341_rr_extreme/me/features_labeled.parquet",
        help="需含 2024-11 SOL/XRP 行（与 rolling 同月标定一起用）",
    )
    p.add_argument(
        "--rolling-root",
        type=Path,
        default=_ROOT / "results/me/slow-rolling-sim/_rolling_sim/20260411_195721",
    )
    p.add_argument("--start", default="2024-11-01")
    p.add_argument("--end", default="2024-11-30")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="默认: <event-json 同目录>/investigation_me_202411_big_move.txt",
    )
    args = p.parse_args()

    out_path = args.out
    if out_path is None:
        out_path = args.event_json.parent / "investigation_me_202411_big_move.txt"

    t0 = pd.Timestamp(args.start, tz="UTC")
    t1 = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)
    t_rally0 = pd.Timestamp("2024-11-08", tz="UTC")
    syms = {"SOLUSDT", "XRPUSDT"}

    data = json.loads(args.event_json.read_text(encoding="utf-8"))
    funnel = data.get("funnel", {})
    rows = data.get("funnel_per_bar", [])
    has_gr = sum(1 for r in rows if r.get("gate_reasons") is not None)
    has_pr = sum(1 for r in rows if r.get("prefilter_reason") is not None)

    lines: list[str] = []
    lines.append("ME fast_month 2024-11 — big-move / funnel investigation")
    lines.append(f"event_json: {args.event_json}")
    lines.append(f"parquet: {args.parquet}")
    lines.append(f"rolling_root: {args.rolling_root}")
    lines.append("")

    lines.append("== Aggregate funnel (event JSON top-level) ==")
    for k in sorted(funnel.keys()):
        lines.append(f"  {k}: {funnel[k]}")
    lines.append("")
    lines.append(
        "Note: reject_kill_switch counts intents blocked after PCM produced signals "
        "(see pipeline.log Kill Switch section); early-month losses trigger limits."
    )
    plog = args.event_json.parent / "pipeline.log"
    if plog.exists():
        raw = plog.read_text(encoding="utf-8", errors="replace").splitlines()
        ks = [
            ln
            for ln in raw
            if "Kill Switch" in ln or "kill_switch" in ln or "daily_loss" in ln
        ][:25]
        if ks:
            lines.append("pipeline.log (kill switch excerpt):")
            for ln in ks[:15]:
                lines.append(f"  {ln}")
    lines.append("")

    def dump_block(title: str, rowset: list):
        lines.append(title)
        s = _stats(rowset)
        for k, v in s.items():
            if isinstance(v, float) and "pct" in k:
                lines.append(f"  {k}: {v:.1%}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")

    dump_block(
        "== funnel_per_bar: SOL+XRP, Nov full ==",
        _filter_rows(rows, symbols=syms, t0=t0, t1=t1),
    )
    dump_block(
        "== funnel_per_bar: SOL+XRP, Nov 08 .. Nov 30 ==",
        _filter_rows(rows, symbols=syms, t0=t_rally0, t1=t1),
    )
    for sym in ["SOLUSDT", "XRPUSDT"]:
        dump_block(
            f"== funnel_per_bar: {sym} Nov full ==",
            _filter_rows(rows, symbols={sym}, t0=t0, t1=t1),
        )
        dump_block(
            f"== funnel_per_bar: {sym} Nov 08-30 ==",
            _filter_rows(rows, symbols={sym}, t0=t_rally0, t1=t1),
        )

    lines.append("== funnel_per_bar optional fields ==")
    lines.append(f"  rows with gate_reasons: {has_gr} / {len(rows)}")
    lines.append(f"  rows with prefilter_reason: {has_pr} / {len(rows)}")
    if has_gr == 0:
        lines.append(
            "  (Regenerate with current event_backtest.py to persist reasons per bar.)"
        )
    lines.append("")

    lines.append("== Offline rolling-month diagnose (parquet + month YAML) ==")
    diag = _ROOT / "scripts/diagnose_me_rolling_month_funnel.py"
    for sym in ["SOLUSDT", "XRPUSDT"]:
        lines.append(f"--- subprocess: {sym} ---")
        cp = subprocess.run(
            [
                sys.executable,
                str(diag),
                "--parquet",
                str(args.parquet),
                "--rolling-root",
                str(args.rolling_root),
                "--start",
                args.start,
                "--end",
                args.end,
                "--symbol",
                sym,
            ],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
        )
        lines.append(cp.stdout or "")
        if cp.stderr:
            lines.append(cp.stderr)
        if cp.returncode != 0:
            lines.append(f"EXIT {cp.returncode}")
    lines.append("")

    if args.parquet.exists():
        df = pd.read_parquet(args.parquet)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        m = (df["datetime"] >= t0) & (df["datetime"] < t1)
        m &= df["symbol"].astype(str).isin(syms)
        dnov = df.loc[m].copy()
        lines.append("== High-vol subset (parquet, me_atr_pct month p75 per symbol) ==")
        lines.append(
            _high_vol_chain_report(
                dnov, rolling_root=args.rolling_root, month_token="2024-11"
            )
        )
    else:
        lines.append("== High-vol: skipped (parquet missing) ==")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(out_path.read_text(encoding="utf-8"))
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
