#!/usr/bin/env python3
"""
按「自然月」使用 rolling_sim 当月 strategies_calibrated 配置，复盘 ME 漏斗死因。

典型用法（与一次 slow rolling 输出对齐）:
  python scripts/diagnose_me_rolling_month_funnel.py \\
    --parquet results/train_final_*_rr_extreme/me/features_labeled.parquet \\
    --rolling-root results/me/slow-rolling-sim/_rolling_sim/20260411_150326 \\
    --start 2024-11-01 --end 2024-12-31

对 2024-11 行读 fast_month_2024-11/strategies_calibrated/me/archetypes/prefilter.yaml，
对 2024-12 行读 fast_month_2024-12/...（与 event 回测当月配置一致）。
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import yaml

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument(
        "--rolling-root", type=Path, required=True, help="_rolling_sim/<run_id> 目录"
    )
    p.add_argument("--start", type=str, default="2024-11-01")
    p.add_argument("--end", type=str, default="2024-12-31")
    p.add_argument("--symbol", type=str, default="", help="可选，如 ADAUSDT")
    return p.parse_args()


def _month_token(ts: pd.Timestamp) -> str:
    return f"{ts.year:04d}-{ts.month:02d}"


def _prefilter_path(rolling_root: Path, month_token: str) -> Path:
    return (
        rolling_root
        / f"fast_month_{month_token}"
        / "strategies_calibrated"
        / "me"
        / "archetypes"
        / "prefilter.yaml"
    )


def _direction_path(rolling_root: Path, month_token: str) -> Path:
    return (
        rolling_root
        / f"fast_month_{month_token}"
        / "strategies_calibrated"
        / "me"
        / "archetypes"
        / "direction.yaml"
    )


def _gate_path(rolling_root: Path, month_token: str) -> Path:
    return (
        rolling_root
        / f"fast_month_{month_token}"
        / "strategies_calibrated"
        / "me"
        / "archetypes"
        / "gate.yaml"
    )


def _entry_path(rolling_root: Path, month_token: str) -> Path:
    return (
        rolling_root
        / f"fast_month_{month_token}"
        / "strategies_calibrated"
        / "me"
        / "archetypes"
        / "entry_filters.yaml"
    )


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main() -> None:
    args = _parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)

    df = pd.read_parquet(args.parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    m = (df["datetime"] >= start) & (df["datetime"] < end)
    if args.symbol:
        m &= df["symbol"].astype(str) == args.symbol
    d = df.loc[m].copy()
    if len(d) == 0:
        print("无样本")
        return
    d = d.reset_index(drop=True)

    reasons_pf = Counter()
    reasons_dir = Counter()
    reasons_gate = Counter()
    reasons_ef = Counter()
    n_pf_ok = n_dir_ok = n_gate_ok = n_ef_ok = 0
    n_rows = len(d)

    for i in range(n_rows):
        row = d.iloc[i]
        ts = row["datetime"]
        if pd.isna(ts):
            continue
        mt = _month_token(ts)
        pf_p = _prefilter_path(args.rolling_root, mt)
        if not pf_p.exists():
            print(f"WARN: 缺少当月 prefilter: {pf_p}")
            continue
        pf_cfg = PrefilterConfig.from_yaml(pf_p)
        feats = {}
        for k, v in row.items():
            if isinstance(v, float) and pd.isna(v):
                continue
            feats[k] = v
        feats["timestamp"] = ts

        ok, rsn = pf_cfg.evaluate(feats)
        if not ok:
            rsn = rsn or "prefilter_fail"
            if "prefilter_fail:" in rsn and "(actual=" in rsn:
                key = rsn.split("(actual=", 1)[0].strip()[:100]
            else:
                key = rsn[:120]
            reasons_pf[key] += 1
            continue
        n_pf_ok += 1

        dir_p = _direction_path(args.rolling_root, mt)
        raw_dir = _load_yaml(dir_p)
        de = DirectionEvaluator(raw_dir)
        direction, rule_id = de.evaluate(feats)
        if direction == 0:
            reasons_dir["no_direction"] += 1
            continue
        n_dir_ok += 1

        gate_p = _gate_path(args.rolling_root, mt)
        exec_p = gate_p.parent / "execution.yaml"
        arche = StrategyArchetype(
            name="me",
            gate=GateConfig.from_yaml(gate_p),
            evidence=EvidenceConfig(),
            execution=ExecutionConfig.from_yaml(exec_p),
            prefilter=pf_cfg,
        )
        ge = GateEvaluator(arche)
        g_ok, g_rsns, _w = ge.evaluate(feats, quantiles=None)
        if not g_ok:
            for gr in g_rsns or ["gate_deny"]:
                reasons_gate[str(gr)[:80]] += 1
            continue
        n_gate_ok += 1

        entry_p = _entry_path(args.rolling_root, mt)
        entry_cfg = load_entry_filters_config(
            "me", strategies_root=entry_p.parent.parent.parent
        )
        ef_state = DerivedEntryFeatureState()
        merged = {**feats, **ef_state.update(feats)}
        if check_entry_filters_or_single(merged, entry_cfg):
            n_ef_ok += 1
        else:
            reasons_ef["entry_filter_deny"] += 1

    print("=== ME rolling-month funnel (per-row, month-scoped YAML) ===")
    print(f"parquet: {args.parquet}")
    print(f"rolling_root: {args.rolling_root}")
    print(
        f"window(UTC): {start.date()} .. {args.end}  rows={n_rows}  symbol={args.symbol or '*'}"
    )
    print()
    print(f"prefilter pass: {n_pf_ok}/{n_rows} ({n_pf_ok / max(n_rows, 1):.1%})")
    if reasons_pf:
        print("  top prefilter_fail reasons:")
        for k, v in reasons_pf.most_common(12):
            print(f"    {v:5d}  {k}")
    print()
    print(
        f"direction != 0 (subset of rows, after pf pass): {n_dir_ok} "
        f"({n_dir_ok / max(n_pf_ok, 1):.1%} of pf-pass)"
    )
    if reasons_dir:
        print("  direction failures:", dict(reasons_dir))
    print()
    print(
        f"gate pass (after direction): {n_gate_ok} "
        f"({n_gate_ok / max(n_dir_ok, 1):.1%} of dir-ok)"
    )
    if reasons_gate:
        print("  gate deny tags:", dict(reasons_gate))
    if reasons_ef:
        print("  entry_filter:", dict(reasons_ef))
    print()
    print(
        f"full chain (pf ∧ dir ∧ gate ∧ entry): {n_ef_ok}/{n_rows} "
        f"({n_ef_ok / max(n_rows, 1):.1%})"
    )


if __name__ == "__main__":
    main()
