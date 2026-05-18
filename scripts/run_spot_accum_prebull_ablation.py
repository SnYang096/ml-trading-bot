#!/usr/bin/env python3
"""Run spot_accum pre-bull hoarding ablations (same 2022-01..2026-05 window)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "config" / "strategies" / "bad-candidates" / "spot_accum"
ABLATE_ROOT = ROOT / "config" / "strategies" / "_ablate_spot_accum"
OUT_ROOT = ROOT / "results" / "120T" / "spot_accum" / "prebull_ablation"
CUTOFF = pd.Timestamp("2023-01-01", tz="UTC")
BUDGET = {"BTCUSDT": 5000.0, "BNBUSDT": 2500.0, "SOLUSDT": 2500.0}


def _load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _dump_yaml(p: Path, obj: dict) -> None:
    p.write_text(
        yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _setup_variant(name: str, exec_patch: dict, entry_patch: dict | None = None) -> str:
    strat = f"spot_accum__{name}"
    vdir = ABLATE_ROOT / strat
    if vdir.exists():
        shutil.rmtree(vdir)
    shutil.copytree(SRC, vdir)
    arch = vdir / "archetypes"
    ex = _load_yaml(arch / "execution.yaml")
    ex.setdefault("stop_loss", {}).setdefault("regime_lifecycle_exit", {})
    sl = ex["stop_loss"]
    for k, v in (exec_patch.get("stop_loss") or {}).items():
        if k == "regime_lifecycle_exit" and isinstance(v, dict):
            sl.setdefault("regime_lifecycle_exit", {}).update(v)
        else:
            sl[k] = v
    rle = sl.setdefault("regime_lifecycle_exit", {})
    rle.update(exec_patch.get("regime_lifecycle_exit") or {})
    if "accumulation_policy" in exec_patch:
        ex["accumulation_policy"] = dict(exec_patch.get("accumulation_policy") or {})
    for k, v in (exec_patch.get("execution_constraints") or {}).items():
        ex.setdefault("execution_constraints", {})[k] = v
    _dump_yaml(arch / "execution.yaml", ex)
    if entry_patch is not None:
        ef = _load_yaml(arch / "entry_filters.yaml")
        ef.update(entry_patch)
        _dump_yaml(arch / "entry_filters.yaml", ef)
    return strat


def _run(name: str, strat: str) -> Path:
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "event_backtest.py"),
        "--strategy",
        strat,
        "--symbols",
        "BTCUSDT,BNBUSDT,SOLUSDT",
        "--start-date",
        "2022-01-01",
        "--end-date",
        "2026-05-01",
        "--data-path",
        str(ROOT / "data" / "parquet_data"),
        "--constitution-yaml",
        str(ROOT / "config" / "constitution" / "constitution.yaml"),
        "--strategies-root",
        str(ABLATE_ROOT),
        "--trades-csv",
        str(out / "spot_trades.csv"),
        "--output",
        str(out / "spot.json"),
        "--capital-report",
        str(out),
        "--quiet-signal-logs",
    ]
    print(f"\n=== RUN {name} ===", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return out


def _metrics(out: Path) -> dict:
    tr = pd.read_csv(out / "spot_trades.csv")
    tr["ts"] = pd.to_datetime(tr["entry_timestamp"].fillna(tr["entry_time"]), utc=True)
    closes = tr[tr["exit_reason"].notna() & (tr["exit_reason"] != "")].copy()
    closes["entry_ts"] = pd.to_datetime(closes["entry_time"], utc=True)
    closes["exit_ts"] = pd.to_datetime(closes["exit_time"], utc=True)
    obj = json.loads((out / "spot.json").read_text(encoding="utf-8"))
    curve = obj.get("equity_curve") or []
    final_eq = float(curve[-1]) if curve else float("nan")
    open_at = 0.0
    per_sym = {}
    for sym, cap in BUDGET.items():
        o = closes[
            (closes["symbol"] == sym)
            & (closes["entry_ts"] < CUTOFF)
            & (closes["exit_ts"] >= CUTOFF)
        ]
        d = float(o["notional_usdt"].sum()) if len(o) else 0.0
        per_sym[sym] = d
        open_at += d
    return {
        "closes": len(closes),
        "pnl_usd": float(closes["pnl_usd_realized"].sum()) if len(closes) else 0.0,
        "final_equity": final_eq,
        "return_pct": (
            100.0 * (final_eq / 10000.0 - 1.0) if final_eq == final_eq else float("nan")
        ),
        "open_2023_usdt": open_at,
        "open_2023_pct": 100.0 * open_at / sum(BUDGET.values()),
        "btc_open": per_sym.get("BTCUSDT", 0),
        "bnb_open": per_sym.get("BNBUSDT", 0),
        "sol_open": per_sym.get("SOLUSDT", 0),
        "signals": int((obj.get("funnel") or {}).get("signals_generated", 0) or 0),
        "exit_risk_off": int(
            (closes["exit_reason"] == "structural_exit_abc_macro_regime_risk_off").sum()
        ),
    }


def main() -> None:
    variants = {
        "a_simple_cycle_death": {
            "exec": {},
        },
        "lifecycle_relaxed": {
            "exec": {
                "regime_lifecycle_exit": {
                    "risk_off_drop_min": 2.5,
                    "risk_off_floor_score": 2.0,
                    "arm_risk_off_min_peak": 5.0,
                },
            },
        },
        "deploy_fast": {
            "exec": {
                "execution_constraints": {
                    "max_deploy_legs_per_day": 3,
                    "min_order_interval_minutes": 60,
                    "entry_order": {"type": "market"},
                },
            },
        },
        "entry_relaxed": {
            "exec": {},
            "entry": {
                "filters": [
                    {
                        "id": "ef_spot_accum_weak_regime_reclaim_ema200",
                        "enabled": True,
                        "description": "EMA200 band only (no RSI gate)",
                        "conditions": [
                            {
                                "feature": "ema_200_position",
                                "operator": ">=",
                                "value": -0.03,
                            },
                            {
                                "feature": "ema_200_position",
                                "operator": "<=",
                                "value": 0.03,
                            },
                        ],
                    }
                ],
                "combination_mode": "or",
            },
        },
        "hoard_all": {
            "exec": {
                "regime_lifecycle_exit": {
                    "risk_off_drop_min": 2.5,
                    "risk_off_floor_score": 2.0,
                    "arm_risk_off_min_peak": 5.0,
                },
                "execution_constraints": {
                    "max_deploy_legs_per_day": 3,
                    "min_order_interval_minutes": 60,
                    "entry_order": {"type": "market"},
                },
            },
            "entry": {
                "filters": [
                    {
                        "id": "ef_spot_accum_weak_regime_reclaim_ema200",
                        "enabled": True,
                        "description": "EMA200 band only",
                        "conditions": [
                            {
                                "feature": "ema_200_position",
                                "operator": ">=",
                                "value": -0.03,
                            },
                            {
                                "feature": "ema_200_position",
                                "operator": "<=",
                                "value": 0.03,
                            },
                        ],
                    }
                ],
                "combination_mode": "or",
            },
        },
    }

    rows = []
    v4 = OUT_ROOT.parent / "retest_v4_lifecycle"
    if (v4 / "spot_trades.csv").exists():
        m = _metrics(v4)
        m["variant"] = "v4_baseline (current)"
        rows.append(m)

    for name, spec in variants.items():
        strat = _setup_variant(
            name,
            spec.get("exec") or {},
            spec.get("entry"),
        )
        out = _run(name, strat)
        m = _metrics(out)
        m["variant"] = name
        rows.append(m)

    df = pd.DataFrame(rows)
    cols = [
        "variant",
        "open_2023_usdt",
        "open_2023_pct",
        "btc_open",
        "bnb_open",
        "sol_open",
        "final_equity",
        "return_pct",
        "pnl_usd",
        "closes",
        "signals",
        "exit_risk_off",
    ]
    df = df[cols]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_ROOT / "comparison_table.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nWrote {OUT_ROOT / 'comparison_table.csv'}")


if __name__ == "__main__":
    main()
