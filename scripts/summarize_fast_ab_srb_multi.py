#!/usr/bin/env python
"""多臂 AB 汇总：列出 root/<arm>/<month>/trades.csv 的 total_r / add 统计。

用法：python scripts/summarize_fast_ab_srb_multi.py <OUT_ROOT>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _load_month(arm_dir: Path, month: str) -> Dict[str, Any]:
    trades_csv = arm_dir / month / "trades.csv"
    summary_json = arm_dir / month / "summary.json"
    out: Dict[str, Any] = {
        "month": month,
        "n_trades": 0,
        "total_r": 0.0,
        "add_r": 0.0,
        "add_n": 0,
        "rej_retrace": 0,
        "rej_prefilter": 0,
        "rej_staged_2b_arm": 0,
        "rej_other": 0,
    }
    if trades_csv.exists():
        try:
            df = pd.read_csv(trades_csv)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            out["n_trades"] = int(len(df))
            r_col = (
                "pnl_r"
                if "pnl_r" in df.columns
                else ("r_multiple" if "r_multiple" in df.columns else None)
            )
            if r_col is not None:
                out["total_r"] = float(df[r_col].sum())
                if "is_add_position" in df.columns:
                    out["add_r"] = float(
                        df.loc[df["is_add_position"].astype(bool), r_col].sum()
                    )
                    out["add_n"] = int(df["is_add_position"].astype(bool).sum())
    if summary_json.exists():
        try:
            with open(summary_json) as f:
                js = json.load(f)
            funnel = js.get("funnel_stats") or {}
            out["rej_retrace"] = int(
                funnel.get("reject_add_shape_gate_retrace", 0) or 0
            )
            out["rej_prefilter"] = int(funnel.get("reject_prefilter_deny", 0) or 0)
            out["rej_staged_2b_arm"] = int(
                funnel.get("reject_srb_staged_2b_arm", 0) or 0
            )
        except Exception:
            pass
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: summarize_fast_ab_srb_multi.py <OUT_ROOT>")
        return 2
    root = Path(sys.argv[1])
    if not root.exists():
        print(f"no such dir: {root}")
        return 2

    arms = sorted([p.name for p in root.iterdir() if p.is_dir()])
    months_set: set[str] = set()
    for a in arms:
        months_set |= {p.name for p in (root / a).iterdir() if p.is_dir()}
    months = sorted(months_set)

    rows = []
    for m in months:
        row = {"month": m}
        for a in arms:
            v = _load_month(root / a, m)
            row[f"{a}_R"] = round(v["total_r"], 2)
            row[f"{a}_n"] = v["n_trades"]
            row[f"{a}_addR"] = round(v["add_r"], 2)
            row[f"{a}_addN"] = v["add_n"]
            row[f"{a}_rej"] = v["rej_retrace"]
            row[f"{a}_rejPF"] = v["rej_prefilter"]
            row[f"{a}_rej2b"] = v["rej_staged_2b_arm"]
        rows.append(row)

    df = pd.DataFrame(rows)
    total = {"month": "TOTAL"}
    for col in df.columns:
        if col == "month":
            continue
        total[col] = round(df[col].sum(), 2) if df[col].dtype != "O" else ""
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)

    # 先打印每臂 totalR 对比
    print("\n=== per-arm total R ===")
    print(
        df[[c for c in df.columns if c == "month" or c.endswith("_R")]].to_string(
            index=False
        )
    )

    print("\n=== per-arm add R ===")
    print(
        df[[c for c in df.columns if c == "month" or c.endswith("_addR")]].to_string(
            index=False
        )
    )

    print("\n=== per-arm rejections (add shape retrace) ===")
    print(
        df[[c for c in df.columns if c == "month" or c.endswith("_rej")]].to_string(
            index=False
        )
    )

    print("\n=== per-arm rejections (prefilter_deny) ===")
    print(
        df[[c for c in df.columns if c == "month" or c.endswith("_rejPF")]].to_string(
            index=False
        )
    )

    if any(c.endswith("_rej2b") for c in df.columns):
        print("\n=== per-arm rejections (srb_staged_2b arm) ===")
        print(
            df[
                [c for c in df.columns if c == "month" or c.endswith("_rej2b")]
            ].to_string(index=False)
        )

    print("\n=== per-arm n / add_n ===")
    print(
        df[
            [
                c
                for c in df.columns
                if c == "month" or c.endswith("_n") or c.endswith("_addN")
            ]
        ].to_string(index=False)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
