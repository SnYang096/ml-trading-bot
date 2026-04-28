#!/usr/bin/env python3
"""
ME: gate / prefilter 相关特征 — 决策行 vs 开仓截面
================================================

两类用法（解决「开仓 2h 对齐 ≠ gate bar」问题）:

1) logs_gated 模式（推荐，能直接对齐 gate 语义）
   - 读 ``logs_gated.parquet``（含 ``gate_decision``、特征列、通常含 ``entry_direction``）。
   - 在 **gate_decision == allow** 且 **有入场意图** 的行上统计
     ``me_semantic_chop``、``ema_1200_position``、box 代理等；并回答：
     「被 gate 放行的样本里，是否仍出现 me_semantic_chop > 0.40？」
     若长期接近 0，说明 hard gate 与列一致；若明显 >0，需查列名/版本/评估顺序。

   示例::

     PYTHONPATH=src python scripts/diagnose_me_gate_vs_entry_features.py \\
       --mode logs_gated --logs-gated results/train_final_XXXX/me/logs_gated.parquet

2) rolling_trades 模式（无 logs_gated 时的弱证据）
   - 汇总 ``results/me/{turbo|slow}-rolling-sim/_rolling_sim/<run_id>/fast_month_*/me/event_trades_me.csv``
     的开仓腿（``is_add_position == False``），用 ``--feature-store-root`` 下 120T parquet
     按 ``--align`` 对齐到特征（默认 floor 2h bar open，与先前分析一致）。

   示例::

     PYTHONPATH=src python scripts/diagnose_me_gate_vs_entry_features.py \\
       --mode rolling_trades \\
       --runs slow:20260416_222324,turbo:20260427_142642 \\
       --feature-store-root feature_store/features_me_120T_0e69c1c57b
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent


def _bad_box_mid_row(row: pd.Series) -> bool:
    bp, bu, bd = (
        row.get("box_pos_120"),
        row.get("box_breakout_up"),
        row.get("box_breakout_down"),
    )
    if any(pd.isna(x) for x in (bp, bu, bd)):
        return False
    in_mid = 0.15 < float(bp) < 0.85
    weak = float(bu) < 0.5 and float(bd) < 0.5
    return bool(in_mid and weak)


def _load_floor2h_row(
    fs_root: Path, symbol: str, ts: pd.Timestamp, cols: list[str]
) -> pd.Series | None:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    ym = f"{ts.year:04d}-{ts.month:02d}"
    p = fs_root / symbol / "120T" / f"{ym}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=cols)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.sort_index()
    ts_bar = ts.floor("2h")
    if ts_bar in df.index:
        return df.loc[ts_bar]
    i = int(df.index.searchsorted(ts_bar)) - 1
    if i < 0:
        return None
    return df.iloc[i]


def _collect_opens_from_rolling(base: str, run_id: str) -> pd.DataFrame:
    root = PROJECT / "results" / "me" / f"{base}-rolling-sim" / "_rolling_sim" / run_id
    parts: list[pd.DataFrame] = []
    for csv in sorted(root.glob("fast_month_*/me/event_trades_me.csv")):
        if csv.stat().st_size < 50:
            continue
        try:
            df = pd.read_csv(csv)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def cmd_logs_gated(path: Path) -> int:
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.name != "timestamp":
        df = df.reset_index()
    if "timestamp" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})

    mask = pd.Series(True, index=df.index)
    if "gate_decision" in df.columns:
        gated = df["gate_decision"].astype(str).str.lower()
        allow_mask = gated == "allow"
    elif "gate_passed" in df.columns:
        allow_mask = df["gate_passed"] == True  # noqa: E712
    else:
        print("WARN: no gate_decision / gate_passed; using all rows", file=sys.stderr)
        allow_mask = pd.Series(True, index=df.index)

    mask &= allow_mask
    if "entry_direction" in df.columns:
        mask &= df["entry_direction"].fillna(0).astype(int) != 0

    sub = df.loc[mask].copy()
    n = len(sub)
    print(f"logs_gated: {path}")
    print(f"  rows gate_allow(+entry_dir!=0 if present): n={n}")
    if n == 0:
        return 1

    need = [
        "me_semantic_chop",
        "ema_1200_position",
        "box_pos_120",
        "box_breakout_up",
        "box_breakout_down",
    ]
    missing = [c for c in need if c not in sub.columns]
    if missing:
        print("  missing columns:", missing, file=sys.stderr)
    chop = sub["me_semantic_chop"] if "me_semantic_chop" in sub.columns else None
    if chop is not None and chop.notna().any():
        c = chop.astype(float)
        print(
            f"  me_semantic_chop: P(>0.40)={float((c > 0.40).mean()) * 100:.2f}%  "
            f"median={c.median():.4f} p90={c.quantile(0.9):.4f}"
        )
    if "ema_1200_position" in sub.columns:
        e = sub["ema_1200_position"].astype(float)
        print(
            f"  ema_1200_position: P(|x|<0.03)={float((e.abs() < 0.03).mean()) * 100:.2f}%  median={e.median():+.4f}"
        )
    if all(
        c in sub.columns
        for c in ("box_pos_120", "box_breakout_up", "box_breakout_down")
    ):
        bb = sub.apply(_bad_box_mid_row, axis=1)
        print(f"  bad_box_mid (proxy): P={float(bb.mean()) * 100:.2f}%")

    # Outcome column if present
    for oc in ("forward_rr", "rr", "realized_rr", "return_atr"):
        if oc in sub.columns:
            r = sub[oc].astype(float)
            if chop is not None and "me_semantic_chop" in sub.columns:
                print(
                    f"  Spearman(chop, {oc})={sub['me_semantic_chop'].astype(float).corr(r, method='spearman'):+.4f}"
                )
            break

    return 0


def cmd_rolling_trades(
    fs_root: Path, runs_spec: list[tuple[str, str]], align: str
) -> int:
    cols = [
        "me_semantic_chop",
        "ema_1200_position",
        "box_pos_120",
        "box_breakout_up",
        "box_breakout_down",
        "ema_1200_slope_10",
    ]
    for base, rid in runs_spec:
        trades = _collect_opens_from_rolling(base, rid)
        if trades.empty:
            print(f"{base}/{rid}: no trades csv")
            continue
        opens = trades[trades["is_add_position"] == False].copy()
        miss = 0
        rows = []
        for _, t in opens.iterrows():
            if align != "floor2h":
                print("only floor2h supported for now", file=sys.stderr)
                return 2
            ser = _load_floor2h_row(
                fs_root, str(t["symbol"]), pd.Timestamp(t["entry_time"]), cols
            )
            if ser is None:
                miss += 1
                continue
            d = ser.to_dict()
            d["pnl_r"] = float(t["pnl_r"])
            d["run"] = f"{base}/{rid}"
            rows.append(d)
        if not rows:
            print(f"{base}/{rid}: no aligned rows miss={miss}")
            continue
        odf = pd.DataFrame(rows)
        chop = odf["me_semantic_chop"].astype(float)
        ema = odf["ema_1200_position"].astype(float)
        pnl = odf["pnl_r"].astype(float)
        print(
            f"{base}/{rid}: opens={len(opens)} aligned={len(odf)} miss={miss}  "
            f"P(chop>0.40)={float((chop > 0.40).mean()) * 100:.2f}%  "
            f"P(|ema|<0.03)={float((ema.abs() < 0.03).mean()) * 100:.2f}%  "
            f"P(bad_box)={float(odf.apply(_bad_box_mid_row, axis=1).mean()) * 100:.2f}%  "
            f"corr(pnl,chop)={chop.corr(pnl):+.3f}"
        )
    return 0


def _parse_runs(s: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"bad run token {part!r}, want base:run_id")
        base, rid = part.split(":", 1)
        if base not in ("slow", "turbo"):
            raise ValueError(f"base must be slow|turbo, got {base}")
        out.append((base, rid))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mode",
        choices=["logs_gated", "rolling_trades"],
        required=True,
    )
    ap.add_argument(
        "--logs-gated",
        type=Path,
        default=None,
        help="logs_gated.parquet (mode=logs_gated)",
    )
    ap.add_argument(
        "--runs",
        type=str,
        default="",
        help="mode=rolling_trades: slow:20260416_222324,turbo:20260427_142642",
    )
    ap.add_argument(
        "--feature-store-root",
        type=Path,
        default=PROJECT / "feature_store" / "features_me_120T_0e69c1c57b",
        help="ME 120T FeatureStore root (rolling_trades mode)",
    )
    ap.add_argument("--align", choices=["floor2h"], default="floor2h")
    args = ap.parse_args()

    if args.mode == "logs_gated":
        if not args.logs_gated:
            ap.error("--logs-gated required")
        if not args.logs_gated.is_file():
            print(f"not a file: {args.logs_gated}", file=sys.stderr)
            return 1
        return cmd_logs_gated(args.logs_gated.resolve())

    runs = _parse_runs(args.runs) if args.runs else []
    if not runs:
        ap.error("--runs required for rolling_trades")
    root = args.feature_store_root.resolve()
    if not root.is_dir():
        print(f"feature store not found: {root}", file=sys.stderr)
        return 1
    return cmd_rolling_trades(root, runs, args.align)


if __name__ == "__main__":
    raise SystemExit(main())
