"""Compare box_pullback_trend diagnostic entries with TPC trades."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


def _load_bpt(samples_path: Path, execution: str) -> pd.DataFrame:
    df = pd.read_csv(samples_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[
        (df["signal"] == "edge_chop")
        & (df["macro_alignment"] == "aligned")
        & (df["execution"] == execution)
    ].copy()


def _load_tpc(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(run_dir.glob("fast_month_*/tpc/event_trades_tpc.csv")):
        try:
            df = pd.read_csv(path)
        except EmptyDataError:
            continue
        if df.empty:
            continue
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    tpc = pd.concat(frames, ignore_index=True)
    tpc["entry_time"] = pd.to_datetime(tpc["entry_time"], utc=True)
    tpc["exit_time"] = pd.to_datetime(tpc["exit_time"], utc=True)
    return tpc


def _overlap(
    bpt: pd.DataFrame, tpc: pd.DataFrame, tolerance_hours: int
) -> pd.DataFrame:
    if bpt.empty or tpc.empty:
        return bpt.assign(tpc_near=False, nearest_tpc_hours=pd.NA)

    rows = []
    tol = pd.Timedelta(hours=tolerance_hours)
    grouped_tpc = {
        key: group.sort_values("entry_time")
        for key, group in tpc.groupby(["symbol", "side"], sort=False)
    }
    for _, row in bpt.iterrows():
        cand = grouped_tpc.get((row["symbol"], row["side"]))
        nearest_hours = pd.NA
        tpc_near = False
        if cand is not None and not cand.empty:
            delta = (cand["entry_time"] - row["timestamp"]).abs()
            nearest = delta.min()
            nearest_hours = nearest / pd.Timedelta(hours=1)
            tpc_near = bool(nearest <= tol)
        out = row.to_dict()
        out["tpc_near"] = tpc_near
        out["nearest_tpc_hours"] = nearest_hours
        rows.append(out)
    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in {
        "all_bpt": df,
        "overlap_tpc": df[df["tpc_near"]],
        "exclusive_bpt": df[~df["tpc_near"]],
    }.items():
        rows.append(
            {
                "bucket": name,
                "n": len(g),
                "sum_r": g["r"].sum() if len(g) else 0.0,
                "mean_r": g["r"].mean() if len(g) else 0.0,
                "median_r": g["r"].median() if len(g) else 0.0,
                "win_rate": (g["r"] > 0).mean() if len(g) else 0.0,
                "long_n": (g["side"] == "LONG").sum() if len(g) else 0,
                "short_n": (g["side"] == "SHORT").sum() if len(g) else 0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bpt-samples", required=True)
    p.add_argument("--tpc-run-dir", required=True)
    p.add_argument("--execution", default="atr_stop_box_mid")
    p.add_argument("--tolerance-hours", type=int, default=24)
    p.add_argument(
        "--out-dir",
        default="results/bad-candidates/box_pullback_trend/diagnostic/overlap_tpc",
    )
    args = p.parse_args()

    bpt = _load_bpt(Path(args.bpt_samples), args.execution)
    tpc = _load_tpc(Path(args.tpc_run_dir))
    if tpc.empty:
        raise SystemExit(f"No TPC trades found under {args.tpc_run_dir}")

    start, end = tpc["entry_time"].min(), tpc["entry_time"].max()
    bpt = bpt[(bpt["timestamp"] >= start) & (bpt["timestamp"] <= end)].copy()
    compared = _overlap(bpt, tpc, args.tolerance_hours)
    summary = _summary(compared)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compared.to_csv(out_dir / "bpt_vs_tpc_overlap.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    print(
        f"TPC trades: n={len(tpc)} range={start}..{end} pnl_r={tpc['pnl_r'].sum():.2f}"
    )
    print(f"BPT compared: n={len(compared)} execution={args.execution}")
    print(summary.to_string(index=False))
    print(f"Saved outputs -> {out_dir}")


if __name__ == "__main__":
    main()
