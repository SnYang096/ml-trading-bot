#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _load_mode(mode_path: Path, symbol: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(mode_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if symbol is not None and "symbol" in df.columns:
        df = df[df["symbol"].astype(str) == symbol]
    return df


def _plot_scores(df: pd.DataFrame, *, title: str, out_path: Path) -> None:
    for col in ["tc_score", "te_score", "mean_score", "regime_score", "regime"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    bins = 40

    # TC score distribution
    axes[0].hist(df["tc_score"].dropna(), bins=bins, alpha=0.6, label="all")
    axes[0].hist(
        df.loc[df["regime"].astype(str) == "TC", "tc_score"].dropna(),
        bins=bins,
        alpha=0.6,
        label="TC regime",
    )
    axes[0].set_title("TC score distribution")
    axes[0].legend(loc="best")

    # TE score distribution
    axes[1].hist(df["te_score"].dropna(), bins=bins, alpha=0.6, label="all")
    axes[1].hist(
        df.loc[df["regime"].astype(str) == "TE", "te_score"].dropna(),
        bins=bins,
        alpha=0.6,
        label="TE regime",
    )
    axes[1].set_title("TE score distribution")
    axes[1].legend(loc="best")

    # MEAN score distribution
    axes[2].hist(df["mean_score"].dropna(), bins=bins, alpha=0.6, label="all")
    axes[2].hist(
        df.loc[df["regime"].astype(str) == "MEAN", "mean_score"].dropna(),
        bins=bins,
        alpha=0.6,
        label="MEAN regime",
    )
    axes[2].set_title("MEAN score distribution")
    axes[2].legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot regime score distributions.")
    ap.add_argument("--mode", required=True, help="mode_3action parquet")
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--all-symbols", action="store_true")
    ap.add_argument("--out", required=True, help="Output PNG or dir (all symbols)")
    args = ap.parse_args()

    mode_all = _load_mode(Path(args.mode), None)

    if args.all_symbols:
        if "symbol" not in mode_all.columns:
            raise ValueError("--all-symbols requires symbol column in mode file")
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for sym in sorted(set(mode_all["symbol"].astype(str).tolist())):
            df = mode_all[mode_all["symbol"].astype(str) == sym]
            out_path = out_dir / f"regime_scores_{sym}.png"
            _plot_scores(df, title=f"Regime scores: {sym}", out_path=out_path)
        return

    if args.symbol:
        df = (
            mode_all
            if "symbol" not in mode_all.columns
            else mode_all[mode_all["symbol"].astype(str) == str(args.symbol)]
        )
        _plot_scores(df, title=f"Regime scores: {args.symbol}", out_path=Path(args.out))
        return

    # No symbol: aggregate view
    _plot_scores(mode_all, title="Regime scores (all symbols)", out_path=Path(args.out))


if __name__ == "__main__":
    main()
