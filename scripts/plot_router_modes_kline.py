#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _load_mode(mode_path: Path, symbol: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(mode_path)
    if "timestamp" not in df.columns:
        raise ValueError("mode_3action must include timestamp column")
    if symbol is not None and "symbol" in df.columns:
        df = df[df["symbol"].astype(str) == symbol]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    keep = ["timestamp", "mode"]
    if "symbol" in df.columns:
        keep.append("symbol")
    if "gate_decision" in df.columns:
        keep.append("gate_decision")
    return df[keep].dropna()


def _load_ohlc(feature_store_root: Path, layer: str, symbol: str) -> pd.DataFrame:
    sym_dir = feature_store_root / layer / symbol / "240T"
    if not sym_dir.exists():
        raise FileNotFoundError(f"FeatureStore path not found: {sym_dir}")
    frames = []
    for p in sorted(sym_dir.glob("*.parquet")):
        df = pd.read_parquet(p)
        if "timestamp" not in df.columns:
            if df.index.name == "timestamp":
                df = df.reset_index()
            else:
                raise ValueError(f"timestamp column missing in {p}")
        frames.append(df[["timestamp", "open", "high", "low", "close"]])
    if not frames:
        raise ValueError(f"No featurestore parquet found under {sym_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out.sort_values("timestamp")


def _plot_one(
    *,
    mode_df: pd.DataFrame,
    ohlc_df: pd.DataFrame,
    symbol: str,
    out_path: Path,
    gate_filter: str,
) -> None:
    merged = pd.merge(ohlc_df, mode_df, on="timestamp", how="left")
    merged["mode"] = merged["mode"].fillna("NO_TRADE")
    if "gate_decision" in merged.columns:
        merged["gate_decision"] = merged["gate_decision"].fillna("no_trade")
        if gate_filter != "all":
            merged = merged[merged["gate_decision"] == gate_filter]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        merged["timestamp"],
        merged["close"],
        color="black",
        linewidth=1.0,
        label="close",
    )

    if "gate_decision" in merged.columns and gate_filter == "all":
        allow = merged[merged["gate_decision"] == "allow"]
        veto = merged[merged["gate_decision"] == "veto"]

        trend_allow = allow[allow["mode"] == "TREND"]
        mean_allow = allow[allow["mode"] == "MEAN"]
        trend_veto = veto[veto["mode"] == "TREND"]
        mean_veto = veto[veto["mode"] == "MEAN"]

        ax.scatter(
            trend_allow["timestamp"],
            trend_allow["close"],
            color="red",
            s=12,
            label="TREND (allow)",
        )
        ax.scatter(
            mean_allow["timestamp"],
            mean_allow["close"],
            color="blue",
            s=12,
            label="MEAN (allow)",
        )
        ax.scatter(
            trend_veto["timestamp"],
            trend_veto["close"],
            color="orange",
            s=10,
            marker="x",
            label="TREND (veto)",
        )
        ax.scatter(
            mean_veto["timestamp"],
            mean_veto["close"],
            color="cyan",
            s=10,
            marker="x",
            label="MEAN (veto)",
        )
    else:
        trend = merged[merged["mode"] == "TREND"]
        mean = merged[merged["mode"] == "MEAN"]

        ax.scatter(trend["timestamp"], trend["close"], color="red", s=12, label="TREND")
        ax.scatter(mean["timestamp"], mean["close"], color="blue", s=12, label="MEAN")

    title_suffix = ""
    if "gate_decision" in merged.columns:
        title_suffix = f" (gate={gate_filter})"
    ax.set_title(f"Router modes on OHLC close: {symbol}{title_suffix}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot router mode on OHLC close series.")
    ap.add_argument("--mode", required=True, help="mode_3action parquet")
    ap.add_argument("--feature-store-root", default="feature_store")
    ap.add_argument("--feature-store-layer", required=True)
    ap.add_argument("--symbol", default=None)
    ap.add_argument(
        "--all-symbols", action="store_true", help="Plot all symbols from mode_3action."
    )
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument(
        "--out",
        required=True,
        help="Output PNG path (single) or output dir (all symbols)",
    )
    ap.add_argument(
        "--gate-only",
        choices=["all", "allow", "veto"],
        default="all",
        help="If gate_decision exists, filter plotted points.",
    )
    args = ap.parse_args()

    mode_all = _load_mode(Path(args.mode), None)
    if args.start_date:
        start_ts = pd.to_datetime(args.start_date)
        mode_all = mode_all[mode_all["timestamp"] >= start_ts]
    if args.end_date:
        end_ts = pd.to_datetime(args.end_date)
        mode_all = mode_all[mode_all["timestamp"] <= end_ts]

    if args.all_symbols:
        if "symbol" not in mode_all.columns:
            raise ValueError(
                "--all-symbols requires mode_3action to include symbol column"
            )
        symbols = sorted(set(mode_all["symbol"].astype(str).tolist()))
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for sym in symbols:
            mode_df = mode_all[mode_all["symbol"].astype(str) == sym][
                ["timestamp", "mode"]
            ]
            ohlc_df = _load_ohlc(
                Path(args.feature_store_root), args.feature_store_layer, sym
            )
            if args.start_date:
                ohlc_df = ohlc_df[ohlc_df["timestamp"] >= start_ts]
            if args.end_date:
                ohlc_df = ohlc_df[ohlc_df["timestamp"] <= end_ts]
            out_path = out_dir / f"router_modes_{sym}.png"
            _plot_one(
                mode_df=mode_df,
                ohlc_df=ohlc_df,
                symbol=sym,
                out_path=out_path,
                gate_filter=args.gate_only,
            )
        return

    if not args.symbol:
        raise ValueError("Either --symbol or --all-symbols must be provided.")
    sym = str(args.symbol)
    mode_df = mode_all
    if "symbol" in mode_df.columns:
        mode_df = mode_df[mode_df["symbol"].astype(str) == sym][["timestamp", "mode"]]
    ohlc_df = _load_ohlc(Path(args.feature_store_root), args.feature_store_layer, sym)
    if args.start_date:
        ohlc_df = ohlc_df[ohlc_df["timestamp"] >= start_ts]
    if args.end_date:
        ohlc_df = ohlc_df[ohlc_df["timestamp"] <= end_ts]
    _plot_one(
        mode_df=mode_df,
        ohlc_df=ohlc_df,
        symbol=sym,
        out_path=Path(args.out),
        gate_filter=args.gate_only,
    )


if __name__ == "__main__":
    main()
