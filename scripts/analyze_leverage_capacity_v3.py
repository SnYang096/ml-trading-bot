"""Leverage capacity analysis v3 — BTC weekly bull regime + v2 pipeline.

Adds a **macro gate** aligned with `config/strategies/bad-candidates/lottery100/gate_draft.yaml`:

  - Resample BTC `close` to weekly (W-FRI last), compare to EWM(ema_span_weeks).
  - Lag by 1 resolved week (shift on weekly series), expand to bar index with ffill.
  - Attach `bull_regime` to every symbol's bars (ETH follows BTC anchor).

With `--bull-only`, all bucket / lift / tree stats use **only** rows where
`bull_regime` is True. This answers whether v2 rules recover lift **inside**
declared bull weeks vs mixing bear/chop.

Depends on `analyze_leverage_capacity_v2.py` (imported via importlib).

Example:
  python scripts/analyze_leverage_capacity_v3.py \\
      --bull-only \\
      --output-dir reports/leverage_capacity_v3
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_v2():
    path = ROOT / "scripts" / "analyze_leverage_capacity_v2.py"
    mod_name = "_lev_cap_v2"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # dataclasses resolves cls.__module__ ; must exist
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.__name__ = mod_name
    return mod


V2 = _load_v2()

# Re-export commonly used symbols
FEATURES = V2.FEATURES
MMR_DEFAULT = V2.MMR_DEFAULT


def btc_weekly_bull_on_bars(
    close: pd.Series,
    ema_weeks: int = 50,
    weekly_anchor: str = "W-FRI",
) -> pd.Series:
    """Weekly close > weekly EMA, lagged one resolved week, ffill to bar index."""
    w = close.resample(weekly_anchor).last().dropna()
    ema = w.ewm(span=ema_weeks, adjust=False).mean()
    bull_w = (w > ema).astype(bool)
    bull_w = bull_w.shift(1).fillna(False).astype(bool)
    out = bull_w.reindex(close.index, method="ffill").fillna(False).astype(bool)
    return out.rename("bull_regime")


@dataclass
class V3Cfg:
    symbols: List[str]
    timeframe: str
    horizons: List[int]
    fs_layer: str
    train: Tuple[str, str]
    test: Tuple[str, str]
    oos: Optional[Tuple[str, str]]
    output_dir: Path
    mmr: float
    regime_symbol: str
    ema_weeks: int
    bull_only: bool


def run(cfg: V3Cfg) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Weekly bull regime anchor: {cfg.regime_symbol} ===", flush=True)
    btc_df = V2.load_feature_store(cfg.fs_layer, cfg.regime_symbol, cfg.timeframe)
    bull_bar = btc_weekly_bull_on_bars(btc_df["close"].astype(float), cfg.ema_weeks)
    bull_frac = float(bull_bar.mean())
    print(
        f"  bull_regime=True fraction (all bars): {bull_frac:.3f}\n"
        f"  range: {bull_bar.index.min()} → {bull_bar.index.max()}",
        flush=True,
    )

    all_samples: List[pd.DataFrame] = []
    for sym in cfg.symbols:
        print(f"\n=== {sym} {cfg.timeframe} ===", flush=True)
        try:
            df = V2.load_feature_store(cfg.fs_layer, sym, cfg.timeframe)
        except FileNotFoundError as e:
            print(f"  MISS: {e}")
            continue
        df["bull_regime"] = bull_bar.reindex(df.index, method="ffill").fillna(False)
        print(f"  loaded {len(df)} bars {df.index.min()} → {df.index.max()}")
        for H in cfg.horizons:
            sp = V2.build_samples(df, H, sym, cfg.timeframe, mmr=cfg.mmr)
            # long block then short block — duplicate bull_regime
            br = pd.concat([df["bull_regime"], df["bull_regime"]], ignore_index=True)
            sp.insert(0, "bull_regime", br.values)
            sp = V2.dedup_plateaus(sp, lmax_col="lmax_adj")
            if cfg.bull_only:
                sp = sp[sp["bull_regime"]].copy()
            all_samples.append(sp)

    if not all_samples:
        print("no samples, aborted")
        return
    samples = pd.concat(all_samples, axis=0)

    tag = "bull_only" if cfg.bull_only else "all"

    for (sym, H), g in samples.groupby(["symbol", "horizon"]):
        path = cfg.output_dir / f"{sym}_{cfg.timeframe}_H{H}_samples_{tag}.parquet"
        g.reset_index().rename(columns={"index": "timestamp"}).to_parquet(
            path, index=False
        )
        print(f"  wrote {path.name}: {len(g)} rows")

    periods: Dict[str, Tuple[Optional[str], Optional[str]]] = {
        "train": cfg.train,
        "test": cfg.test,
    }
    if cfg.oos is not None:
        periods["oos"] = cfg.oos

    for H in cfg.horizons:
        sub_H = samples[samples["horizon"] == H]

        for per_name, (s, e) in periods.items():
            sub = V2.slice_period(sub_H, s, e)
            if sub.empty:
                continue
            bk = V2.bucket_summary(sub, lmax_col="lmax_adj")
            bk.insert(0, "horizon", H)
            bk.insert(0, "period", per_name)
            bk.to_csv(
                cfg.output_dir / f"bucket_counts_H{H}_{per_name}_{tag}.csv",
                index=False,
            )
            bk_raw = V2.bucket_summary(sub, lmax_col="lmax_raw")
            bk_raw.insert(0, "horizon", H)
            bk_raw.insert(0, "period", per_name)
            bk_raw.to_csv(
                cfg.output_dir / f"bucket_counts_H{H}_{per_name}_{tag}_raw.csv",
                index=False,
            )
            sl = V2.subset_lift_table(sub, lmax_col="lmax_adj")
            sl.insert(0, "horizon", H)
            sl.insert(0, "period", per_name)
            sl.to_csv(
                cfg.output_dir / f"subset_lift_H{H}_{per_name}_{tag}.csv",
                index=False,
            )
            fl = V2.feature_lift(sub, lmax_col="lmax_adj", use_dedup=True)
            if not fl.empty:
                fl.insert(0, "horizon", H)
                fl.insert(0, "period", per_name)
                fl.to_csv(
                    cfg.output_dir / f"feature_lift_H{H}_{per_name}_{tag}.csv",
                    index=False,
                )

        train = V2.slice_period(sub_H, *cfg.train)
        test = V2.slice_period(sub_H, *cfg.test)
        res = V2.train_tree(train, test, lmax_col="lmax_adj")
        path = cfg.output_dir / f"tree_rules_H{H}_{tag}.md"
        with open(path, "w") as f:
            f.write(
                f"# Decision tree rules H={H} ({tag})\n\n"
                f"Bull regime: weekly BTC close > EMA({cfg.ema_weeks}), lag 1w\n\n"
                f"train: {cfg.train}\n\ntest: {cfg.test}\n\n"
            )
            for side, txt in res.items():
                f.write(f"## side={side}\n\n```\n{txt}\n```\n\n")

            if cfg.oos is not None:
                oos = V2.slice_period(sub_H, *cfg.oos)
                if not oos.empty:
                    res_oos = V2.train_tree(train, oos, lmax_col="lmax_adj")
                    f.write(f"## Additional OOS: {cfg.oos}\n\n")
                    for side, txt in res_oos.items():
                        f.write(f"### side={side}\n\n```\n{txt}\n```\n\n")

        bull_train = V2.slice_period(sub_H, "2023-10-01", "2023-12-31")
        bull_test = V2.slice_period(sub_H, "2024-01-01", "2024-03-31")
        if len(bull_train) > 200 and len(bull_test) > 200:
            res_bull = V2.train_tree(bull_train, bull_test, lmax_col="lmax_adj")
            path_b = cfg.output_dir / f"tree_rules_H{H}_bullsplit_{tag}.md"
            with open(path_b, "w") as fb:
                fb.write(
                    f"# Within-bull split H={H} ({tag})\n\n"
                    "train: 2023-10-01..2023-12-31 | "
                    "test: 2024-01-01..2024-03-31\n\n"
                )
                for side, txt in res_bull.items():
                    fb.write(f"## side={side}\n\n```\n{txt}\n```\n\n")

    print(f"\nAll outputs written to {cfg.output_dir}")


def _parse_window(spec: str) -> Tuple[str, str]:
    a, b = spec.split(":")
    return a.strip(), b.strip()


def parse_args() -> V3Cfg:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--timeframe", default="120T")
    p.add_argument("--horizons", default="48,120")
    p.add_argument("--fs-layer", default="features_me_120T_e98fe79b58")
    p.add_argument("--train", default="2022-08-01:2023-09-30")
    p.add_argument("--test", default="2023-10-01:2024-03-31")
    p.add_argument("--oos", default="2024-04-01:2026-02-28")
    p.add_argument("--output-dir", default="reports/leverage_capacity_v3")
    p.add_argument("--mmr", type=float, default=MMR_DEFAULT)
    p.add_argument(
        "--regime-symbol",
        default="BTCUSDT",
        help="Symbol used for weekly bull regime (default BTC)",
    )
    p.add_argument("--ema-weeks", type=int, default=50)
    p.add_argument(
        "--bull-only",
        action="store_true",
        help="Keep only samples where bull_regime is True",
    )
    args = p.parse_args()
    return V3Cfg(
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        timeframe=args.timeframe,
        horizons=[int(x) for x in args.horizons.split(",")],
        fs_layer=args.fs_layer,
        train=_parse_window(args.train),
        test=_parse_window(args.test),
        oos=_parse_window(args.oos) if args.oos else None,
        output_dir=Path(args.output_dir),
        mmr=args.mmr,
        regime_symbol=args.regime_symbol.strip(),
        ema_weeks=args.ema_weeks,
        bull_only=args.bull_only,
    )


if __name__ == "__main__":
    run(parse_args())
