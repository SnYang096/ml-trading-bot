"""Leverage capacity analysis v4 — YAML-driven **stricter** BTC bull regime + v2 pipeline.

Reads `config/strategies/bad-candidates/lottery100/leverage_capacity_v4.yaml` (override with `--config`).

Default stricter rule vs v3:
  - **weekly** close > weekly EMA(50), lagged one week (same as v3)
  - AND **monthly** 6-month return > `min_return`, with monthly signal lagged
    `lag_months` (default 1) before expanding to bars

Optional:
  - `weekly_ema_uptrend`: weekly EMA > EMA.shift(weeks_back), lagged

Outputs component flags on each sample row:
  `regime_weekly`, `regime_6m`, `regime_slope`, `bull_regime` (final AND).

Example:
  python scripts/analyze_leverage_capacity_v4.py \\
      --config config/strategies/bad-candidates/lottery100/leverage_capacity_v4.yaml \\
      --bull-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_v2():
    path = ROOT / "scripts" / "analyze_leverage_capacity_v2.py"
    mod_name = "_lev_cap_v2"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.__name__ = mod_name
    return mod


V2 = _load_v2()
MMR_DEFAULT = V2.MMR_DEFAULT


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def btc_weekly_bull_on_bars(
    close: pd.Series,
    ema_weeks: int,
    weekly_anchor: str,
    lag_weeks: int,
) -> pd.Series:
    w = close.resample(weekly_anchor).last().dropna()
    ema = w.ewm(span=ema_weeks, adjust=False).mean()
    bull_w = (w > ema).astype(bool)
    for _ in range(max(1, lag_weeks)):
        bull_w = bull_w.shift(1).fillna(False).astype(bool)
    out = bull_w.reindex(close.index, method="ffill").fillna(False).astype(bool)
    return out.rename("regime_weekly")


def monthly_6m_return_gate(
    close: pd.Series,
    min_return: float,
    lag_months: int,
) -> pd.Series:
    """True when lagged 6m monthly return > min_return."""
    m = close.resample("M").last().dropna()
    r6 = m.pct_change(6)
    for _ in range(max(1, lag_months)):
        r6 = r6.shift(1)
    ok = (r6 > min_return).fillna(False)
    out = ok.reindex(close.index, method="ffill").fillna(False).astype(bool)
    return out.rename("regime_6m")


def weekly_ema_uptrend_gate(
    close: pd.Series,
    ema_weeks: int,
    weekly_anchor: str,
    weeks_back: int,
    lag_weeks: int,
) -> pd.Series:
    w = close.resample(weekly_anchor).last().dropna()
    ema = w.ewm(span=ema_weeks, adjust=False).mean()
    upt = (ema > ema.shift(weeks_back)).astype(bool)
    for _ in range(max(1, lag_weeks)):
        upt = upt.shift(1).fillna(False).astype(bool)
    out = upt.reindex(close.index, method="ffill").fillna(False).astype(bool)
    return out.rename("regime_slope")


def build_bull_regime(close: pd.Series, regime_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Columns: regime_weekly, regime_6m, regime_slope, bull_regime."""
    rc = regime_cfg
    close = close.astype(float)
    wcfg = rc.get("weekly_close_gt_ema") or {}

    if wcfg.get("enabled", True):
        rw = btc_weekly_bull_on_bars(
            close,
            int(wcfg.get("ema_span_weeks", 50)),
            str(wcfg.get("weekly_anchor", "W-FRI")),
            int(wcfg.get("lag_weeks", 1)),
        )
    else:
        rw = pd.Series(True, index=close.index, name="regime_weekly")

    mcfg = rc.get("monthly_6m_return") or {}
    if mcfg.get("enabled", False):
        r6 = monthly_6m_return_gate(
            close,
            float(mcfg.get("min_return", 0.0)),
            int(mcfg.get("lag_months", 1)),
        )
    else:
        r6 = pd.Series(True, index=close.index, name="regime_6m")

    scfg = rc.get("weekly_ema_uptrend") or {}
    if scfg.get("enabled", False):
        ema_w = int(wcfg.get("ema_span_weeks", 50))
        rs = weekly_ema_uptrend_gate(
            close,
            ema_w,
            str(wcfg.get("weekly_anchor", "W-FRI")),
            int(scfg.get("weeks_back", 4)),
            int(scfg.get("lag_weeks", 1)),
        )
    else:
        rs = pd.Series(True, index=close.index, name="regime_slope")

    df = pd.concat([rw, r6, rs], axis=1)
    df["bull_regime"] = df.all(axis=1)
    return df


def _dup_sides(s: pd.Series) -> pd.Series:
    return pd.concat([s, s], ignore_index=True)


@dataclass
class V4Cfg:
    yaml_path: Path
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
    bull_only: bool
    regime_cfg: Dict[str, Any]


def _parse_window(spec: str) -> Tuple[str, str]:
    a, b = spec.split(":")
    return a.strip(), b.strip()


def run(cfg: V4Cfg) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== v4 regime ({cfg.yaml_path.name}) anchor: {cfg.regime_symbol} ===",
        flush=True,
    )
    btc_df = V2.load_feature_store(cfg.fs_layer, cfg.regime_symbol, cfg.timeframe)
    close = btc_df["close"].astype(float)
    reg = build_bull_regime(close, cfg.regime_cfg)
    print(
        f"  regime_weekly mean: {reg['regime_weekly'].mean():.3f} | "
        f"regime_6m mean: {reg['regime_6m'].mean():.3f} | "
        f"regime_slope mean: {reg['regime_slope'].mean():.3f}\n"
        f"  bull_regime (AND) mean: {reg['bull_regime'].mean():.3f}\n"
        f"  range: {close.index.min()} → {close.index.max()}",
        flush=True,
    )

    # KPI / bundle：anchor 全样本牛门占比（与 bull_only 子样本 parquet 里的 bull_bar_fraction=1 区分）
    rs_path = cfg.output_dir / "regime_summary.json"
    rs_path.write_text(
        json.dumps(
            {
                "anchor_symbol": cfg.regime_symbol,
                "n_anchor_bars": int(len(close)),
                "regime_weekly_mean": float(reg["regime_weekly"].mean()),
                "regime_6m_mean": float(reg["regime_6m"].mean()),
                "regime_slope_mean": float(reg["regime_slope"].mean()),
                "bull_bar_fraction": float(reg["bull_regime"].mean()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    all_samples: List[pd.DataFrame] = []
    for sym in cfg.symbols:
        print(f"\n=== {sym} {cfg.timeframe} ===", flush=True)
        try:
            df = V2.load_feature_store(cfg.fs_layer, sym, cfg.timeframe)
        except FileNotFoundError as e:
            print(f"  MISS: {e}")
            continue
        r = reg.reindex(df.index, method="ffill")
        for c in ("regime_weekly", "regime_6m", "regime_slope", "bull_regime"):
            df[c] = r[c].fillna(False).values
        print(f"  loaded {len(df)} bars {df.index.min()} → {df.index.max()}")
        for H in cfg.horizons:
            sp = V2.build_samples(df, H, sym, cfg.timeframe, mmr=cfg.mmr)
            for col in ("regime_weekly", "regime_6m", "regime_slope", "bull_regime"):
                sp[col] = _dup_sides(df[col]).values
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
                f"config: `{cfg.yaml_path}`\n\n"
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


def parse_args() -> V4Cfg:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(
            ROOT
            / "config/strategies/bad-candidates/lottery100/leverage_capacity_v4.yaml"
        ),
        help="YAML spec (regime + optional default windows)",
    )
    p.add_argument("--symbols", default=None, help="Override YAML defaults (comma-sep)")
    p.add_argument("--timeframe", default=None)
    p.add_argument("--horizons", default=None)
    p.add_argument("--fs-layer", default=None)
    p.add_argument("--train", default=None)
    p.add_argument("--test", default=None)
    p.add_argument("--oos", default=None)
    p.add_argument("--output-dir", default="reports/leverage_capacity_v4")
    p.add_argument("--mmr", type=float, default=None)
    p.add_argument(
        "--regime-symbol",
        default=None,
        help="Anchor symbol for regime (default from YAML regime.anchor_symbol)",
    )
    p.add_argument("--bull-only", action="store_true")
    args = p.parse_args()

    ypath = Path(args.config)
    yml = load_yaml(ypath)
    reg = yml["regime"]
    defs = yml.get("defaults") or {}
    wins = yml.get("windows") or {}

    sym_s = args.symbols or ",".join(defs.get("symbols", ["BTCUSDT", "ETHUSDT"]))
    horizons_s = args.horizons or ",".join(
        str(h) for h in defs.get("horizons", [48, 120])
    )

    train_s = args.train or wins.get("train", "2022-08-01:2023-09-30")
    test_s = args.test or wins.get("test", "2023-10-01:2024-03-31")
    oos_s = (
        args.oos if args.oos is not None else wins.get("oos", "2024-04-01:2026-02-28")
    )

    return V4Cfg(
        yaml_path=ypath,
        symbols=[s.strip() for s in sym_s.split(",") if s.strip()],
        timeframe=args.timeframe or defs.get("timeframe", "120T"),
        horizons=[int(x) for x in horizons_s.split(",")],
        fs_layer=args.fs_layer or defs.get("fs_layer", "features_me_120T_e98fe79b58"),
        train=_parse_window(train_s),
        test=_parse_window(test_s),
        oos=_parse_window(oos_s) if oos_s else None,
        output_dir=Path(args.output_dir),
        mmr=float(args.mmr if args.mmr is not None else defs.get("mmr", MMR_DEFAULT)),
        regime_symbol=(
            args.regime_symbol.strip()
            if args.regime_symbol
            else reg.get("anchor_symbol", "BTCUSDT")
        ),
        bull_only=args.bull_only,
        regime_cfg=reg,
    )


if __name__ == "__main__":
    run(parse_args())
