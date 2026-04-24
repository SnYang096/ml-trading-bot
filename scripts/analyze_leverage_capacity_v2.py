"""Leverage capacity analysis v2 — orderflow / OI / funding + OOS.

Versus v1 (OHLCV-only), this version:

  1. Loads the **existing feature store** (ME layer by default) which already
     contains orderflow (CVD/VPIN/SHD/OFCI), OI, funding, ME/BPC semantic scores
     and EMA/VWAP1200 position. Covers 2022-08 onwards.
  2. Adjusts MAE / MFE for **funding cost** over the holding window (long vs short).
  3. **Deduplicates** consecutive ≥100x bars within a symbol/side: each "plateau"
     contributes only its first bar, so the base rate reflects independent
     opportunities rather than time-autocorrelated ones.
  4. Runs **OOS** training: train a shallow decision tree on a train window,
     report precision / recall / lift on a held-out test window.
  5. Also reports lift when sampled points pass **strategy-like thresholds**
     (e.g. `bpc_score_continuation > 0.3`) so we can quantify how existing
     signal constructs help select ≥100x bucket.

Usage:
  python scripts/analyze_leverage_capacity_v2.py \
      --symbols BTCUSDT,ETHUSDT \
      --timeframe 120T \
      --horizons 48,120 \
      --fs-layer features_me_120T_e98fe79b58 \
      --train 2022-08-01:2023-09-30 \
      --test 2023-10-01:2024-03-31 \
      --oos 2024-04-01:2026-02-28 \
      --output-dir reports/leverage_capacity_v2
"""

from __future__ import annotations

import argparse
import glob
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LEVERAGE_BUCKETS: List[Tuple[str, float, float]] = [
    (">=100x", 100.0, float("inf")),
    ("50-100x", 50.0, 100.0),
    ("20-50x", 20.0, 50.0),
    ("10-20x", 10.0, 20.0),
    ("5-10x", 5.0, 10.0),
    ("<5x", 0.0, 5.0),
]

MMR_DEFAULT = 0.004

# Feature subset for v2 lift / tree analysis. Keep ~30 features.
FEATURES: List[str] = [
    # vol / range (OHLCV) — same baseline as v1
    "me_atr_pct",
    "vol_zscore",
    "vol_persistence",
    # trend / location
    "ema_1200_position",
    "macro_tp_vwap_1200_position",
    "price_to_vwap_pct",
    "sma_200_position",
    # ME semantic
    "me_accel_persistence",
    "me_multi_tf_alignment",
    "me_cvd_alignment",
    "me_cvd_strength",
    "me_flow_exhaustion",
    # Order flow
    "cvd_roll20",
    "cvd_roll60",
    "ofci_pct",
    "shd_pct",
    "taker_buy_ratio",
    "vpin_zscore_20",
    "vpin_zscore_50",
    "vpin_quantile_rank_50",
    # LV / crowding
    "oi_change_pct",
    "oi_zscore",
    "funding_rate_zscore_50",
    "funding_oi_crowding_score",
    # Compression / breakout semantics
    "bpc_vol_compression_state",
    "bpc_bb_compression_state",
    "bpc_score_continuation",
    "bpc_score_breakout",
    "compression_duration",
    # Tail / structure
    "evt_tail_shape",
    "hurst_price_rolling",
]

OHLCV_COLS = ["open", "high", "low", "close", "volume"]
FUNDING_COL = "funding_rate"


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────


def load_feature_store(fs_layer: str, symbol: str, timeframe: str) -> pd.DataFrame:
    pattern = f"feature_store/{fs_layer}/{symbol}/{timeframe}/*.parquet"
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    keep = set(OHLCV_COLS + FEATURES + [FUNDING_COL])
    dfs: List[pd.DataFrame] = []
    for f in files:
        d = pd.read_parquet(f)
        cols = [c for c in d.columns if c in keep]
        dfs.append(d[cols])
    out = pd.concat(dfs, axis=0).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Forward MAE / MFE + funding cost
# ─────────────────────────────────────────────────────────────────────────────


def _forward_extreme(values: np.ndarray, horizon: int, mode: str) -> np.ndarray:
    n = len(values)
    if n <= horizon:
        return np.full(n, np.nan)
    padded = np.concatenate([values[1:], np.full(horizon, np.nan)])
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(padded, window_shape=horizon)
    assert windows.shape[0] == n
    if mode == "max":
        agg = np.nanmax(windows, axis=1)
    elif mode == "min":
        agg = np.nanmin(windows, axis=1)
    elif mode == "sum":
        agg = np.nansum(windows, axis=1)
    else:
        raise ValueError(mode)
    agg[n - horizon :] = np.nan
    return agg


def build_samples(
    df: pd.DataFrame,
    horizon: int,
    symbol: str,
    timeframe: str,
    mmr: float = MMR_DEFAULT,
) -> pd.DataFrame:
    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    fmax = _forward_extreme(high, horizon, "max")
    fmin = _forward_extreme(low, horizon, "min")

    # Funding cost over the horizon: funding_rate is per-bar funding fraction
    # (Binance: paid every 8h; in FS it's already resampled to bar interval).
    # For conservative estimate, assume funding_rate stays "current" over H bars.
    if FUNDING_COL in df.columns:
        fr = df[FUNDING_COL].astype(float).fillna(0.0).values
        # sum of future-window funding (bars t+1..t+H)
        fr_sum_window = _forward_extreme(fr, horizon, "sum")
    else:
        fr_sum_window = np.full_like(close, 0.0)

    # MAE / MFE (high/low based, unadjusted)
    raw_long_mae = np.clip((close - fmin) / close, 0.0, None)
    raw_long_mfe = np.clip((fmax - close) / close, 0.0, None)
    raw_short_mae = np.clip((fmax - close) / close, 0.0, None)
    raw_short_mfe = np.clip((close - fmin) / close, 0.0, None)

    # Funding-adjusted MAE: long pays positive funding (treat as additional adverse)
    # Conservative: add |funding sum| to adverse side.
    long_fund_cost = np.clip(fr_sum_window, 0.0, None)  # positive funding hurts long
    short_fund_cost = np.clip(-fr_sum_window, 0.0, None)  # negative funding hurts short
    long_mae_adj = raw_long_mae + long_fund_cost
    short_mae_adj = raw_short_mae + short_fund_cost
    long_mfe_adj = np.clip(raw_long_mfe - long_fund_cost, 0.0, None)
    short_mfe_adj = np.clip(raw_short_mfe - short_fund_cost, 0.0, None)

    def _lmax(mae: np.ndarray, mmr_val: float) -> np.ndarray:
        out = np.full_like(mae, np.nan, dtype=float)
        m = np.isfinite(mae) & (mae > 0)
        out[m] = (1.0 - mmr_val) / mae[m]
        out[mae == 0] = 10000.0
        return out

    out_rows: List[pd.DataFrame] = []
    for side, mae_adj, mfe_adj, raw_mae, raw_mfe in [
        ("long", long_mae_adj, long_mfe_adj, raw_long_mae, raw_long_mfe),
        ("short", short_mae_adj, short_mfe_adj, raw_short_mae, raw_short_mfe),
    ]:
        rec = pd.DataFrame(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "side": side,
                "horizon": horizon,
                "raw_mae_pct": raw_mae,
                "raw_mfe_pct": raw_mfe,
                "mae_pct_adj": mae_adj,
                "mfe_pct_adj": mfe_adj,
                "fund_cost_window": (
                    long_fund_cost if side == "long" else short_fund_cost
                ),
                "lmax_adj": _lmax(mae_adj, mmr),
                "lmax_raw": _lmax(raw_mae, mmr),
            },
            index=df.index,
        )
        for c in FEATURES:
            if c in df.columns:
                rec[c] = df[c].values
        rec["close"] = df["close"].values
        out_rows.append(rec)
    return pd.concat(out_rows, axis=0)


def dedup_plateaus(samples: pd.DataFrame, lmax_col: str) -> pd.DataFrame:
    """For each (symbol, side), mark only the first bar of each ≥100x run.

    Returns a copy with an additional bool column `is_first_100x`.
    Subsequent analysis can filter by this flag when computing base rate.
    """
    # Reset index to a unique integer "_row" so we can safely assign back
    s = samples.copy()
    ts_col = s.index.name or "ts"
    s = s.reset_index().rename(
        columns={s.columns[0] if s.index.name is None else s.index.name: ts_col}
    )
    # After reset_index, original datetime index becomes a column (usually first col).
    s["is_100x"] = (s[lmax_col] >= 100.0).astype(bool)
    s["is_first_100x"] = False

    for (_, _), g in s.groupby(["symbol", "side"], sort=False):
        g_sorted = g.sort_values(ts_col, kind="stable") if ts_col in g.columns else g
        f = g_sorted["is_100x"].values
        is_first = np.zeros_like(f, dtype=bool)
        prev = False
        for i, v in enumerate(f):
            if v and not prev:
                is_first[i] = True
            prev = v
        s.loc[g_sorted.index[is_first], "is_first_100x"] = True

    # Restore datetime index
    if ts_col in s.columns:
        s = s.set_index(ts_col)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Period slicing + bucket
# ─────────────────────────────────────────────────────────────────────────────


def _parse_window(spec: str) -> Tuple[str, str]:
    a, b = spec.split(":")
    return a.strip(), b.strip()


def slice_period(
    df: pd.DataFrame, start: Optional[str], end: Optional[str]
) -> pd.DataFrame:
    idx = df.index
    mask = pd.Series(True, index=idx)
    if start:
        mask &= idx >= pd.Timestamp(start)
    if end:
        mask &= idx <= pd.Timestamp(end)
    return df[mask]


def assign_bucket(lmax: float) -> str:
    if not np.isfinite(lmax):
        return "nan"
    for label, lo, hi in LEVERAGE_BUCKETS:
        if lo <= lmax < hi:
            return label
    return "<5x"


def bucket_summary(samples: pd.DataFrame, lmax_col: str = "lmax_adj") -> pd.DataFrame:
    s = samples.dropna(subset=[lmax_col]).copy()
    s["bucket"] = s[lmax_col].apply(assign_bucket)
    grp = s.groupby(["symbol", "side", "bucket"], dropna=False)
    agg = grp.agg(
        count=("mae_pct_adj", "size"),
        mae_p50=("mae_pct_adj", lambda x: np.nanquantile(x, 0.5)),
        mae_p90=("mae_pct_adj", lambda x: np.nanquantile(x, 0.9)),
        mfe_p50=("mfe_pct_adj", lambda x: np.nanquantile(x, 0.5)),
        mfe_p90=("mfe_pct_adj", lambda x: np.nanquantile(x, 0.9)),
    ).reset_index()
    t = s.groupby(["symbol", "side"]).size().rename("side_total")
    agg = agg.merge(t, on=["symbol", "side"], how="left")
    agg["share"] = agg["count"] / agg["side_total"]
    order = {lab: i for i, (lab, _, _) in enumerate(LEVERAGE_BUCKETS)}
    agg["_o"] = agg["bucket"].map(order).fillna(99).astype(int)
    return agg.sort_values(["symbol", "side", "_o"]).drop(columns="_o")


# ─────────────────────────────────────────────────────────────────────────────
# Feature lift  (optionally using dedup)
# ─────────────────────────────────────────────────────────────────────────────


def feature_lift(
    samples: pd.DataFrame,
    target_bucket: str = ">=100x",
    n_quantiles: int = 10,
    lmax_col: str = "lmax_adj",
    use_dedup: bool = False,
) -> pd.DataFrame:
    s = samples.dropna(subset=[lmax_col]).copy()
    s["bucket"] = s[lmax_col].apply(assign_bucket)
    if use_dedup and "is_first_100x" in s.columns:
        s["is_target"] = s["is_first_100x"].astype(int)
    else:
        s["is_target"] = (s["bucket"] == target_bucket).astype(int)
    out_rows: List[pd.DataFrame] = []
    for side in s["side"].unique():
        s_side = s[s["side"] == side]
        base = s_side["is_target"].mean() if len(s_side) else np.nan
        for col in FEATURES:
            if col not in s_side.columns:
                continue
            x = s_side[[col, "is_target"]].dropna()
            if len(x) < 500 or x[col].nunique() < n_quantiles:
                continue
            try:
                x["q"] = pd.qcut(x[col], q=n_quantiles, labels=False, duplicates="drop")
            except ValueError:
                continue
            g = x.groupby("q").agg(
                n=("is_target", "size"),
                hit=("is_target", "sum"),
                feat_lo=(col, "min"),
                feat_hi=(col, "max"),
                feat_med=(col, "median"),
            )
            g["rate"] = g["hit"] / g["n"]
            g["lift"] = g["rate"] / base if base and base > 0 else np.nan
            g = g.reset_index()
            g.insert(0, "feature", col)
            g.insert(0, "side", side)
            g["base_rate"] = base
            out_rows.append(g)
    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic subset overlays
# ─────────────────────────────────────────────────────────────────────────────


SEMANTIC_SUBSETS: Dict[str, Dict[str, Tuple[str, float]]] = {
    # condition: column > threshold
    "bpc_continuation_0.3": {"bpc_score_continuation": (">", 0.3)},
    "bpc_breakout_0.3": {"bpc_score_breakout": (">", 0.3)},
    "me_accel_0.6_align_0.5": {
        "me_accel_persistence": (">=", 0.6),
        "me_multi_tf_alignment": (">=", 0.5),
    },
    "low_vol_q10": {"me_atr_pct": ("<", None)},  # threshold computed at runtime
    "compression_state": {"bpc_vol_compression_state": (">=", 1)},
}


def eval_subset(
    samples: pd.DataFrame, name: str, cond: Dict[str, Tuple[str, float]]
) -> pd.Series:
    m = pd.Series(True, index=samples.index)
    for col, (op, thr) in cond.items():
        if col not in samples.columns:
            return pd.Series(False, index=samples.index)
        v = samples[col].astype(float)
        if thr is None and col == "me_atr_pct":
            # dynamic: lower 10% quantile
            thr = np.nanquantile(v, 0.10)
        if op == ">":
            m &= v > thr
        elif op == ">=":
            m &= v >= thr
        elif op == "<":
            m &= v < thr
        elif op == "<=":
            m &= v <= thr
    return m


def subset_lift_table(
    samples: pd.DataFrame, lmax_col: str = "lmax_adj"
) -> pd.DataFrame:
    s = samples.dropna(subset=[lmax_col]).copy()
    s["bucket"] = s[lmax_col].apply(assign_bucket)
    rows = []
    for side in s["side"].unique():
        ss = s[s["side"] == side]
        base_rate = (ss["bucket"] == ">=100x").mean()
        rows.append(
            {
                "side": side,
                "subset": "ALL",
                "n": len(ss),
                "hit": int((ss["bucket"] == ">=100x").sum()),
                "rate": float(base_rate),
                "lift": 1.0,
                "mfe_p50_in_100x": (
                    float(ss.loc[ss["bucket"] == ">=100x", "mfe_pct_adj"].median())
                    if (ss["bucket"] == ">=100x").any()
                    else np.nan
                ),
            }
        )
        for subset_name, cond in SEMANTIC_SUBSETS.items():
            m = eval_subset(ss, subset_name, cond)
            sub = ss[m]
            if len(sub) < 100:
                continue
            rate = (sub["bucket"] == ">=100x").mean()
            rows.append(
                {
                    "side": side,
                    "subset": subset_name,
                    "n": len(sub),
                    "hit": int((sub["bucket"] == ">=100x").sum()),
                    "rate": float(rate),
                    "lift": float(rate / base_rate) if base_rate > 0 else np.nan,
                    "mfe_p50_in_100x": (
                        float(
                            sub.loc[sub["bucket"] == ">=100x", "mfe_pct_adj"].median()
                        )
                        if (sub["bucket"] == ">=100x").any()
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# OOS tree — train on train, evaluate on test and (optional) oos
# ─────────────────────────────────────────────────────────────────────────────


def _prepare_xy(
    samples: pd.DataFrame,
    lmax_col: str,
    median_impute: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, float]]:
    """Build X, y; impute per-feature median (from `median_impute` if provided)."""
    feat_cols = [c for c in FEATURES if c in samples.columns]
    X = samples[feat_cols].astype(float).copy()
    y_raw = samples[lmax_col]
    row_mask = y_raw.notna()
    X = X[row_mask]
    y = (y_raw[row_mask] >= 100.0).astype(int)
    # build/apply medians
    if median_impute is None:
        median_impute = {c: float(np.nanmedian(X[c].values)) for c in feat_cols}
    for c in feat_cols:
        med = median_impute.get(c, 0.0)
        if not np.isfinite(med):
            med = 0.0
        X[c] = X[c].fillna(med)
    return X, y, median_impute


def _topk_precision(
    proba: np.ndarray, y: np.ndarray, top_frac: float
) -> Tuple[float, int, int]:
    if len(proba) == 0:
        return np.nan, 0, 0
    k = max(1, int(len(proba) * top_frac))
    idx = np.argsort(-proba)[:k]
    hit = int(y[idx].sum())
    return hit / k, hit, k


def train_tree(
    train: pd.DataFrame,
    test: pd.DataFrame,
    lmax_col: str = "lmax_adj",
    max_depth: int = 4,
) -> Dict[str, str]:
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import precision_score, recall_score

    results: Dict[str, str] = {}
    feat_cols = [c for c in FEATURES if c in train.columns]
    for side in ["long", "short"]:
        tr = train[train["side"] == side]
        te = test[test["side"] == side]
        Xtr, ytr, medians = _prepare_xy(tr, lmax_col)
        Xte, yte, _ = _prepare_xy(te, lmax_col, median_impute=medians)
        if len(Xtr) < 1000 or ytr.sum() < 30 or len(Xte) < 500:
            results[side] = (
                f"insufficient samples: train={len(Xtr)} pos={int(ytr.sum())} "
                f"test={len(Xte)}"
            )
            continue
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=max(50, int(len(ytr) * 0.005)),
            class_weight="balanced",
            random_state=0,
        )
        clf.fit(Xtr, ytr)
        txt = export_text(clf, feature_names=feat_cols)
        pred_tr = clf.predict(Xtr)
        pred_te = clf.predict(Xte)
        prec_tr = precision_score(ytr, pred_tr, zero_division=0)
        rec_tr = recall_score(ytr, pred_tr, zero_division=0)
        prec_te = precision_score(yte, pred_te, zero_division=0)
        rec_te = recall_score(yte, pred_te, zero_division=0)
        base_te = yte.mean()
        lift_te = prec_te / base_te if base_te > 0 else np.nan

        # Top-K precision sweep (more actionable than threshold=0.5)
        proba_te = clf.predict_proba(Xte)[:, 1]
        y_te_arr = yte.values
        topk_lines = []
        for frac in [0.01, 0.03, 0.05, 0.10, 0.20]:
            p, hit, k = _topk_precision(proba_te, y_te_arr, frac)
            lift_k = p / base_te if base_te > 0 else np.nan
            topk_lines.append(
                f"  top {int(frac*100):>2}%  n={k:>4}  hit={hit:>4}  "
                f"prec={p:.3f}  lift={lift_k:.2f}"
            )

        fi = (
            pd.DataFrame({"feature": feat_cols, "importance": clf.feature_importances_})
            .sort_values("importance", ascending=False)
            .head(8)
            .to_string(index=False)
        )
        results[side] = (
            f"Train: n={len(ytr)} pos={int(ytr.sum())} "
            f"precision={prec_tr:.3f} recall={rec_tr:.3f}\n"
            f"Test:  n={len(yte)} pos={int(yte.sum())} "
            f"precision={prec_te:.3f} recall={rec_te:.3f} "
            f"base={base_te:.3f} lift={lift_te:.2f}\n"
            f"Top-K precision:\n" + "\n".join(topk_lines) + "\n\n"
            f"Top features:\n{fi}\n\nTree:\n{txt}"
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class V2Cfg:
    symbols: List[str]
    timeframe: str
    horizons: List[int]
    fs_layer: str
    train: Tuple[str, str]
    test: Tuple[str, str]
    oos: Optional[Tuple[str, str]]
    output_dir: Path
    mmr: float


def run(cfg: V2Cfg) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    all_samples: List[pd.DataFrame] = []
    for sym in cfg.symbols:
        print(f"\n=== {sym} {cfg.timeframe} ===", flush=True)
        try:
            df = load_feature_store(cfg.fs_layer, sym, cfg.timeframe)
        except FileNotFoundError as e:
            print(f"  MISS: {e}")
            continue
        print(f"  loaded {len(df)} bars {df.index.min()} → {df.index.max()}")
        for H in cfg.horizons:
            sp = build_samples(df, H, sym, cfg.timeframe, mmr=cfg.mmr)
            sp = dedup_plateaus(sp, lmax_col="lmax_adj")
            all_samples.append(sp)

    if not all_samples:
        print("no samples, aborted")
        return
    samples = pd.concat(all_samples, axis=0)
    # Persist raw samples per symbol × horizon
    for (sym, H), g in samples.groupby(["symbol", "horizon"]):
        path = cfg.output_dir / f"{sym}_{cfg.timeframe}_H{H}_samples.parquet"
        g.reset_index().rename(columns={"index": "ts"}).to_parquet(path, index=False)
        print(f"  wrote {path.name}: {len(g)} rows")

    # Per (horizon, period) analysis
    periods: Dict[str, Tuple[Optional[str], Optional[str]]] = {
        "train": cfg.train,
        "test": cfg.test,
    }
    if cfg.oos is not None:
        periods["oos"] = cfg.oos

    for H in cfg.horizons:
        sub_H = samples[samples["horizon"] == H]

        # 1) bucket summary per period
        for per_name, (s, e) in periods.items():
            sub = slice_period(sub_H, s, e)
            if sub.empty:
                continue
            bk = bucket_summary(sub, lmax_col="lmax_adj")
            bk.insert(0, "horizon", H)
            bk.insert(0, "period", per_name)
            bk.to_csv(
                cfg.output_dir / f"bucket_counts_H{H}_{per_name}.csv", index=False
            )

            # also unadjusted
            bk_raw = bucket_summary(sub, lmax_col="lmax_raw")
            bk_raw.insert(0, "horizon", H)
            bk_raw.insert(0, "period", per_name)
            bk_raw.to_csv(
                cfg.output_dir / f"bucket_counts_H{H}_{per_name}_raw.csv", index=False
            )

            # 2) subset lift table
            sl = subset_lift_table(sub, lmax_col="lmax_adj")
            sl.insert(0, "horizon", H)
            sl.insert(0, "period", per_name)
            sl.to_csv(cfg.output_dir / f"subset_lift_H{H}_{per_name}.csv", index=False)

            # 3) feature lift (dedup version)
            fl = feature_lift(sub, lmax_col="lmax_adj", use_dedup=True)
            if not fl.empty:
                fl.insert(0, "horizon", H)
                fl.insert(0, "period", per_name)
                fl.to_csv(
                    cfg.output_dir / f"feature_lift_H{H}_{per_name}.csv", index=False
                )

        # 4) OOS tree: train on 'train', eval on 'test' (+ 'oos' extra)
        train = slice_period(sub_H, *cfg.train)
        test = slice_period(sub_H, *cfg.test)
        res = train_tree(train, test, lmax_col="lmax_adj")
        path = cfg.output_dir / f"tree_rules_H{H}.md"
        with open(path, "w") as f:
            f.write(f"# Decision tree rules H={H}\n\n")
            f.write(f"train: {cfg.train}\n\ntest: {cfg.test}\n\n")
            for side, txt in res.items():
                f.write(f"## side={side}\n\n```\n{txt}\n```\n\n")

            if cfg.oos is not None:
                oos = slice_period(sub_H, *cfg.oos)
                if not oos.empty:
                    res_oos = train_tree(train, oos, lmax_col="lmax_adj")
                    f.write(f"## Additional OOS window: {cfg.oos}\n\n")
                    for side, txt in res_oos.items():
                        f.write(
                            f"### side={side} (train → oos window)\n\n```\n{txt}\n```\n\n"
                        )

        # 5) Within-bull split: train on 2023-10..2023-12, test on 2024-01..2024-03
        bull_train = slice_period(sub_H, "2023-10-01", "2023-12-31")
        bull_test = slice_period(sub_H, "2024-01-01", "2024-03-31")
        if len(bull_train) > 200 and len(bull_test) > 200:
            res_bull = train_tree(bull_train, bull_test, lmax_col="lmax_adj")
            path_b = cfg.output_dir / f"tree_rules_H{H}_bull.md"
            with open(path_b, "w") as fb:
                fb.write(
                    f"# Within-bull split H={H}\n\n"
                    f"train: 2023-10-01..2023-12-31 | test: 2024-01-01..2024-03-31\n\n"
                )
                for side, txt in res_bull.items():
                    fb.write(f"## side={side}\n\n```\n{txt}\n```\n\n")

    print(f"\nAll outputs written to {cfg.output_dir}")


def parse_args() -> V2Cfg:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--timeframe", default="120T")
    p.add_argument("--horizons", default="48,120")
    p.add_argument("--fs-layer", default="features_me_120T_e98fe79b58")
    p.add_argument("--train", default="2022-08-01:2023-09-30")
    p.add_argument("--test", default="2023-10-01:2024-03-31")
    p.add_argument("--oos", default="2024-04-01:2026-02-28")
    p.add_argument("--output-dir", default="reports/leverage_capacity_v2")
    p.add_argument("--mmr", type=float, default=MMR_DEFAULT)
    args = p.parse_args()
    return V2Cfg(
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        timeframe=args.timeframe,
        horizons=[int(x) for x in args.horizons.split(",")],
        fs_layer=args.fs_layer,
        train=_parse_window(args.train),
        test=_parse_window(args.test),
        oos=_parse_window(args.oos) if args.oos else None,
        output_dir=Path(args.output_dir),
        mmr=args.mmr,
    )


if __name__ == "__main__":
    run(parse_args())
