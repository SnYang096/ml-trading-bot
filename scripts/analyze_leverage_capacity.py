"""
Leverage capacity analysis.

问题: 历史上哪些入场点能承受 100x / 50x / 20x / 10x / 5x 杠杆而不爆仓?
      哪些 OHLCV 派生特征可以识别这些点?

方法:
  1. 加载 highcap symbols 的 OHLCV (timeframe 可配).
  2. 对每根 bar t 的 close 作为假想入场价, 计算 forward 窗口 H 内的
     - MAE_pct (最大不利波动占入场价比例, 基于 high/low, 保守口径)
     - MFE_pct (最大有利波动, 同上)
     对 long / short 两个方向分别记录.
  3. 按 MAE_pct 映射到"可支撑最大杠杆" L_max = floor( (1-MMR)/MAE_pct ).
     默认 MMR=0.4%, 另外同时给 MMR=0 的 naive 口径作参考.
  4. 分桶 (>=100x, 50-100x, 20-50x, 10-20x, 5-10x, <5x) 做计数与分布统计.
  5. 计算一组轻量特征 (ATR%, EMA 距离/斜率, 波动分位, 量能 z 分数, realized vol 等),
     对每桶做 quantile 分层 lift; 并训练浅层 DecisionTree 导出规则.
  6. 支持按时间切片 (全历史 / 2020H2-2021Q2 牛市 / 2023Q4-2024Q1 牛市).

输出: reports/leverage_capacity/<symbol>_<tf>/
  - samples_<horizon>.parquet            逐样本 (方向、MAE、MFE、L_max、特征)
  - bucket_counts_<horizon>_<period>.csv 分桶计数 + MAE/MFE 分位
  - feature_lift_<horizon>_<period>.csv  每特征每分位段命中 >=100x 的占比
  - tree_rules_<horizon>_<period>.md     浅层决策树可读规则

用法:
  python scripts/analyze_leverage_capacity.py \
      --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
      --timeframe 120T \
      --horizons 12,48,120 \
      --start-date 2020-01-01 \
      --end-date 2026-02-28
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_tools.data_handler import DataHandler  # noqa: E402


LEVERAGE_BUCKETS: List[Tuple[str, float, float]] = [
    # (label, min_L, max_L) — L_max falls into bucket if min_L <= L < max_L
    (">=100x", 100.0, float("inf")),
    ("50-100x", 50.0, 100.0),
    ("20-50x", 20.0, 50.0),
    ("10-20x", 10.0, 20.0),
    ("5-10x", 5.0, 10.0),
    ("<5x", 0.0, 5.0),
]

DEFAULT_PERIODS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "all": (None, None),
    "bull_2020_2021": ("2020-07-01", "2021-05-31"),
    "bull_2023_2024": ("2023-10-01", "2024-03-31"),
}

MMR_DEFAULT = 0.004


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering (light, OHLCV-only)
# ─────────────────────────────────────────────────────────────────────────────


def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """输入需含 open/high/low/close/volume (datetime index). 返回原 df + 一组派生特征."""
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    vol = out["volume"].astype(float)

    # True range / ATR
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    out["atr_pct"] = atr14 / close
    out["bar_range_pct"] = (high - low) / close

    # Log returns
    log_close = np.log(close.replace(0, np.nan))
    out["log_ret_1"] = log_close.diff(1)
    out["log_ret_5"] = log_close.diff(5)
    out["log_ret_20"] = log_close.diff(20)

    # Realized vol (std of 1-bar log returns, rolling 20)
    out["rv_20"] = out["log_ret_1"].rolling(20, min_periods=20).std()

    # EMA distances + slopes
    ema200 = _ema(close, 200)
    ema1200 = _ema(close, 1200)
    out["close_ema200_dist"] = (close - ema200) / ema200
    out["close_ema1200_dist"] = (close - ema1200) / ema1200

    # Normalized slope: (ema now - ema N bars ago) / ema_now / N
    def _slope_norm(s: pd.Series, lag: int) -> pd.Series:
        return (s - s.shift(lag)) / s / lag

    out["ema200_slope_20"] = _slope_norm(ema200, 20)
    out["ema1200_slope_100"] = _slope_norm(ema1200, 100)

    # Volume zscore
    vmean = vol.rolling(20, min_periods=20).mean()
    vstd = vol.rolling(20, min_periods=20).std()
    out["vol_z_20"] = (vol - vmean) / vstd

    # Consecutive same-direction bars (sign of log_ret_1)
    sgn = np.sign(out["log_ret_1"].fillna(0.0))
    # running streak counter
    streak = np.zeros(len(sgn), dtype=np.int32)
    prev = 0.0
    prev_streak = 0
    for i, s in enumerate(sgn.values):
        if s == 0 or s != prev:
            prev_streak = int(s)
        else:
            prev_streak = prev_streak + int(s)
        streak[i] = prev_streak
        prev = s
    out["consec_dir"] = streak

    # Position of close within last 20-bar high-low (0..1)
    hi20 = high.rolling(20, min_periods=20).max()
    lo20 = low.rolling(20, min_periods=20).min()
    out["pos_in_range_20"] = (close - lo20) / (hi20 - lo20)

    return out


FEATURE_COLS: List[str] = [
    "atr_pct",
    "bar_range_pct",
    "log_ret_5",
    "log_ret_20",
    "rv_20",
    "close_ema200_dist",
    "close_ema1200_dist",
    "ema200_slope_20",
    "ema1200_slope_100",
    "vol_z_20",
    "consec_dir",
    "pos_in_range_20",
]


# ─────────────────────────────────────────────────────────────────────────────
# Forward MAE / MFE over horizon H (vectorized)
# ─────────────────────────────────────────────────────────────────────────────


def _forward_extreme(values: np.ndarray, horizon: int, mode: str) -> np.ndarray:
    """Return array of per-index aggregate over values[i+1 : i+1+horizon].

    mode: "max" or "min". Positions without full horizon are NaN.
    """
    n = len(values)
    if n <= horizon:
        return np.full(n, np.nan)
    # pad so each index has a full window
    padded = np.concatenate([values[1:], np.full(horizon, np.nan)])
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(padded, window_shape=horizon)  # (n, horizon)
    assert windows.shape[0] == n
    if mode == "max":
        agg = np.nanmax(windows, axis=1)
    elif mode == "min":
        agg = np.nanmin(windows, axis=1)
    else:
        raise ValueError(mode)
    # Mark last `horizon` bars (no full future) as NaN
    agg[n - horizon :] = np.nan
    return agg


def build_samples(
    features: pd.DataFrame,
    horizon: int,
    symbol: str,
    mmr: float = MMR_DEFAULT,
) -> pd.DataFrame:
    """For each bar, emit 2 rows (long/short) with MAE, MFE, L_max."""
    close = features["close"].values.astype(float)
    high = features["high"].values.astype(float)
    low = features["low"].values.astype(float)

    fmax = _forward_extreme(high, horizon, "max")
    fmin = _forward_extreme(low, horizon, "min")

    # long: MAE = (entry - fmin)/entry clipped >=0
    long_mae = np.clip((close - fmin) / close, a_min=0.0, a_max=None)
    long_mfe = np.clip((fmax - close) / close, a_min=0.0, a_max=None)
    # short
    short_mae = np.clip((fmax - close) / close, a_min=0.0, a_max=None)
    short_mfe = np.clip((close - fmin) / close, a_min=0.0, a_max=None)

    def _lmax(mae: np.ndarray, mmr_val: float) -> np.ndarray:
        # avoid div-by-zero, cap at 10000x
        out = np.full_like(mae, np.nan, dtype=float)
        m = np.isfinite(mae) & (mae > 0)
        out[m] = (1.0 - mmr_val) / mae[m]
        out[mae == 0] = 10000.0  # never liquidated in window
        return out

    rows = []
    for side_name, mae, mfe in [
        ("long", long_mae, long_mfe),
        ("short", short_mae, short_mfe),
    ]:
        df = pd.DataFrame(
            {
                "symbol": symbol,
                "side": side_name,
                "horizon": horizon,
                "mae_pct": mae,
                "mfe_pct": mfe,
                "lmax_mmr": _lmax(mae, mmr),
                "lmax_naive": _lmax(mae, 0.0),
            },
            index=features.index,
        )
        for c in FEATURE_COLS:
            df[c] = features[c].values
        df["close"] = features["close"].values
        rows.append(df)
    return pd.concat(rows, axis=0)


def assign_bucket(lmax: float) -> str:
    if not np.isfinite(lmax):
        return "nan"
    for label, lo, hi in LEVERAGE_BUCKETS:
        if lo <= lmax < hi:
            return label
    return "<5x"


# ─────────────────────────────────────────────────────────────────────────────
# Aggregations
# ─────────────────────────────────────────────────────────────────────────────


def summarize_buckets(
    samples: pd.DataFrame, lmax_col: str = "lmax_mmr"
) -> pd.DataFrame:
    s = samples.dropna(subset=[lmax_col, "mae_pct"]).copy()
    s["bucket"] = s[lmax_col].apply(assign_bucket)
    grp = s.groupby(["side", "bucket"], dropna=False)
    agg = grp.agg(
        count=("mae_pct", "size"),
        mae_p50=("mae_pct", lambda x: np.nanquantile(x, 0.5)),
        mae_p90=("mae_pct", lambda x: np.nanquantile(x, 0.9)),
        mfe_p50=("mfe_pct", lambda x: np.nanquantile(x, 0.5)),
        mfe_p90=("mfe_pct", lambda x: np.nanquantile(x, 0.9)),
        mfe_mean=("mfe_pct", "mean"),
    ).reset_index()
    total_per_side = s.groupby("side").size().rename("side_total")
    agg = agg.merge(total_per_side, on="side", how="left")
    agg["share"] = agg["count"] / agg["side_total"]
    order = {label: i for i, (label, _, _) in enumerate(LEVERAGE_BUCKETS)}
    agg["_order"] = agg["bucket"].map(order).fillna(99).astype(int)
    agg = agg.sort_values(["side", "_order"]).drop(columns="_order")
    return agg


def feature_lift(
    samples: pd.DataFrame,
    target_bucket: str = ">=100x",
    n_quantiles: int = 10,
    lmax_col: str = "lmax_mmr",
) -> pd.DataFrame:
    s = samples.dropna(subset=[lmax_col]).copy()
    s["bucket"] = s[lmax_col].apply(assign_bucket)
    s["is_target"] = (s["bucket"] == target_bucket).astype(int)
    out_rows = []
    base_rate_per_side = s.groupby("side")["is_target"].mean().to_dict()
    for side in s["side"].unique():
        s_side = s[s["side"] == side]
        base = base_rate_per_side.get(side, float("nan"))
        for col in FEATURE_COLS:
            x = s_side[[col, "is_target"]].dropna()
            if len(x) < 200:
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
            g["lift"] = g["rate"] / base if base > 0 else np.nan
            g = g.reset_index()
            g.insert(0, "feature", col)
            g.insert(0, "side", side)
            g["base_rate"] = base
            out_rows.append(g)
    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)


def fit_tree_rules(
    samples: pd.DataFrame,
    target_bucket: str = ">=100x",
    max_depth: int = 4,
    lmax_col: str = "lmax_mmr",
) -> Dict[str, str]:
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
    except ImportError:
        return {"error": "sklearn not installed"}
    out = {}
    s = samples.dropna(subset=[lmax_col]).copy()
    s["bucket"] = s[lmax_col].apply(assign_bucket)
    s["is_target"] = (s["bucket"] == target_bucket).astype(int)
    for side in s["side"].unique():
        s_side = s[s["side"] == side].dropna(subset=FEATURE_COLS)
        if len(s_side) < 500:
            out[side] = "not enough samples"
            continue
        X = s_side[FEATURE_COLS].values
        y = s_side["is_target"].values
        if y.sum() < 50:
            out[side] = f"not enough positives ({int(y.sum())})"
            continue
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=max(50, int(len(y) * 0.005)),
            class_weight="balanced",
            random_state=0,
        )
        clf.fit(X, y)
        txt = export_text(clf, feature_names=FEATURE_COLS)
        # feature importance
        fi = pd.DataFrame(
            {"feature": FEATURE_COLS, "importance": clf.feature_importances_}
        ).sort_values("importance", ascending=False)
        top = fi.head(6).to_string(index=False)
        out[side] = f"Feature importance (top 6):\n{top}\n\nTree:\n{txt}"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Period filter
# ─────────────────────────────────────────────────────────────────────────────


def slice_period(
    df: pd.DataFrame, start: Optional[str], end: Optional[str]
) -> pd.DataFrame:
    if start is None and end is None:
        return df
    idx = df.index
    mask = pd.Series(True, index=idx)
    if start is not None:
        mask &= idx >= pd.Timestamp(start)
    if end is not None:
        mask &= idx <= pd.Timestamp(end)
    return df[mask]


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    symbols: List[str]
    timeframe: str
    horizons: List[int]
    start_date: Optional[str]
    end_date: Optional[str]
    data_path: str
    output_dir: Path
    mmr: float
    target_bucket: str


def run(cfg: RunConfig) -> None:
    handler = DataHandler(cfg.data_path)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    global_rows: List[pd.DataFrame] = []
    for sym in cfg.symbols:
        print(f"\n=== {sym} {cfg.timeframe} ===", flush=True)
        try:
            df_raw = handler.load_ohlcv(
                symbol=sym,
                timeframe=cfg.timeframe,
                start_date=cfg.start_date,
                end_date=cfg.end_date,
            )
        except Exception as e:
            print(f"  load failed: {e}")
            continue
        if df_raw is None or df_raw.empty:
            print("  empty")
            continue
        print(
            f"  loaded {len(df_raw)} bars: {df_raw.index.min()} -> {df_raw.index.max()}"
        )
        feats = compute_features(df_raw)
        sym_dir = cfg.output_dir / f"{sym}_{cfg.timeframe}"
        sym_dir.mkdir(parents=True, exist_ok=True)

        for H in cfg.horizons:
            samples = build_samples(feats, H, sym, mmr=cfg.mmr)
            samples_path = sym_dir / f"samples_H{H}.parquet"
            samples.reset_index().rename(columns={"index": "ts"}).to_parquet(
                samples_path, index=False
            )
            print(f"  H={H}: {len(samples)} samples -> {samples_path.name}")

            for period_name, (s, e) in DEFAULT_PERIODS.items():
                sub = slice_period(samples, s, e)
                if sub.empty:
                    continue
                bucket_df = summarize_buckets(sub, lmax_col="lmax_mmr")
                bucket_df.insert(0, "period", period_name)
                bucket_df.insert(0, "horizon", H)
                bucket_df.insert(0, "symbol", sym)
                bucket_df.to_csv(
                    sym_dir / f"bucket_counts_H{H}_{period_name}.csv", index=False
                )

                lift_df = feature_lift(sub, target_bucket=cfg.target_bucket)
                if not lift_df.empty:
                    lift_df.insert(0, "period", period_name)
                    lift_df.insert(0, "horizon", H)
                    lift_df.insert(0, "symbol", sym)
                    lift_df.to_csv(
                        sym_dir / f"feature_lift_H{H}_{period_name}.csv", index=False
                    )

                rules = fit_tree_rules(sub, target_bucket=cfg.target_bucket)
                with open(sym_dir / f"tree_rules_H{H}_{period_name}.md", "w") as f:
                    f.write(f"# {sym} {cfg.timeframe} H={H} period={period_name}\n\n")
                    f.write(f"target_bucket: {cfg.target_bucket}\n\n")
                    for side, txt in rules.items():
                        f.write(f"## side={side}\n\n```\n{txt}\n```\n\n")

                global_rows.append(bucket_df.copy())

    if global_rows:
        allg = pd.concat(global_rows, ignore_index=True)
        allg.to_csv(cfg.output_dir / "global_bucket_counts.csv", index=False)
        print(f"\nGlobal summary -> {cfg.output_dir / 'global_bucket_counts.csv'}")

        # cross-symbol aggregate
        agg = (
            allg.groupby(["horizon", "period", "side", "bucket"], dropna=False)
            .agg(
                count=("count", "sum"),
                mae_p50=("mae_p50", "mean"),
                mae_p90=("mae_p90", "mean"),
                mfe_p50=("mfe_p50", "mean"),
                mfe_p90=("mfe_p90", "mean"),
            )
            .reset_index()
        )
        totals = agg.groupby(["horizon", "period", "side"])["count"].transform("sum")
        agg["share"] = agg["count"] / totals
        order = {label: i for i, (label, _, _) in enumerate(LEVERAGE_BUCKETS)}
        agg["_order"] = agg["bucket"].map(order).fillna(99).astype(int)
        agg = agg.sort_values(["horizon", "period", "side", "_order"]).drop(
            columns="_order"
        )
        agg.to_csv(cfg.output_dir / "global_bucket_counts_agg.csv", index=False)
        print(
            f"Cross-symbol aggregate -> {cfg.output_dir / 'global_bucket_counts_agg.csv'}"
        )


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT",
        help="Comma-separated symbols",
    )
    p.add_argument("--timeframe", default="120T")
    p.add_argument("--horizons", default="12,48,120")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--data-path", default="data/parquet_data")
    p.add_argument("--output-dir", default="reports/leverage_capacity")
    p.add_argument("--mmr", type=float, default=MMR_DEFAULT)
    p.add_argument("--target-bucket", default=">=100x")
    args = p.parse_args()
    return RunConfig(
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        timeframe=args.timeframe,
        horizons=[int(x) for x in args.horizons.split(",") if x.strip()],
        start_date=args.start_date,
        end_date=args.end_date,
        data_path=args.data_path,
        output_dir=Path(args.output_dir),
        mmr=args.mmr,
        target_bucket=args.target_bucket,
    )


if __name__ == "__main__":
    run(parse_args())
