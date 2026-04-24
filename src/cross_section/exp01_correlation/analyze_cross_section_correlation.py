"""
Cross-Section / Pairs Trading 可行性分析脚本
============================================

目的：分析多币种之间的相关性、协整性、横截面动量稳定性，判断 pairs trading
或 cross-sectional long-short 是否能作为"脱离 fat-tail 大行情"的稳定盈利来源。

分析内容：
    1. 对数收益率相关性矩阵（全样本 + 按年/按牛熊分段）
    2. 滚动相关性时间序列（观察相关性是否稳定）
    3. Engle-Granger 协整检验（pairs trading 的理论基础）
    4. 价差 Z-score 均值回归特征（半衰期 / Hurst 指数）
    5. 横截面动量简单回测（强者做多、弱者做空，市场中性）

数据源：data/parquet_data/<SYMBOL>_<YYYY-MM>.parquet（逐笔聚合）
重采样为 1H K线收盘价做分析。

用法：
    python -m src.cross_section.exp01_correlation.analyze_cross_section_correlation \
        --symbols BTCUSDT ETHUSDT SOLUSDT ADAUSDT XRPUSDT BNBUSDT \
        --start 2022-01 --end 2026-03 --timeframe 1H \
        --outdir reports/cross_section/exp01

输出：
    reports/cross_section/exp01/
        corr_full.csv                 # 全样本相关性矩阵
        corr_by_regime.csv            # 按年份/牛熊分段的相关性
        rolling_corr.csv + .png       # 滚动相关性
        cointegration.csv             # 协整检验结果（每对 pair）
        spread_halflife.csv           # 价差均值回归半衰期
        xs_momentum_backtest.csv+.png # 横截面动量净值曲线
        summary.md                    # 结论摘要
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def _month_range(start: str, end: str) -> List[str]:
    """生成 YYYY-MM 月份列表，闭区间。"""
    s = pd.Timestamp(start + "-01" if len(start) == 7 else start)
    e = pd.Timestamp(end + "-01" if len(end) == 7 else end)
    months = pd.date_range(s, e, freq="MS")
    return [m.strftime("%Y-%m") for m in months]


def load_symbol_bars(
    symbol: str,
    months: List[str],
    data_dir: Path,
    timeframe: str = "1h",
) -> Optional[pd.Series]:
    """加载单个 symbol 多月 tick，重采样为 close 价格序列。"""
    frames = []
    for ym in months:
        p = data_dir / f"{symbol}_{ym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["timestamp", "price"])
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df.set_index("timestamp").sort_index()
    # 重采样取 close
    close = df["price"].resample(timeframe).last().ffill()
    close.name = symbol
    return close


def build_price_panel(
    symbols: List[str],
    start: str,
    end: str,
    data_dir: Path,
    timeframe: str = "1h",
) -> pd.DataFrame:
    months = _month_range(start, end)
    series = {}
    for sym in symbols:
        s = load_symbol_bars(sym, months, data_dir, timeframe)
        if s is None:
            print(f"[WARN] {sym}: 无数据，跳过")
            continue
        series[sym] = s
        print(f"[OK]   {sym}: {len(s)} bars  {s.index.min()} -> {s.index.max()}")
    panel = pd.concat(series, axis=1).dropna(how="all")
    # 对齐：要求所有币都有值的时点
    panel = panel.dropna()
    return panel


# ---------------------------------------------------------------------------
# 1. 相关性分析
# ---------------------------------------------------------------------------


def corr_full(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def corr_by_regime(returns: pd.DataFrame, btc_col: str = "BTCUSDT") -> pd.DataFrame:
    """按年 + 按 BTC 趋势（上涨月/下跌月）分段，输出相关性均值表。"""
    rows = []
    # 按年
    for year, grp in returns.groupby(returns.index.year):
        c = grp.corr()
        rows.append(
            {
                "regime": f"year_{year}",
                "mean_corr": _mean_off_diag(c),
                **_flatten_corr(c),
            }
        )
    # 按 BTC 月度方向
    if btc_col in returns.columns:
        btc_m = returns[btc_col].resample("1M").sum()
        up_months = btc_m[btc_m > 0].index.to_period("M")
        dn_months = btc_m[btc_m <= 0].index.to_period("M")
        up_mask = returns.index.to_period("M").isin(up_months)
        dn_mask = returns.index.to_period("M").isin(dn_months)
        if up_mask.any():
            c = returns.loc[up_mask].corr()
            rows.append(
                {
                    "regime": "btc_bull_months",
                    "mean_corr": _mean_off_diag(c),
                    **_flatten_corr(c),
                }
            )
        if dn_mask.any():
            c = returns.loc[dn_mask].corr()
            rows.append(
                {
                    "regime": "btc_bear_months",
                    "mean_corr": _mean_off_diag(c),
                    **_flatten_corr(c),
                }
            )
    return pd.DataFrame(rows)


def _mean_off_diag(c: pd.DataFrame) -> float:
    arr = c.values.copy()
    np.fill_diagonal(arr, np.nan)
    return float(np.nanmean(arr))


def _flatten_corr(c: pd.DataFrame) -> Dict[str, float]:
    out = {}
    cols = c.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            out[f"{a}-{b}"] = float(c.loc[a, b])
    return out


def rolling_corr(returns: pd.DataFrame, window: int = 24 * 30) -> pd.DataFrame:
    """所有 pair 的滚动相关性（默认 30 天窗口，1H bar）。"""
    pairs = list(itertools.combinations(returns.columns, 2))
    out = {}
    for a, b in pairs:
        out[f"{a}-{b}"] = returns[a].rolling(window).corr(returns[b])
    return pd.DataFrame(out).dropna(how="all")


# ---------------------------------------------------------------------------
# 2. 协整 & 均值回归
# ---------------------------------------------------------------------------


def cointegration_table(prices: pd.DataFrame) -> pd.DataFrame:
    """对所有 pair 做 Engle-Granger 协整检验（基于对数价格）。"""
    from statsmodels.tsa.stattools import coint

    log_p = np.log(prices)
    rows = []
    for a, b in itertools.combinations(log_p.columns, 2):
        try:
            t_stat, p_value, crit = coint(log_p[a], log_p[b])
        except Exception as e:
            rows.append(
                {
                    "pair": f"{a}-{b}",
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "crit_5pct": np.nan,
                    "cointegrated_5pct": False,
                    "error": str(e),
                }
            )
            continue
        rows.append(
            {
                "pair": f"{a}-{b}",
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "crit_5pct": float(crit[1]),
                "cointegrated_5pct": bool(p_value < 0.05),
            }
        )
    return pd.DataFrame(rows).sort_values("p_value")


def spread_halflife(
    prices: pd.DataFrame, coint_df: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """对最显著协整的若干 pair 估计 OLS hedge ratio 并计算 AR(1) 半衰期。"""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    log_p = np.log(prices)
    rows = []
    for _, r in coint_df.head(top_n).iterrows():
        pair = r["pair"]
        a, b = pair.split("-")
        y = log_p[a]
        x = add_constant(log_p[b])
        try:
            res = OLS(y, x).fit()
            beta = float(res.params[b])
            alpha = float(res.params["const"])
            spread = y - (alpha + beta * log_p[b])
            # AR(1): spread_t = phi * spread_{t-1} + eps => half-life = -ln(2)/ln(phi)
            lag = spread.shift(1).dropna()
            cur = spread.loc[lag.index]
            phi_res = OLS(cur.values, add_constant(lag.values)).fit()
            phi = float(phi_res.params[1])
            if 0 < phi < 1:
                hl_bars = -np.log(2) / np.log(phi)
            else:
                hl_bars = np.nan
            rows.append(
                {
                    "pair": pair,
                    "alpha": alpha,
                    "beta_hedge": beta,
                    "spread_std": float(spread.std()),
                    "phi_ar1": phi,
                    "halflife_bars": float(hl_bars) if hl_bars == hl_bars else np.nan,
                    "halflife_hours": float(hl_bars) if hl_bars == hl_bars else np.nan,
                    "coint_p_value": float(r["p_value"]),
                }
            )
        except Exception as e:
            rows.append({"pair": pair, "error": str(e)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. 横截面动量回测（市场中性）
# ---------------------------------------------------------------------------


def cross_section_momentum_backtest(
    returns: pd.DataFrame,
    lookback_bars: int = 24 * 7,  # 7天动量
    hold_bars: int = 24,  # 持仓1天
    top_k: int = 2,
    bottom_k: int = 2,
    fee_bps: float = 5.0,  # 单边5bp（taker ~2bp, 滑点3bp）
) -> Tuple[pd.DataFrame, Dict]:
    """
    每 hold_bars 根 K 线 rebalance 一次：
        - 按过去 lookback_bars 累计收益排序
        - 做多 top_k，做空 bottom_k
        - 等权，保持美元中性
    返回：净值曲线 DataFrame，指标 dict
    """
    # 累计 lookback 动量
    mom = returns.rolling(lookback_bars).sum()
    reb_idx = returns.index[::hold_bars]
    port_ret_gross = pd.Series(0.0, index=returns.index)
    port_ret = pd.Series(0.0, index=returns.index)
    weights_log = []
    prev_w = pd.Series(0.0, index=returns.columns)

    for i in range(len(reb_idx) - 1):
        t0 = reb_idx[i]
        t1 = reb_idx[i + 1]
        snapshot = mom.loc[t0].dropna()
        if len(snapshot) < top_k + bottom_k:
            continue
        ranked = snapshot.sort_values(ascending=False)
        longs = ranked.head(top_k).index.tolist()
        shorts = ranked.tail(bottom_k).index.tolist()
        w = pd.Series(0.0, index=returns.columns)
        w[longs] = 0.5 / top_k
        w[shorts] = -0.5 / bottom_k
        weights_log.append({"time": t0, **w.to_dict()})
        seg = returns.loc[(returns.index > t0) & (returns.index <= t1)]
        seg_ret = seg.mul(w, axis=1).sum(axis=1)
        port_ret_gross.loc[seg.index] = seg_ret
        # 成本：只对 turnover 计费（|Δw|），单边 fee_bps
        turnover = float((w - prev_w).abs().sum())
        cost = turnover * fee_bps / 1e4
        port_ret.loc[seg.index] = seg_ret
        if len(seg.index) > 0:
            port_ret.loc[seg.index[0]] -= cost
        prev_w = w

    equity = (1 + port_ret).cumprod()
    equity_gross = (1 + port_ret_gross).cumprod()
    bars_per_year = 24 * 365
    ann_ret = port_ret.mean() * bars_per_year
    ann_vol = port_ret.std() * np.sqrt(bars_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    peak = equity.cummax()
    dd = (equity / peak - 1).min()

    ann_ret_g = port_ret_gross.mean() * bars_per_year
    ann_vol_g = port_ret_gross.std() * np.sqrt(bars_per_year)
    sharpe_g = ann_ret_g / ann_vol_g if ann_vol_g > 0 else np.nan

    metrics = {
        "ann_return_net": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe_net": float(sharpe),
        "ann_return_gross": float(ann_ret_g),
        "sharpe_gross": float(sharpe_g),
        "max_drawdown": float(dd),
        "final_equity_net": float(equity.iloc[-1]),
        "final_equity_gross": float(equity_gross.iloc[-1]),
        "n_rebalances": len(weights_log),
        "lookback_bars": lookback_bars,
        "hold_bars": hold_bars,
        "top_k": top_k,
        "bottom_k": bottom_k,
        "fee_bps_per_side": fee_bps,
    }
    out = pd.DataFrame(
        {
            "port_ret_gross": port_ret_gross,
            "port_ret_net": port_ret,
            "equity_gross": equity_gross,
            "equity_net": equity,
        }
    )
    return out, metrics


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------


def try_plot_rolling_corr(rc: pd.DataFrame, path: Path):
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        rc.plot(ax=ax, alpha=0.7)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title("Rolling Correlation (30d window)")
        ax.set_ylabel("Pearson r")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] plot rolling_corr failed: {e}")


def try_plot_equity(bt: pd.DataFrame, path: Path):
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        bt["equity_gross"].plot(ax=ax, label="Gross", alpha=0.6)
        bt["equity_net"].plot(ax=ax, label="Net of fees")
        ax.legend()
        ax.axhline(1.0, color="k", lw=0.5)
        ax.set_title("Cross-Section Momentum Equity Curve")
        ax.set_ylabel("Equity (start=1.0)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] plot equity failed: {e}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def write_summary(
    outdir: Path,
    corr_f: pd.DataFrame,
    coint_df: pd.DataFrame,
    hl_df: pd.DataFrame,
    metrics: Dict,
    regime_df: pd.DataFrame,
):
    lines = ["# Cross-Section / Pairs 可行性分析摘要\n"]
    lines.append("## 1. 全样本相关性\n")
    lines.append(f"- 平均非对角相关系数: **{_mean_off_diag(corr_f):.3f}**")
    lines.append(f"- 最高 pair: {_top_corr(corr_f, 3)}")
    lines.append(f"- 最低 pair: {_top_corr(corr_f, 3, ascending=True)}")
    lines.append("")
    lines.append("## 2. 不同市场状态下的平均相关性\n")
    if not regime_df.empty:
        for _, r in regime_df.iterrows():
            lines.append(f"- {r['regime']}: mean_corr = **{r['mean_corr']:.3f}**")
    lines.append("")
    lines.append("## 3. 协整 pair（p<0.05 表示可做 pairs trading）\n")
    sig = coint_df[coint_df["cointegrated_5pct"]]
    lines.append(f"- 显著协整 pair 数: **{len(sig)} / {len(coint_df)}**")
    if len(sig) > 0:
        for _, r in sig.head(10).iterrows():
            lines.append(f"  - {r['pair']}: p={r['p_value']:.4f}, t={r['t_stat']:.2f}")
    lines.append("")
    lines.append("## 4. 价差均值回归半衰期\n")
    if not hl_df.empty:
        for _, r in hl_df.head(10).iterrows():
            hl = r.get("halflife_hours", np.nan)
            beta = r.get("beta_hedge", np.nan)
            lines.append(f"- {r.get('pair','?')}: β={beta:.3f}, half-life≈{hl:.1f}h")
    lines.append("")
    lines.append("## 5. 横截面动量 L/S 回测（市场中性）\n")
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.4f}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## 结论建议\n")
    lines.append("- 若所有 pair 相关性>0.8 且牛熊均如此 → 市场共动，很难靠相对强弱赚钱")
    lines.append("- 若存在 p<0.05 协整 pair 且半衰期 < 48h → pairs trading 可行")
    lines.append(
        "- 若 XS 动量 Sharpe > 1 且回撤 < 15% → cross-sectional 策略值得继续开发"
    )
    lines.append("- 若相关性在牛市与熊市差异大 → 需用 regime-switching 模型")
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _top_corr(c: pd.DataFrame, n: int, ascending: bool = False) -> str:
    pairs = []
    cols = c.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            pairs.append((f"{a}-{b}", c.loc[a, b]))
    pairs.sort(key=lambda x: x[1], reverse=not ascending)
    return ", ".join(f"{p}({v:.2f})" for p, v in pairs[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "BNBUSDT"],
    )
    ap.add_argument("--start", default="2023-01", help="YYYY-MM")
    ap.add_argument("--end", default="2026-03", help="YYYY-MM")
    ap.add_argument(
        "--timeframe", default="1h", help="pandas resample freq, e.g. 1h/4h/1d"
    )
    ap.add_argument("--data-dir", default="data/parquet_data")
    ap.add_argument("--outdir", default="reports/cross_section/exp01")
    ap.add_argument("--rolling-window-bars", type=int, default=24 * 30)
    ap.add_argument("--mom-lookback-bars", type=int, default=24 * 7)
    ap.add_argument("--mom-hold-bars", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--bottom-k", type=int, default=2)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print(
        f"[1/6] 加载数据: {args.symbols}  {args.start} -> {args.end}  tf={args.timeframe}"
    )
    prices = build_price_panel(
        args.symbols, args.start, args.end, data_dir, args.timeframe
    )
    if prices.empty or prices.shape[1] < 2:
        raise SystemExit("数据不足，检查 symbol/月份范围")
    prices.to_csv(outdir / "prices.csv")
    returns = np.log(prices).diff().dropna()
    print(f"      对齐后: {prices.shape[0]} bars, {prices.shape[1]} symbols")

    print("[2/6] 全样本 & 分状态相关性")
    corr_f = corr_full(returns)
    corr_f.to_csv(outdir / "corr_full.csv")
    regime_df = corr_by_regime(returns)
    regime_df.to_csv(outdir / "corr_by_regime.csv", index=False)
    print(f"      mean off-diag corr = {_mean_off_diag(corr_f):.3f}")

    print(f"[3/6] 滚动相关性 window={args.rolling_window_bars} bars")
    rc = rolling_corr(returns, window=args.rolling_window_bars)
    rc.to_csv(outdir / "rolling_corr.csv")
    try_plot_rolling_corr(rc, outdir / "rolling_corr.png")

    print("[4/6] 协整检验 (Engle-Granger)")
    coint_df = cointegration_table(prices)
    coint_df.to_csv(outdir / "cointegration.csv", index=False)
    n_sig = int(coint_df["cointegrated_5pct"].sum())
    print(f"      显著协整 pair: {n_sig}/{len(coint_df)}")

    print("[5/6] 价差均值回归半衰期")
    hl_df = spread_halflife(prices, coint_df, top_n=min(10, len(coint_df)))
    hl_df.to_csv(outdir / "spread_halflife.csv", index=False)

    print("[6/6] 横截面动量 L/S 回测")
    bt, metrics = cross_section_momentum_backtest(
        returns,
        lookback_bars=args.mom_lookback_bars,
        hold_bars=args.mom_hold_bars,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        fee_bps=args.fee_bps,
    )
    bt.to_csv(outdir / "xs_momentum_backtest.csv")
    try_plot_equity(bt, outdir / "xs_momentum_backtest.png")
    print(
        f"      Gross Sharpe={metrics['sharpe_gross']:.2f} AnnRet={metrics['ann_return_gross']*100:.1f}%  |  "
        f"Net Sharpe={metrics['sharpe_net']:.2f} AnnRet={metrics['ann_return_net']*100:.1f}%  "
        f"MaxDD={metrics['max_drawdown']*100:.1f}%"
    )

    write_summary(outdir, corr_f, coint_df, hl_df, metrics, regime_df)
    print(f"\n完成。结果输出到: {outdir.resolve()}")
    print(f"  -> summary.md 包含结论建议")


if __name__ == "__main__":
    main()
