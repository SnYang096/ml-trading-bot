"""exp02 入口：多因子 + 板块中性 横截面 L/S 回测。

用法示例：
    python -m src.cross_section.exp02_multi_factor.run \
        --start 2023-01 --end 2026-03 --timeframe 1h \
        --outdir reports/cross_section/exp02

默认 symbols = 65 个带 funding 的 USDT perp。也可通过 --symbols 限定。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from .backtester import BacktestConfig, FactorSpec, run_backtest
from .data_loader import build_panels
from .sectors import SECTOR_MAP, get_sectors


DEFAULT_SYMBOLS: List[str] = sorted(SECTOR_MAP.keys())


def default_factor_specs() -> List[FactorSpec]:
    """默认多因子组合。权重可调。"""
    return [
        FactorSpec(name="mom_7d", kind="mom", lookback=24 * 7, weight=1.0),
        FactorSpec(
            name="mom_30d_skip1d", kind="mom", lookback=24 * 30, skip=24, weight=0.5
        ),
        FactorSpec(name="reversal_1d", kind="reversal", lookback=24, weight=0.5),
        FactorSpec(name="funding_3d", kind="funding", lookback=24 * 3, weight=0.5),
        FactorSpec(name="low_vol_7d", kind="low_vol", lookback=24 * 7, weight=0.3),
    ]


def try_plot_equity(eq: pd.DataFrame, path: Path, title: str = ""):
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        eq["equity_gross"].plot(ax=ax, label="Gross", alpha=0.6)
        eq["equity_net"].plot(ax=ax, label="Net of fees")
        ax.axhline(1.0, color="k", lw=0.5)
        ax.set_title(title or "Multi-Factor XS L/S Equity")
        ax.set_ylabel("Equity (start=1.0)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] plot failed: {e}")


def plot_sector_exposure(weights: pd.DataFrame, sectors: pd.Series, path: Path):
    """绘制每个 rebalance 时点各板块的净暴露（应接近 0）。"""
    try:
        import matplotlib.pyplot as plt

        if weights.empty:
            return
        sec_exp = {}
        for sec in sectors.unique():
            cols = [c for c in weights.columns if sectors.get(c) == sec]
            if cols:
                sec_exp[sec] = weights[cols].sum(axis=1)
        df = pd.DataFrame(sec_exp)
        fig, ax = plt.subplots(figsize=(12, 5))
        df.plot(ax=ax, alpha=0.7)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title("Net Sector Exposure over time (should hover near 0)")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] sector exposure plot failed: {e}")


def btc_benchmark(prices: pd.DataFrame) -> pd.Series:
    if "BTCUSDT" in prices.columns:
        p = prices["BTCUSDT"].dropna()
        return p / p.iloc[0]
    return pd.Series(dtype=float)


def write_summary(
    outdir: Path,
    cfg: BacktestConfig,
    specs: List[FactorSpec],
    metrics: Dict,
    n_symbols: int,
    btc_stats: Dict,
):
    lines = ["# exp02 Multi-Factor XS L/S 回测结果\n"]
    lines.append("## 配置")
    lines.append(f"- symbols (aligned & kept): {n_symbols}")
    lines.append(f"- hold_bars: {cfg.hold_bars}")
    lines.append(f"- sector_neutral: {cfg.sector_neutral}")
    lines.append(
        f"- top_k/bottom_k: {cfg.top_k}/{cfg.bottom_k}  "
        f"(score_threshold={cfg.score_threshold})"
    )
    lines.append(f"- fee_bps_per_side: {cfg.fee_bps_per_side}")
    lines.append("")
    lines.append("## 因子")
    for s in specs:
        lines.append(
            f"- {s.name}: kind={s.kind}, lookback={s.lookback}, "
            f"skip={s.skip}, weight={s.weight}, vol_norm={s.vol_normalize}"
        )
    lines.append("")
    lines.append("## 绩效")
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.4f}")
        elif isinstance(v, list):
            lines.append(f"- {k}: {v}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")
    if btc_stats:
        lines.append("## 对比 BTC buy-and-hold（同期）")
        for k, v in btc_stats.items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("")
    lines.append("## 判读建议")
    lines.append("- Net Sharpe > 1.5 且 MaxDD < 15%: 策略已可部署小资金验证")
    lines.append("- Net Sharpe 0.8-1.5: 方向对，需调因子权重 / 交易频率 / 持仓数")
    lines.append("- Gross vs Net Sharpe 差距大: 换手过高，拉长 hold_bars 或降频")
    lines.append("- 板块净暴露持续偏离 0: 中性化失效，检查 sector 分类")
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _btc_stats(prices: pd.DataFrame) -> Dict:
    if "BTCUSDT" not in prices.columns:
        return {}
    r = np.log(prices["BTCUSDT"]).diff().dropna()
    bars_per_year = 24 * 365
    ann_r = float(r.mean() * bars_per_year)
    ann_v = float(r.std() * np.sqrt(bars_per_year))
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return {
        "btc_ann_return": ann_r,
        "btc_ann_vol": ann_v,
        "btc_sharpe": ann_r / ann_v if ann_v > 0 else np.nan,
        "btc_max_dd": dd,
        "btc_final_equity": float(eq.iloc[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="默认使用 sectors.SECTOR_MAP 全集（65 币种）",
    )
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2026-03")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--price-dir", default="data/parquet_data")
    ap.add_argument("--funding-dir", default="data/funding_rate/parquet")
    ap.add_argument("--outdir", default="reports/cross_section/exp02")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--hold-bars", type=int, default=24)
    ap.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="若给出则用 rank-based；否则用 score-weighted",
    )
    ap.add_argument("--bottom-k", type=int, default=None)
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--no-sector-neutral", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    symbols = args.symbols if args.symbols else DEFAULT_SYMBOLS
    print(f"[1/4] 加载面板数据: {len(symbols)} symbols  {args.start}->{args.end}")
    panels = build_panels(
        symbols=symbols,
        start=args.start,
        end=args.end,
        price_dir=Path(args.price_dir),
        funding_dir=Path(args.funding_dir),
        timeframe=args.timeframe,
        min_coverage=args.min_coverage,
    )
    prices = panels["prices"]
    returns = panels["returns"].fillna(0.0)
    funding = panels["funding"].fillna(0.0)
    print(f"      对齐后: {prices.shape[0]} bars, {prices.shape[1]} symbols")
    prices.to_parquet(outdir / "prices.parquet")
    returns.to_parquet(outdir / "returns.parquet")
    funding.to_parquet(outdir / "funding.parquet")

    print("[2/4] 构建因子 & 复合分数")
    specs = default_factor_specs()
    lookback_max = max(s.lookback + s.skip for s in specs)

    cfg = BacktestConfig(
        lookback_max=lookback_max,
        hold_bars=args.hold_bars,
        top_k=args.top_k,
        bottom_k=args.bottom_k if args.bottom_k is not None else args.top_k,
        score_threshold=args.score_threshold,
        fee_bps_per_side=args.fee_bps,
        sector_neutral=not args.no_sector_neutral,
    )

    print("[3/4] 回测中 ...")
    eq, metrics, weights = run_backtest(returns, funding, specs, cfg)
    eq.to_parquet(outdir / "equity.parquet")
    if not weights.empty:
        weights.to_parquet(outdir / "weights.parquet")

    print(
        f"      Gross Sharpe={metrics['gross_sharpe']:.2f}  "
        f"AnnRet={metrics['gross_ann_return']*100:.1f}%  | "
        f"Net Sharpe={metrics['net_sharpe']:.2f}  "
        f"AnnRet={metrics['net_ann_return']*100:.1f}%  "
        f"MaxDD={metrics['net_max_dd']*100:.1f}%"
    )

    try_plot_equity(
        eq,
        outdir / "equity.png",
        title=f"exp02 Multi-Factor XS L/S  (symbols={prices.shape[1]})",
    )
    sectors = get_sectors(list(returns.columns))
    plot_sector_exposure(weights, sectors, outdir / "sector_exposure.png")

    print("[4/4] 写入 summary.md")
    btc_stats = _btc_stats(prices)
    write_summary(outdir, cfg, specs, metrics, prices.shape[1], btc_stats)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    print(f"\n完成。结果 -> {outdir.resolve()}")


if __name__ == "__main__":
    main()
