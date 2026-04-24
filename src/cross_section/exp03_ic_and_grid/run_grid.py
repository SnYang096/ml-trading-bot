"""exp03-part2: Grid Search 配置对比。

在固定的因子组合上，遍历 (hold_bars, top_k, sector_neutral) 等配置，
输出一张对比表，定位"最优换手率+持仓数"组合。

用法：
    python -m src.cross_section.exp03_ic_and_grid.run_grid \
        --start 2023-01 --end 2026-03 \
        --outdir reports/cross_section/exp03/grid
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from ..exp02_multi_factor.backtester import BacktestConfig, FactorSpec, run_backtest
from ..exp02_multi_factor.data_loader import build_panels
from ..exp02_multi_factor.sectors import SECTOR_MAP


def default_specs() -> List[FactorSpec]:
    return [
        FactorSpec(name="mom_7d", kind="mom", lookback=24 * 7, weight=1.0),
        FactorSpec(
            name="mom_30d_skip1d", kind="mom", lookback=24 * 30, skip=24, weight=0.5
        ),
        FactorSpec(name="reversal_1d", kind="reversal", lookback=24, weight=0.5),
        FactorSpec(name="funding_3d", kind="funding", lookback=24 * 3, weight=0.5),
        FactorSpec(name="low_vol_7d", kind="low_vol", lookback=24 * 7, weight=0.3),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2026-03")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--price-dir", default="data/parquet_data")
    ap.add_argument("--funding-dir", default="data/funding_rate/parquet")
    ap.add_argument("--outdir", default="reports/cross_section/exp03/grid")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    # grid
    ap.add_argument(
        "--hold-bars-grid", nargs="+", type=int, default=[24, 72, 168, 336]
    )  # 1d, 3d, 7d, 14d
    ap.add_argument("--topk-grid", nargs="+", type=int, default=[3, 5, 8, 12])
    ap.add_argument("--sector-neutral-grid", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    symbols = sorted(SECTOR_MAP.keys())
    print(f"[1/3] 加载面板: {len(symbols)} symbols {args.start}->{args.end}")
    panels = build_panels(
        symbols,
        args.start,
        args.end,
        Path(args.price_dir),
        Path(args.funding_dir),
        args.timeframe,
        args.min_coverage,
        verbose=False,
    )
    returns = panels["returns"].fillna(0.0)
    funding = panels["funding"].fillna(0.0)
    print(f"      aligned: {returns.shape[0]} bars, {returns.shape[1]} symbols")

    specs = default_specs()
    lookback_max = max(s.lookback + s.skip for s in specs)

    combos = list(
        itertools.product(args.hold_bars_grid, args.topk_grid, args.sector_neutral_grid)
    )
    print(f"[2/3] 跑 {len(combos)} 组配置")
    rows = []
    equity_store: Dict[str, pd.Series] = {}
    for hb, tk, sn in combos:
        cfg = BacktestConfig(
            lookback_max=lookback_max,
            hold_bars=hb,
            top_k=tk,
            bottom_k=tk,
            fee_bps_per_side=args.fee_bps,
            sector_neutral=bool(sn),
        )
        eq, m, _ = run_backtest(returns, funding, specs, cfg)
        tag = f"hb{hb}_k{tk}_{'sec' if sn else 'raw'}"
        rows.append(
            {
                "tag": tag,
                "hold_bars": hb,
                "top_k": tk,
                "sector_neutral": bool(sn),
                "gross_sharpe": m["gross_sharpe"],
                "net_sharpe": m["net_sharpe"],
                "gross_ann_return": m["gross_ann_return"],
                "net_ann_return": m["net_ann_return"],
                "net_max_dd": m["net_max_dd"],
                "n_rebalances": m["n_rebalances"],
            }
        )
        equity_store[tag] = eq["equity_net"]
        print(
            f"  {tag:<22s} | Gross SR={m['gross_sharpe']:+.2f} "
            f"Net SR={m['net_sharpe']:+.2f} "
            f"AnnRet={m['net_ann_return']*100:+.1f}% DD={m['net_max_dd']*100:.1f}%"
        )

    df = pd.DataFrame(rows).sort_values("net_sharpe", ascending=False)
    df.to_csv(outdir / "grid_results.csv", index=False)

    # 绘图：top 5 Net equity curves
    try:
        import matplotlib.pyplot as plt

        top5 = df.head(5)["tag"].tolist()
        fig, ax = plt.subplots(figsize=(12, 6))
        for tag in top5:
            equity_store[tag].plot(ax=ax, label=tag)
        ax.axhline(1.0, color="k", lw=0.5)
        ax.set_title("Top 5 configurations (Net equity)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "top5_equity.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    print("[3/3] 写 summary")
    lines = [
        "# exp03 Part 2 - Grid Search 结果\n",
        f"- Period: {args.start} -> {args.end}  ({returns.shape[1]} symbols)\n",
        f"- Fee: {args.fee_bps} bps/side  |  Factors: default (7d+30d mom, 1d rev, 3d fund, 7d low-vol)\n",
        "## Top 10 by Net Sharpe\n",
        "| tag | hold_bars | top_k | sec_neutral | gross_SR | net_SR | ann_ret | MaxDD |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.head(10).iterrows():
        lines.append(
            f"| `{r['tag']}` | {r['hold_bars']} | {r['top_k']} | {r['sector_neutral']} "
            f"| {r['gross_sharpe']:+.2f} | **{r['net_sharpe']:+.2f}** "
            f"| {r['net_ann_return']*100:+.1f}% | {r['net_max_dd']*100:.1f}% |"
        )
    lines.append("\n## Bottom 5（避免使用）\n")
    for _, r in df.tail(5).iterrows():
        lines.append(f"- `{r['tag']}`: net_SR={r['net_sharpe']:+.2f}")
    lines.append("\n## 判读\n")
    lines.append("- hold_bars 增大通常提升 Net Sharpe（降换手）")
    lines.append("- top_k 过小（2-3）单币风险大；过大（>12）alpha 被稀释")
    lines.append("- sector_neutral 在强单边年份可能反而减收益，多年样本看应提升稳定性")
    (outdir / "summary.md").write_text("\n".join(lines))
    print(f"完成。Best: {df.iloc[0]['tag']}  Net SR={df.iloc[0]['net_sharpe']:+.2f}")
    print(f"-> {outdir.resolve()}")


if __name__ == "__main__":
    main()
