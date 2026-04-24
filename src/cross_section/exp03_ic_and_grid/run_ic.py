"""exp03-part1: 因子 IC 分析入口。

对每个候选因子（不同 lookback / kind）：
    - 计算 IC 时间序列（Spearman rank correlation）
    - 计算 IC 均值、波动、IR、命中率
    - 分 5 分位 forward return（看单调性）

同时对 panel 做横截面/板块中性化后再看一次 IC（看中性化是否还保留 alpha）。

用法：
    python -m src.cross_section.exp03_ic_and_grid.run_ic \
        --start 2023-01 --end 2026-03 \
        --outdir reports/cross_section/exp03/ic
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

from ..exp02_multi_factor import factors as F
from ..exp02_multi_factor.data_loader import build_panels
from ..exp02_multi_factor.sectors import (
    SECTOR_MAP,
    cross_sectional_zscore,
    get_sectors,
    sector_neutralize,
)
from .ic_analysis import (
    factor_ic_series,
    factor_quantile_returns,
    forward_returns,
    ic_stats,
)


CANDIDATES: List[Dict] = [
    # kind, lookback, skip
    {"name": "mom_24h", "kind": "mom", "lookback": 24, "skip": 0},
    {"name": "mom_3d", "kind": "mom", "lookback": 24 * 3, "skip": 0},
    {"name": "mom_7d", "kind": "mom", "lookback": 24 * 7, "skip": 0},
    {"name": "mom_14d", "kind": "mom", "lookback": 24 * 14, "skip": 0},
    {"name": "mom_30d_skip1d", "kind": "mom", "lookback": 24 * 30, "skip": 24},
    {"name": "rev_6h", "kind": "reversal", "lookback": 6, "skip": 0},
    {"name": "rev_24h", "kind": "reversal", "lookback": 24, "skip": 0},
    {"name": "fund_1d", "kind": "funding", "lookback": 24, "skip": 0},
    {"name": "fund_3d", "kind": "funding", "lookback": 24 * 3, "skip": 0},
    {"name": "fund_7d", "kind": "funding", "lookback": 24 * 7, "skip": 0},
    {"name": "low_vol_7d", "kind": "low_vol", "lookback": 24 * 7, "skip": 0},
    {"name": "low_vol_14d", "kind": "low_vol", "lookback": 24 * 14, "skip": 0},
]

HORIZONS = [24, 24 * 3, 24 * 7]  # 1d, 3d, 7d


def _compute_raw(
    cand: Dict, returns: pd.DataFrame, funding: pd.DataFrame
) -> pd.DataFrame:
    k = cand["kind"]
    lb = cand["lookback"]
    sk = cand["skip"]
    if k == "mom":
        return F.momentum(returns, lb, sk)
    if k == "reversal":
        return F.short_term_reversal(returns, lb)
    if k == "funding":
        return F.funding_factor(funding, lb)
    if k == "low_vol":
        return F.low_vol_factor(returns, lb)
    raise ValueError(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2026-03")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--price-dir", default="data/parquet_data")
    ap.add_argument("--funding-dir", default="data/funding_rate/parquet")
    ap.add_argument("--outdir", default="reports/cross_section/exp03/ic")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument(
        "--sample-every",
        type=int,
        default=24,
        help="IC 采样间隔（bar）。默认每天 1 次，足以统计",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    symbols = sorted(SECTOR_MAP.keys())
    print(f"[1/3] 加载数据: {len(symbols)} symbols  {args.start}->{args.end}")
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
    sectors = get_sectors(list(returns.columns))
    print(f"      aligned: {returns.shape[0]} bars, {returns.shape[1]} symbols")

    # 预计算各 horizon forward returns
    fwd = {h: forward_returns(returns, h) for h in HORIZONS}

    # 三种形态：raw / 跨截面 z / 板块中性 z
    results = []
    quantile_rows = []
    print(f"[2/3] 计算 {len(CANDIDATES)} 个候选因子 × {len(HORIZONS)} 个 horizon 的 IC")
    for cand in CANDIDATES:
        raw = _compute_raw(cand, returns, funding)
        raw_w = F.winsorize(raw, 0.02, 0.98)
        xs_z = cross_sectional_zscore(raw_w)
        sec_z = sector_neutralize(raw_w, sectors)
        for variant_name, variant_df in (
            ("raw", raw_w),
            ("xs_z", xs_z),
            ("sec_z", sec_z),
        ):
            for h, fw in fwd.items():
                ic = factor_ic_series(variant_df, fw, sample_every=args.sample_every)
                stats = ic_stats(ic)
                row = {
                    "factor": cand["name"],
                    "kind": cand["kind"],
                    "lookback": cand["lookback"],
                    "skip": cand["skip"],
                    "variant": variant_name,
                    "horizon_bars": h,
                    **stats,
                }
                results.append(row)
                # 只对 xs_z / horizon=24 做分位分析（最具代表性）
                if variant_name == "xs_z" and h == 24:
                    q = factor_quantile_returns(
                        variant_df, fw, n_q=5, sample_every=args.sample_every
                    )
                    if not q.empty:
                        for qi, qrow in q.iterrows():
                            quantile_rows.append(
                                {
                                    "factor": cand["name"],
                                    "quantile": qi,
                                    "mean_fwd_return": qrow["mean_fwd_return"],
                                    "count": qrow["count"],
                                }
                            )
        print(f"      - {cand['name']:<20s} done")

    ic_df = pd.DataFrame(results)
    ic_df.to_csv(outdir / "ic_all.csv", index=False)

    # 最佳因子（按 xs_z / horizon=24 的 IR 排序）
    best = ic_df[(ic_df["variant"] == "xs_z") & (ic_df["horizon_bars"] == 24)].copy()
    best = best.sort_values("ic_ir", ascending=False)
    best.to_csv(outdir / "ic_xs_z_h24_ranked.csv", index=False)

    q_df = pd.DataFrame(quantile_rows)
    if not q_df.empty:
        q_df.to_csv(outdir / "quantile_fwd_returns.csv", index=False)

    # 绘图：top/bottom 分位 spread
    try:
        import matplotlib.pyplot as plt

        if not q_df.empty:
            pivot = q_df.pivot_table(
                index="factor", columns="quantile", values="mean_fwd_return"
            )
            pivot["Q5-Q1"] = pivot[5] - pivot[1]
            pivot = pivot.sort_values("Q5-Q1", ascending=False)
            fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(pivot))))
            pivot["Q5-Q1"].plot.barh(ax=ax)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_title("Factor Q5-Q1 forward return spread (horizon=24h, xs_z)")
            fig.tight_layout()
            fig.savefig(outdir / "q5_q1_spread.png", dpi=120)
            plt.close(fig)
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    print("[3/3] 写入报告")
    lines = [
        "# exp03 Part 1 - Factor IC 分析\n",
        f"- Period: {args.start} -> {args.end}",
        f"- Symbols: {returns.shape[1]}  Bars: {returns.shape[0]}",
        f"- Horizon set: {HORIZONS}",
        "",
        "## Top 10 因子（按 xs_z / horizon=24h 的 IC IR 排序）\n",
    ]
    for _, r in best.head(10).iterrows():
        lines.append(
            f"- **{r['factor']}**: IC={r['ic_mean']:+.4f} "
            f"IR={r['ic_ir']:+.3f} hit={r['ic_hit_rate']:.2%}  (n={r['n_samples']})"
        )
    lines.append("\n## Bottom 5（IR 最差 / 方向可能反了）\n")
    for _, r in best.tail(5).iterrows():
        lines.append(f"- {r['factor']}: IC={r['ic_mean']:+.4f} IR={r['ic_ir']:+.3f}")
    lines.append("\n## 判读\n")
    lines.append("- IC mean > 0.02、IR > 0.3 的因子值得保留")
    lines.append("- IC 为负 -> 在 factors.py 里翻转符号（或从组合中删除）")
    lines.append("- 分位单调递增（Q1 < Q2 < ... < Q5）是最理想的因子形态")
    (outdir / "summary.md").write_text("\n".join(lines))

    print(f"完成。-> {outdir.resolve()}")
    print(f"  Top 3: {best.head(3)[['factor','ic_mean','ic_ir']].to_dict('records')}")


if __name__ == "__main__":
    main()
