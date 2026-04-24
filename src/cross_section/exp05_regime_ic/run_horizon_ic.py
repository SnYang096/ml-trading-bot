"""exp05 Part 1: Target-horizon IC 重算。

核心：exp03 只测了 24h horizon，但 exp04 的实际持仓是 14d。IC 应在目标 horizon 上测量。
这里对所有候选因子同时算 1d / 3d / 7d / 14d / 30d 五个 horizon 的 IC，
看因子排名是否随 horizon 改变。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from ..exp02_multi_factor import factors as F
from ..exp02_multi_factor.data_loader import build_panels
from ..exp02_multi_factor.sectors import SECTOR_MAP, cross_sectional_zscore
from ..exp03_ic_and_grid.ic_analysis import (
    factor_ic_series,
    forward_returns,
    ic_stats,
)
from ..exp04_small_account.config import LIQUID_POOL


CANDIDATES: List[Dict] = [
    {"name": "mom_3d", "kind": "mom", "lookback": 24 * 3, "skip": 0},
    {"name": "mom_7d", "kind": "mom", "lookback": 24 * 7, "skip": 0},
    {"name": "mom_14d", "kind": "mom", "lookback": 24 * 14, "skip": 0},
    {"name": "mom_30d", "kind": "mom", "lookback": 24 * 30, "skip": 0},
    {"name": "mom_30d_skip1d", "kind": "mom", "lookback": 24 * 30, "skip": 24},
    {"name": "rev_24h", "kind": "reversal", "lookback": 24, "skip": 0},
    {"name": "rev_3d", "kind": "reversal", "lookback": 24 * 3, "skip": 0},
    {"name": "fund_3d", "kind": "funding", "lookback": 24 * 3, "skip": 0},
    {"name": "fund_7d", "kind": "funding", "lookback": 24 * 7, "skip": 0},
    {"name": "low_vol_7d", "kind": "low_vol", "lookback": 24 * 7, "skip": 0},
    {"name": "low_vol_14d", "kind": "low_vol", "lookback": 24 * 14, "skip": 0},
    {"name": "low_vol_30d", "kind": "low_vol", "lookback": 24 * 30, "skip": 0},
]

HORIZONS_BARS = {"1d": 24, "3d": 72, "7d": 168, "14d": 336, "30d": 720}


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
    ap.add_argument(
        "--outdir", default="reports/cross_section/exp05_regime_ic/horizon_ic"
    )
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument(
        "--symbol-pool",
        choices=["liquid", "all"],
        default="liquid",
        help="liquid=20 liquid pool (exp04), all=65 symbols (exp02/03)",
    )
    ap.add_argument(
        "--sample-every",
        type=int,
        default=24,
        help="IC 采样间隔 bar 数；14d horizon 下不用太密",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    symbols = LIQUID_POOL if args.symbol_pool == "liquid" else sorted(SECTOR_MAP.keys())
    print(f"[1/3] 加载 {len(symbols)} symbols  {args.start}->{args.end}")
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

    # 预计算 forward returns
    fwds = {
        h_name: forward_returns(returns, h_bars)
        for h_name, h_bars in HORIZONS_BARS.items()
    }

    print(f"[2/3] 计算 {len(CANDIDATES)} 个因子 × {len(HORIZONS_BARS)} horizon IC")
    rows = []
    for cand in CANDIDATES:
        raw = _compute_raw(cand, returns, funding)
        raw_w = F.winsorize(raw, 0.02, 0.98)
        xs_z = cross_sectional_zscore(raw_w)
        for h_name, fw in fwds.items():
            ic_s = factor_ic_series(xs_z, fw, sample_every=args.sample_every)
            stats = ic_stats(ic_s)
            rows.append(
                {
                    "factor": cand["name"],
                    "kind": cand["kind"],
                    "lookback": cand["lookback"],
                    "skip": cand["skip"],
                    "horizon": h_name,
                    "horizon_bars": HORIZONS_BARS[h_name],
                    **stats,
                }
            )
        print(f"      - {cand['name']:<20s} done")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "ic_by_horizon.csv", index=False)

    # pivot: 每个因子 × horizon 的 IC mean / IR
    ic_mean_pvt = df.pivot_table(index="factor", columns="horizon", values="ic_mean")
    ic_mean_pvt = ic_mean_pvt[list(HORIZONS_BARS.keys())]
    ic_ir_pvt = df.pivot_table(index="factor", columns="horizon", values="ic_ir")
    ic_ir_pvt = ic_ir_pvt[list(HORIZONS_BARS.keys())]
    ic_mean_pvt.to_csv(outdir / "ic_mean_matrix.csv")
    ic_ir_pvt.to_csv(outdir / "ic_ir_matrix.csv")

    # heatmap
    try:
        import matplotlib.pyplot as plt

        for name, pvt in (("ic_mean", ic_mean_pvt), ("ic_ir", ic_ir_pvt)):
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(
                pvt.values,
                cmap="RdBu_r",
                aspect="auto",
                vmin=-pvt.abs().values.max(),
                vmax=pvt.abs().values.max(),
            )
            ax.set_xticks(range(len(pvt.columns)))
            ax.set_xticklabels(pvt.columns)
            ax.set_yticks(range(len(pvt.index)))
            ax.set_yticklabels(pvt.index)
            for i in range(pvt.shape[0]):
                for j in range(pvt.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{pvt.values[i, j]:+.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                    )
            ax.set_title(f"Factor {name} across horizons (xs_z, {args.symbol_pool})")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(outdir / f"{name}_heatmap.png", dpi=120)
            plt.close(fig)
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    # 14d 专门 ranking
    target = df[df["horizon"] == "14d"].sort_values("ic_ir", ascending=False)
    target.to_csv(outdir / "ranked_14d.csv", index=False)

    print("[3/3] 写 summary")
    lines = [
        "# exp05 Part 1 - Target-Horizon IC\n",
        f"- Period: {args.start} -> {args.end}",
        f"- Pool: {args.symbol_pool} ({returns.shape[1]} symbols)",
        f"- Bars: {returns.shape[0]}\n",
        "## 14d Horizon IC ranking (target horizon)\n",
    ]
    for _, r in target.head(10).iterrows():
        lines.append(
            f"- **{r['factor']}**: IC={r['ic_mean']:+.4f} IR={r['ic_ir']:+.3f} "
            f"hit={r['ic_hit_rate']:.2%}  n={r['n_samples']}"
        )
    lines.append("\n## IC mean across all horizons\n")
    lines.append(ic_mean_pvt.round(4).to_markdown())
    lines.append("\n## IC IR across all horizons\n")
    lines.append(ic_ir_pvt.round(3).to_markdown())
    lines.append("\n## 判读\n")
    lines.append(
        "- **IC 随 horizon 变化** = 因子对不同持仓周期作用不同。若 mom_7d 在 1d IC<0 但 14d IC>0.05，就说明它是 mid-term 因子。"
    )
    lines.append(
        "- **选因子时只看 target horizon 的 IC**：exp04 是 14d 持仓，则只按 14d IR 排序"
    )
    lines.append(
        "- **exp05 Part 2 (regime_ic)** 会进一步在 14d horizon 下按 regime 切分样本"
    )
    (outdir / "summary.md").write_text("\n".join(lines))
    print(f"完成 -> {outdir.resolve()}")


if __name__ == "__main__":
    main()
