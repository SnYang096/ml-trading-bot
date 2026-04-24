"""按 BTC 趋势 + funding regime 归因每次 rebalance 的 PnL。

输入：reports/cross_section/exp04_batch/<preset>__full/{trades.parquet, equity.parquet}
输出：reports/cross_section/exp04_batch/regime_attribution/
    - regime_labels.csv             # 每次 rebalance 时点的 regime 标签
    - regime_pnl_matrix.csv         # 行=preset, 列=regime, 值=累计 PnL / 次数 / 平均 PnL
    - regime_pnl.png                # 分组柱状
    - regime_attribution.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from ..exp02_multi_factor.data_loader import build_panels
from .config import LIQUID_POOL


def label_trend(btc_ret_30d: float) -> str:
    if btc_ret_30d > 0.10:
        return "bull"
    if btc_ret_30d < -0.10:
        return "bear"
    return "range"


def label_funding(funding_mean_7d: float) -> str:
    # Binance funding 8h 一次，正常区间约 ±0.01%。阈值用较高的 0.02% (=0.0002)
    if funding_mean_7d > 0.0002:
        return "long_crowd"
    if funding_mean_7d < -0.0001:
        return "short_crowd"
    return "normal"


def compute_regime_series(prices: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """返回每个时刻的 trend / funding / combined regime 标签。"""
    btc = prices["BTCUSDT"].dropna()
    btc_30d = btc.pct_change(24 * 30)
    trend = btc_30d.apply(label_trend).rename("trend")

    # 平均 funding（所有币的平均）
    fmean = funding.mean(axis=1)
    fmean_7d = fmean.rolling(24 * 7).mean()
    fund_lbl = fmean_7d.apply(label_funding).rename("funding")

    combined = (trend + "_" + fund_lbl).rename("combined")
    return pd.concat([trend, fund_lbl, combined], axis=1)


def _load_trades(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "trades.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def _rebalance_pnl(
    equity: pd.DataFrame, rebalance_times: List[pd.Timestamp]
) -> pd.Series:
    """给定 rebalance 时间序列，计算每段 [t_i, t_{i+1}] 的 port_ret_net 累计。"""
    pnl = {}
    rb = sorted(set(rebalance_times))
    for i in range(len(rb) - 1):
        t0 = rb[i]
        t1 = rb[i + 1]
        seg = equity.loc[(equity.index > t0) & (equity.index <= t1), "port_ret_net"]
        if len(seg) == 0:
            continue
        pnl[t0] = float(seg.sum())
    return pd.Series(pnl, name="period_pnl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", default="reports/cross_section/exp04_batch")
    ap.add_argument("--price-dir", default="data/parquet_data")
    ap.add_argument("--funding-dir", default="data/funding_rate/parquet")
    args = ap.parse_args()

    root = Path(args.batch_dir)
    outdir = root / "regime_attribution"
    outdir.mkdir(parents=True, exist_ok=True)

    # 加载价格/funding 全样本
    panels = build_panels(
        LIQUID_POOL,
        "2023-01",
        "2026-03",
        Path(args.price_dir),
        Path(args.funding_dir),
        "1h",
        0.5,
        verbose=False,
    )
    regimes = compute_regime_series(panels["prices"], panels["funding"].fillna(0.0))
    regimes.to_csv(outdir / "regime_series.csv")

    # 3 preset 的 full 期
    preset_dirs = {
        p.name.split("__")[0]: p
        for p in root.iterdir()
        if p.is_dir() and p.name.endswith("__full")
    }
    if not preset_dirs:
        raise SystemExit(f"no __full run dirs under {root}")

    attr_rows: List[Dict] = []
    for preset, pdir in preset_dirs.items():
        trades = _load_trades(pdir)
        if trades.empty:
            print(f"[skip] {preset}: no trades")
            continue
        trades["time"] = pd.to_datetime(trades["time"])
        equity = pd.read_parquet(pdir / "equity.parquet")

        reb_times = sorted(trades["time"].unique())
        pnl_ser = _rebalance_pnl(equity, reb_times)

        # 把 regime 标签贴到 rebalance 时点
        reg_at_rb = regimes.reindex(pnl_ser.index, method="ffill")
        df = pd.concat([pnl_ser, reg_at_rb], axis=1)
        df["preset"] = preset
        df.index.name = "rebalance_time"

        for _, grp in df.groupby(["trend", "funding"]):
            attr_rows.append(
                {
                    "preset": preset,
                    "trend": grp["trend"].iloc[0],
                    "funding": grp["funding"].iloc[0],
                    "n_periods": int(len(grp)),
                    "mean_pnl": float(grp["period_pnl"].mean()),
                    "total_pnl": float(grp["period_pnl"].sum()),
                    "hit_rate": float((grp["period_pnl"] > 0).mean()),
                    "std_pnl": float(grp["period_pnl"].std()),
                }
            )
        df.to_csv(outdir / f"{preset}_trades_with_regime.csv")

    attr = pd.DataFrame(attr_rows).sort_values(["preset", "trend", "funding"])
    attr.to_csv(outdir / "regime_pnl_matrix.csv", index=False)

    # 画图：trend × preset，mean_pnl
    try:
        import matplotlib.pyplot as plt

        # 只按 trend 聚合（更清晰）
        g = (
            attr.groupby(["preset", "trend"])
            .agg(mean_pnl=("mean_pnl", "mean"), n=("n_periods", "sum"))
            .reset_index()
        )
        pvt = g.pivot(index="trend", columns="preset", values="mean_pnl")
        pvt = pvt.reindex(["bull", "range", "bear"])
        fig, ax = plt.subplots(figsize=(9, 5))
        pvt.plot.bar(ax=ax)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title("Mean rebalance PnL by BTC 30d trend regime")
        ax.set_ylabel("Mean period PnL (fraction of capital)")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(outdir / "regime_pnl_trend.png", dpi=120)
        plt.close(fig)

        # funding regime
        g2 = (
            attr.groupby(["preset", "funding"])
            .agg(mean_pnl=("mean_pnl", "mean"))
            .reset_index()
        )
        pvt2 = g2.pivot(index="funding", columns="preset", values="mean_pnl")
        pvt2 = pvt2.reindex(["long_crowd", "normal", "short_crowd"])
        fig, ax = plt.subplots(figsize=(9, 5))
        pvt2.plot.bar(ax=ax)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title("Mean rebalance PnL by funding regime")
        ax.set_ylabel("Mean period PnL (fraction of capital)")
        fig.tight_layout()
        fig.savefig(outdir / "regime_pnl_funding.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    # 写 markdown
    lines = ["# exp04 Regime Attribution\n", "## Trend × preset (mean period PnL)\n"]
    t_pvt = attr.groupby(["preset", "trend"])["mean_pnl"].mean().unstack("trend")
    t_pvt = t_pvt.reindex(columns=["bull", "range", "bear"]) * 100
    lines.append(t_pvt.round(3).to_markdown() + "  (单位: %)\n")

    lines.append("\n## Funding × preset (mean period PnL %)\n")
    f_pvt = attr.groupby(["preset", "funding"])["mean_pnl"].mean().unstack("funding")
    f_pvt = f_pvt.reindex(columns=["long_crowd", "normal", "short_crowd"]) * 100
    lines.append(f_pvt.round(3).to_markdown() + "\n")

    lines.append("\n## Trend × Funding × preset (mean PnL %)\n")
    c_pvt = (
        attr.groupby(["preset", "trend", "funding"])["mean_pnl"]
        .mean()
        .unstack(["trend", "funding"])
        * 100
    )
    lines.append(c_pvt.round(3).to_markdown() + "\n")

    lines.append("\n## 样本数 (n_periods by regime)\n")
    n_pvt = attr.groupby(["trend", "funding"])["n_periods"].first().unstack("funding")
    lines.append(n_pvt.fillna(0).astype(int).to_markdown() + "\n")

    (outdir / "regime_attribution.md").write_text("\n".join(lines))
    print(f"完成 -> {outdir.resolve()}")


if __name__ == "__main__":
    main()
