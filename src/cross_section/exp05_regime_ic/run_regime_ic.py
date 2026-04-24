"""exp05 Part 2: Regime-Conditional IC + regime_weights.yaml 生成。

对每个 (factor, regime)：
    - 只取该 regime 下的时间样本
    - 计算 14d-horizon IC 均值 / IR
输出每个 regime 的最优因子权重（按 IC 正比分配，负 IC 设 0）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml

from ..exp02_multi_factor import factors as F
from ..exp02_multi_factor.data_loader import build_panels
from ..exp02_multi_factor.sectors import cross_sectional_zscore
from ..exp03_ic_and_grid.ic_analysis import factor_ic_series, forward_returns, ic_stats
from ..exp04_small_account.config import LIQUID_POOL
from .regimes import COLLAPSED_REGIMES, compute_regime_labels
from .run_horizon_ic import CANDIDATES, _compute_raw


TARGET_HORIZON_BARS = 336  # 14d
MIN_REGIME_SAMPLES = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2026-03")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--price-dir", default="data/parquet_data")
    ap.add_argument("--funding-dir", default="data/funding_rate/parquet")
    ap.add_argument(
        "--outdir", default="reports/cross_section/exp05_regime_ic/regime_ic"
    )
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--sample-every", type=int, default=24)
    ap.add_argument(
        "--ic-threshold",
        type=float,
        default=0.02,
        help="IC 均值低于此值的因子不进入权重",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 加载数据")
    panels = build_panels(
        LIQUID_POOL,
        args.start,
        args.end,
        Path(args.price_dir),
        Path(args.funding_dir),
        args.timeframe,
        args.min_coverage,
        verbose=False,
    )
    prices = panels["prices"]
    returns = panels["returns"].fillna(0.0)
    funding = panels["funding"].fillna(0.0)
    print(f"      aligned: {returns.shape[0]} bars, {returns.shape[1]} symbols")

    print("[2/4] 打 regime 标签")
    regimes = compute_regime_labels(prices, funding)
    regimes.to_csv(outdir / "regimes_timeseries.csv")
    regime_counts = regimes["collapsed"].value_counts()
    print(f"      regime 分布（bars）:\n{regime_counts}")

    fwd = forward_returns(returns, TARGET_HORIZON_BARS)

    print(f"[3/4] 每 (factor, regime) 计算 14d-horizon IC")
    rows = []
    for cand in CANDIDATES:
        raw = _compute_raw(cand, returns, funding)
        raw_w = F.winsorize(raw, 0.02, 0.98)
        xs_z = cross_sectional_zscore(raw_w)

        # 全样本 IC 作基准
        ic_all = factor_ic_series(xs_z, fwd, sample_every=args.sample_every)
        all_stats = ic_stats(ic_all)
        rows.append({"factor": cand["name"], "regime": "ALL", **all_stats})

        for reg in COLLAPSED_REGIMES:
            reg_mask = regimes["collapsed"] == reg
            if reg_mask.sum() < MIN_REGIME_SAMPLES:
                rows.append(
                    {
                        "factor": cand["name"],
                        "regime": reg,
                        "ic_mean": np.nan,
                        "ic_std": np.nan,
                        "ic_ir": np.nan,
                        "ic_hit_rate": np.nan,
                        "n_samples": int(reg_mask.sum()),
                    }
                )
                continue
            xs_z_reg = xs_z.loc[reg_mask]
            fwd_reg = fwd.loc[reg_mask]
            ic_reg = factor_ic_series(xs_z_reg, fwd_reg, sample_every=args.sample_every)
            s = ic_stats(ic_reg)
            rows.append({"factor": cand["name"], "regime": reg, **s})
        print(f"      - {cand['name']:<20s} done")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "regime_ic_long.csv", index=False)

    # pivot matrix
    ic_mean_mat = df.pivot_table(index="factor", columns="regime", values="ic_mean")
    regime_order = ["ALL"] + COLLAPSED_REGIMES
    ic_mean_mat = ic_mean_mat[[c for c in regime_order if c in ic_mean_mat.columns]]
    ic_ir_mat = df.pivot_table(index="factor", columns="regime", values="ic_ir")
    ic_ir_mat = ic_ir_mat[[c for c in regime_order if c in ic_ir_mat.columns]]
    ic_mean_mat.to_csv(outdir / "regime_ic_mean_matrix.csv")
    ic_ir_mat.to_csv(outdir / "regime_ic_ir_matrix.csv")

    # 构建 regime_weights：对每个 regime 选出 IC > threshold 的因子，按 IC 正比分配权重
    weights_yaml: Dict[str, Dict] = {}
    for reg in regime_order:
        if reg not in ic_mean_mat.columns:
            continue
        col = ic_mean_mat[reg].dropna()
        eligible = col[col > args.ic_threshold]
        if eligible.empty:
            # fallback: 取整体最佳（ALL 列的 top 3 正 IC）
            fallback = ic_mean_mat["ALL"].dropna().sort_values(ascending=False)
            eligible = fallback[fallback > args.ic_threshold].head(3)
            note = "fallback_to_ALL_top3"
        else:
            note = "regime_conditional"
        total = float(eligible.sum())
        fac_weights = {f: round(float(v / total), 4) for f, v in eligible.items()}
        weights_yaml[reg] = {
            "factors": fac_weights,
            "n_factors": len(fac_weights),
            "note": note,
        }

    # 每个 factor 的 kind/lookback（paper + exp05 backtester 需要）
    factor_specs = {
        c["name"]: {"kind": c["kind"], "lookback": c["lookback"], "skip": c["skip"]}
        for c in CANDIDATES
    }

    out_yaml = {
        "meta": {
            "horizon_bars": TARGET_HORIZON_BARS,
            "ic_threshold": args.ic_threshold,
            "min_regime_samples": MIN_REGIME_SAMPLES,
            "period": f"{args.start}_to_{args.end}",
            "symbols": LIQUID_POOL,
        },
        "factor_specs": factor_specs,
        "regime_weights": weights_yaml,
    }
    (outdir / "regime_weights.yaml").write_text(
        yaml.safe_dump(out_yaml, sort_keys=False, allow_unicode=True)
    )

    # heatmap
    try:
        import matplotlib.pyplot as plt

        vmax = ic_mean_mat.abs().values.max()
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(
            ic_mean_mat.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax
        )
        ax.set_xticks(range(len(ic_mean_mat.columns)))
        ax.set_xticklabels(ic_mean_mat.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(ic_mean_mat.index)))
        ax.set_yticklabels(ic_mean_mat.index)
        for i in range(ic_mean_mat.shape[0]):
            for j in range(ic_mean_mat.shape[1]):
                v = ic_mean_mat.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"Regime-conditional IC mean (horizon=14d)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(outdir / "regime_ic_heatmap.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    print("[4/4] 写 summary")
    lines = [
        "# exp05 Part 2 - Regime-Conditional IC (14d horizon)\n",
        f"- Period: {args.start} -> {args.end}",
        f"- Horizon: {TARGET_HORIZON_BARS} bars (14d)",
        f"- IC threshold for weight inclusion: {args.ic_threshold}",
        f"- Min samples per regime: {MIN_REGIME_SAMPLES}\n",
        "## Regime 分布（bars）\n",
        regime_counts.to_markdown(),
        "",
        "## IC mean matrix\n",
        ic_mean_mat.round(3).to_markdown(),
        "",
        "## IC IR matrix\n",
        ic_ir_mat.round(3).to_markdown(),
        "",
        "## 生成的 regime_weights\n",
    ]
    for reg, w in weights_yaml.items():
        lines.append(f"### {reg}  ({w['note']})")
        for f, v in w["factors"].items():
            lines.append(f"- {f}: {v}")
        lines.append("")
    (outdir / "summary.md").write_text("\n".join(lines))
    print(f"完成 -> {outdir.resolve()}")
    print(f"regime_weights.yaml 包含 {len(weights_yaml)} 个 regime")


if __name__ == "__main__":
    main()
