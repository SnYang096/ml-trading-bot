"""exp04 批量回测：3 presets x 5 periods。

直接调用 run_small_account_backtest 避免每次 build_panels 重复加载。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..exp02_multi_factor.data_loader import build_panels
from .backtester import SmallAccountConfig, run_small_account_backtest
from .config import (
    ACCOUNT_SIZE_USD,
    FEE_BPS_PER_SIDE,
    HOLD_BARS_DEFAULT,
    LIQUID_POOL,
    MAX_LONGS,
    MAX_SHORTS,
    STOP_LOSS_PER_LEG,
)
from .run import PRESETS


PERIODS: List[Tuple[str, str, str]] = [
    ("2023_range", "2023-01", "2023-12"),
    ("2024_bull", "2024-01", "2024-12"),
    ("2025_mix", "2025-01", "2025-12"),
    ("2026_q1", "2026-01", "2026-03"),
    ("full", "2023-01", "2026-03"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-dir", default="data/parquet_data")
    ap.add_argument("--funding-dir", default="data/funding_rate/parquet")
    ap.add_argument("--outdir", default="reports/cross_section/exp04_batch")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--hold-bars", type=int, default=HOLD_BARS_DEFAULT)
    ap.add_argument("--presets", nargs="+", default=list(PRESETS.keys()))
    args = ap.parse_args()

    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)

    # 一次性加载 full 范围数据，后续按 period 切片
    print(f"[load] {len(LIQUID_POOL)} symbols  full range 2023-01 -> 2026-03")
    panels = build_panels(
        LIQUID_POOL,
        "2023-01",
        "2026-03",
        Path(args.price_dir),
        Path(args.funding_dir),
        args.timeframe,
        args.min_coverage,
        verbose=False,
    )
    prices = panels["prices"]
    returns_full = panels["returns"].fillna(0.0)
    funding_full = panels["funding"].fillna(0.0)
    print(f"[load] aligned: {prices.shape[0]} bars, {prices.shape[1]} symbols")

    rows: List[Dict] = []
    for preset_name in args.presets:
        specs = PRESETS[preset_name]
        lookback_max = max(s.lookback + s.skip for s in specs)
        for period_name, start, end in PERIODS:
            tag = f"{preset_name}__{period_name}"
            subdir = outroot / tag
            subdir.mkdir(parents=True, exist_ok=True)

            t0 = pd.Timestamp(start + "-01")
            t1 = pd.Timestamp(end + "-01") + pd.offsets.MonthEnd(0)
            ret = returns_full.loc[t0:t1]
            fund = funding_full.loc[t0:t1]
            if len(ret) < lookback_max + 24 * 14:
                print(f"[skip] {tag}: samples {len(ret)} < needed")
                continue

            cfg = SmallAccountConfig(
                account_size_usd=ACCOUNT_SIZE_USD,
                max_longs=MAX_LONGS,
                max_shorts=MAX_SHORTS,
                hold_bars=args.hold_bars,
                fee_bps_per_side=FEE_BPS_PER_SIDE,
                stop_loss_per_leg=STOP_LOSS_PER_LEG,
                sector_neutral=True,
                lookback_max=lookback_max,
            )
            eq, m, trades = run_small_account_backtest(ret, fund, specs, cfg)
            eq.to_parquet(subdir / "equity.parquet")
            if not trades.empty:
                trades.to_parquet(subdir / "trades.parquet")
            (subdir / "metrics.json").write_text(json.dumps(m, indent=2, default=str))

            row = {
                "preset": preset_name,
                "period": period_name,
                "start": start,
                "end": end,
                "gross_sharpe": m["gross_sharpe"],
                "net_sharpe": m["net_sharpe"],
                "gross_ann_return": m["gross_ann_return"],
                "net_ann_return": m["net_ann_return"],
                "net_max_dd": m["net_max_dd"],
                "n_rebalances": m["n_rebalances"],
                "n_trades_total": m.get("n_trades_total", 0),
                "n_trades_stopped": m.get("n_trades_stopped", 0),
                "stop_rate": (
                    m.get("n_trades_stopped", 0) / max(m.get("n_trades_total", 1), 1)
                ),
            }
            rows.append(row)
            print(
                f"  {tag:<32s} | NetSR={m['net_sharpe']:+.2f} "
                f"AnnRet={m['net_ann_return']*100:+.1f}% "
                f"DD={m['net_max_dd']*100:.1f}% "
                f"stopped={row['n_trades_stopped']}/{row['n_trades_total']}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(outroot / "summary.csv", index=False)

    # 写 markdown summary
    lines = [
        "# exp04 全样本批量回测 summary\n",
        f"Periods: {[p[0] for p in PERIODS]}",
        f"Presets: {args.presets}",
        f"Hold: {args.hold_bars} bars ({args.hold_bars/24:.1f} days)",
        f"Symbols kept: {prices.shape[1]}\n",
        "## Net Sharpe matrix\n",
    ]
    pvt = df.pivot_table(index="preset", columns="period", values="net_sharpe")
    # 保持 period 顺序
    col_order = [p[0] for p in PERIODS if p[0] in pvt.columns]
    pvt = pvt[col_order]
    lines.append(pvt.round(2).to_markdown())
    lines.append("\n## Net AnnReturn matrix (%)\n")
    pvt2 = (
        df.pivot_table(index="preset", columns="period", values="net_ann_return")[
            col_order
        ]
        * 100
    )
    lines.append(pvt2.round(1).to_markdown())
    lines.append("\n## MaxDD matrix (%)\n")
    pvt3 = (
        df.pivot_table(index="preset", columns="period", values="net_max_dd")[col_order]
        * 100
    )
    lines.append(pvt3.round(1).to_markdown())
    lines.append("\n## Stop loss rate\n")
    pvt4 = df.pivot_table(index="preset", columns="period", values="stop_rate")[
        col_order
    ]
    lines.append(pvt4.round(3).to_markdown())
    (outroot / "summary.md").write_text("\n".join(lines))
    print(f"\n完成 -> {outroot.resolve()}")


if __name__ == "__main__":
    main()
