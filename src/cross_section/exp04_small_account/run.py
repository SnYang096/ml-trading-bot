"""exp04 入口：小资金（1w USD）专用的低换手 XS L/S 回测。

设计原则：
    - 候选池只取 20 个高流动性币
    - 同时 <=4 个持仓（2 多 2 空）
    - 默认 14 天 rebalance
    - 因子组合偏向 IC 证明可用的（low_vol + reversal；可选择加入动量）
    - 单腿 15% stop loss 兜底黑天鹅

用法：
    python -m src.cross_section.exp04_small_account.run \
        --start 2023-01 --end 2026-03

因子预设：
    --factor-preset ic_top   # (exp03 IC 最好的: low_vol + reversal)
    --factor-preset balanced # (均衡: low_vol + reversal + 小权重 momentum/funding)
    --factor-preset mom_only # (仅动量，对照)
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

from ..exp02_multi_factor.backtester import FactorSpec
from ..exp02_multi_factor.data_loader import build_panels
from ..exp02_multi_factor.sectors import get_sectors
from .backtester import SmallAccountConfig, run_small_account_backtest
from .config import (
    ACCOUNT_SIZE_USD,
    FEE_BPS_PER_SIDE,
    HOLD_BARS_DEFAULT,
    LIQUID_POOL,
    MAX_LONGS,
    MAX_SHORTS,
    MIN_NOTIONAL_USD,
    STOP_LOSS_PER_LEG,
)


PRESETS: Dict[str, List[FactorSpec]] = {
    "ic_top": [
        FactorSpec(name="low_vol_14d", kind="low_vol", lookback=24 * 14, weight=1.0),
        FactorSpec(name="low_vol_7d", kind="low_vol", lookback=24 * 7, weight=0.6),
        FactorSpec(name="rev_24h", kind="reversal", lookback=24, weight=0.6),
    ],
    "balanced": [
        FactorSpec(name="low_vol_14d", kind="low_vol", lookback=24 * 14, weight=1.0),
        FactorSpec(name="rev_24h", kind="reversal", lookback=24, weight=0.6),
        FactorSpec(name="mom_7d", kind="mom", lookback=24 * 7, weight=0.4),
        FactorSpec(name="funding_3d", kind="funding", lookback=24 * 3, weight=0.3),
    ],
    "mom_only": [
        FactorSpec(name="mom_7d", kind="mom", lookback=24 * 7, weight=1.0),
        FactorSpec(name="mom_14d", kind="mom", lookback=24 * 14, weight=0.5),
    ],
}


def try_plot_equity(eq: pd.DataFrame, btc: pd.Series, path: Path, title: str):
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        eq["equity_gross"].plot(ax=ax, label="Strategy Gross", alpha=0.6)
        eq["equity_net"].plot(ax=ax, label="Strategy Net")
        if not btc.empty:
            btc.reindex(eq.index).ffill().plot(
                ax=ax, label="BTC B&H", alpha=0.5, color="orange"
            )
        ax.axhline(1.0, color="k", lw=0.5)
        ax.set_title(title)
        ax.set_ylabel("Equity (start=1.0)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] plot failed: {e}")


def btc_benchmark(prices: pd.DataFrame) -> pd.Series:
    if "BTCUSDT" not in prices.columns:
        return pd.Series(dtype=float)
    p = prices["BTCUSDT"].dropna()
    return p / p.iloc[0]


def write_summary(
    outdir: Path,
    cfg: SmallAccountConfig,
    specs: List[FactorSpec],
    metrics: Dict,
    n_symbols: int,
    btc_stats: Dict,
    trades_df: pd.DataFrame,
):
    lines = [
        "# exp04 小资金版回测结果\n",
        "## 账户配置",
        f"- 账户规模: ${cfg.account_size_usd:,.0f}",
        f"- 最多持仓: {cfg.max_longs} 多 + {cfg.max_shorts} 空",
        f"- 单腿名义: ${cfg.account_size_usd * 0.25:.0f} (假设等权)",
        f"- 持仓周期: {cfg.hold_bars} bars ({cfg.hold_bars/24:.1f} 天)",
        f"- 单腿止损: {cfg.stop_loss_per_leg*100:.0f}%",
        f"- 费率(单边): {cfg.fee_bps_per_side} bps",
        f"- 板块中性: {cfg.sector_neutral}",
        f"- 候选池: {n_symbols} 个币",
        "",
        "## 因子",
    ]
    for s in specs:
        lines.append(f"- {s.name}: kind={s.kind}, lb={s.lookback}h, w={s.weight}")
    lines.append("\n## 绩效")
    for k in (
        "gross_sharpe",
        "net_sharpe",
        "gross_ann_return",
        "net_ann_return",
        "gross_max_dd",
        "net_max_dd",
        "gross_final_equity",
        "net_final_equity",
        "n_rebalances",
        "n_trades_total",
        "n_trades_stopped",
        "avg_notional_usd",
        "pct_feasible",
    ):
        v = metrics.get(k)
        if v is None:
            continue
        if isinstance(v, float):
            if "return" in k or "dd" in k:
                lines.append(f"- {k}: {v*100:+.2f}%")
            else:
                lines.append(f"- {k}: {v:.4f}")
        else:
            lines.append(f"- {k}: {v}")
    if btc_stats:
        lines.append("\n## 对比 BTC B&H (同期)")
        for k, v in btc_stats.items():
            suffix = "%" if any(x in k for x in ("return", "dd")) else ""
            val = v * 100 if suffix else v
            lines.append(f"- {k}: {val:+.2f}{suffix}")
    if not trades_df.empty:
        lines.append("\n## 交易明细统计")
        long_cnt = int((trades_df["side"] == "long").sum())
        short_cnt = int((trades_df["side"] == "short").sum())
        lines.append(
            f"- 总持仓次数: {len(trades_df)} ({long_cnt} long, {short_cnt} short)"
        )
        lines.append(f"- 触发止损次数: {int(trades_df['stopped_out'].sum())}")
        lines.append(f"- 持仓币种 top 10:")
        for sym, n in trades_df["symbol"].value_counts().head(10).items():
            lines.append(f"    - {sym}: {n} 次")
    lines.append("\n## 判读")
    lines.append(f"- Net Sharpe > 1 且 MaxDD < 20%: 小资金可考虑实盘")
    lines.append(f"- Gross 与 Net Sharpe 差距小: 换手设置合理")
    lines.append(f"- 止损触发率 < 10%: 止损阈值合适; > 30%: 应放宽或检查因子方向")
    lines.append(f"- pct_feasible = 100%: 所有单腿 >= {MIN_NOTIONAL_USD} USD 最小下单")
    (outdir / "summary.md").write_text("\n".join(lines))


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
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2026-03")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--price-dir", default="data/parquet_data")
    ap.add_argument("--funding-dir", default="data/funding_rate/parquet")
    ap.add_argument("--outdir", default="reports/cross_section/exp04")
    ap.add_argument("--symbols", nargs="*", default=None, help="默认 LIQUID_POOL 20 个")
    ap.add_argument("--hold-bars", type=int, default=HOLD_BARS_DEFAULT)
    ap.add_argument("--max-longs", type=int, default=MAX_LONGS)
    ap.add_argument("--max-shorts", type=int, default=MAX_SHORTS)
    ap.add_argument("--fee-bps", type=float, default=FEE_BPS_PER_SIDE)
    ap.add_argument("--stop-loss", type=float, default=STOP_LOSS_PER_LEG)
    ap.add_argument("--account-size", type=float, default=ACCOUNT_SIZE_USD)
    ap.add_argument("--no-sector-neutral", action="store_true")
    ap.add_argument("--factor-preset", choices=list(PRESETS.keys()), default="ic_top")
    ap.add_argument("--min-coverage", type=float, default=0.7)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    symbols = args.symbols if args.symbols else LIQUID_POOL
    print(f"[1/3] 加载数据: {len(symbols)} symbols {args.start}->{args.end}")
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
    prices = panels["prices"]
    returns = panels["returns"].fillna(0.0)
    funding = panels["funding"].fillna(0.0)
    print(f"      aligned: {prices.shape[0]} bars, {prices.shape[1]} symbols")

    specs = PRESETS[args.factor_preset]
    lookback_max = max(s.lookback + s.skip for s in specs)

    cfg = SmallAccountConfig(
        account_size_usd=args.account_size,
        max_longs=args.max_longs,
        max_shorts=args.max_shorts,
        hold_bars=args.hold_bars,
        fee_bps_per_side=args.fee_bps,
        stop_loss_per_leg=args.stop_loss,
        sector_neutral=not args.no_sector_neutral,
        lookback_max=lookback_max,
    )

    print(
        f"[2/3] 回测 (preset={args.factor_preset}, hold={args.hold_bars}h, "
        f"L/S={args.max_longs}+{args.max_shorts})"
    )
    eq, metrics, trades = run_small_account_backtest(returns, funding, specs, cfg)

    eq.to_parquet(outdir / "equity.parquet")
    if not trades.empty:
        trades.to_parquet(outdir / "trades.parquet")
    print(
        f"      Gross SR={metrics['gross_sharpe']:+.2f}  "
        f"Net SR={metrics['net_sharpe']:+.2f}  "
        f"AnnRet={metrics['net_ann_return']*100:+.1f}%  "
        f"MaxDD={metrics['net_max_dd']*100:.1f}%  "
        f"stopped={metrics.get('n_trades_stopped',0)}/{metrics.get('n_trades_total',0)}"
    )

    btc = btc_benchmark(prices)
    try_plot_equity(
        eq,
        btc,
        outdir / "equity.png",
        title=f"exp04 Small-Account ({args.factor_preset}, "
        f"L{args.max_longs}/S{args.max_shorts}, hold {args.hold_bars}h)",
    )

    print("[3/3] 写 summary")
    btc_stats = _btc_stats(prices)
    write_summary(outdir, cfg, specs, metrics, prices.shape[1], btc_stats, trades)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"完成。-> {outdir.resolve()}")


if __name__ == "__main__":
    main()
