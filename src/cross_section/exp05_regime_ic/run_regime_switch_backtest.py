"""exp05 Part 3: Regime-Switching 回测。

每次 rebalance 时点：
    1. 读取当前时刻的 regime 标签 (collapsed)
    2. 从 regime_weights.yaml 取该 regime 的 factor weights
    3. 用这些 weights 构建 composite score → 决策多空

对比基线：
    - static: exp04 的 mom_only preset（2024 最佳）
    - static_all: 用 regime_weights.yaml['ALL'] 的权重（全样本最优权重）
    - regime_switch: 每 rebalance 动态切换权重

NOTE: 存在轻度 look-ahead 偏差（regime_weights.yaml 是用 full sample 学出来的）。
这是 POC；下一阶段应做 walk-forward out-of-sample 训练权重。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml

from ..exp02_multi_factor import factors as F
from ..exp02_multi_factor.backtester import FactorSpec, build_composite_score
from ..exp02_multi_factor.data_loader import build_panels
from ..exp02_multi_factor.sectors import cross_sectional_zscore, get_sectors
from ..exp04_small_account.backtester import SmallAccountConfig, _metrics
from ..exp04_small_account.config import (
    ACCOUNT_SIZE_USD,
    FEE_BPS_PER_SIDE,
    HOLD_BARS_DEFAULT,
    LIQUID_POOL,
    MAX_LONGS,
    MAX_SHORTS,
    STOP_LOSS_PER_LEG,
)
from ..exp04_small_account.run import PRESETS
from .regimes import compute_regime_labels


def _specs_from_weights(
    weights: Dict[str, float], factor_specs: Dict[str, Dict]
) -> List[FactorSpec]:
    out = []
    for name, w in weights.items():
        spec = factor_specs.get(name)
        if spec is None:
            continue
        out.append(
            FactorSpec(
                name=name,
                kind=spec["kind"],
                lookback=int(spec["lookback"]),
                skip=int(spec.get("skip", 0)),
                weight=float(w),
            )
        )
    return out


def run_static_backtest(
    returns: pd.DataFrame,
    funding: pd.DataFrame,
    specs: List[FactorSpec],
    cfg: SmallAccountConfig,
) -> Tuple[pd.DataFrame, Dict]:
    """复刻 exp04 backtester.run_small_account_backtest 但 inline 以便 regime_switch 使用同一范式。"""
    sectors = get_sectors(list(returns.columns))
    score = build_composite_score(
        returns,
        funding,
        specs,
        sectors,
        sector_neutral=cfg.sector_neutral,
        winsorize_pct=cfg.winsorize_pct,
    )
    return _execute(returns, score, cfg, label="static"), None


def _execute(
    returns: pd.DataFrame, score: pd.DataFrame, cfg: SmallAccountConfig, label: str
) -> pd.DataFrame:
    reb_idx = returns.index[cfg.lookback_max :: cfg.hold_bars]
    port_ret_gross = pd.Series(0.0, index=returns.index)
    port_ret_net = pd.Series(0.0, index=returns.index)
    prev_w = pd.Series(0.0, index=returns.columns)

    for i in range(len(reb_idx) - 1):
        t0 = reb_idx[i]
        t1 = reb_idx[i + 1]
        s_row = score.loc[t0].dropna()
        if len(s_row) < cfg.max_longs + cfg.max_shorts:
            continue
        ranked = s_row.sort_values(ascending=False)
        longs = ranked.head(cfg.max_longs).index.tolist()
        shorts = ranked.tail(cfg.max_shorts).index.tolist()
        w = pd.Series(0.0, index=returns.columns)
        w[longs] = 0.5 / max(len(longs), 1)
        w[shorts] = -0.5 / max(len(shorts), 1)

        seg = returns.loc[(returns.index > t0) & (returns.index <= t1)]
        if len(seg) == 0:
            continue
        position_w = w.copy()
        cum_pnl_frac = pd.Series(0.0, index=w.index)
        seg_port_ret = pd.Series(0.0, index=seg.index)

        for ts, row in seg.iterrows():
            leg_ret = row * np.sign(position_w)
            active = position_w != 0
            cum_pnl_frac[active] = cum_pnl_frac[active] + leg_ret[active]
            seg_port_ret.loc[ts] = (row * position_w).sum()
            triggered = cum_pnl_frac[
                active & (cum_pnl_frac < -cfg.stop_loss_per_leg)
            ].index.tolist()
            for sym in triggered:
                fee = abs(position_w[sym]) * cfg.fee_bps_per_side / 1e4
                seg_port_ret.loc[ts] -= fee
                position_w[sym] = 0.0

        port_ret_gross.loc[seg.index] = seg_port_ret
        port_ret_net.loc[seg.index] = seg_port_ret
        turnover = float((w - prev_w).abs().sum())
        cost = turnover * cfg.fee_bps_per_side / 1e4
        port_ret_net.loc[seg.index[0]] -= cost
        prev_w = position_w

    return pd.DataFrame(
        {
            "port_ret_gross": port_ret_gross,
            "port_ret_net": port_ret_net,
            "equity_gross": (1 + port_ret_gross).cumprod(),
            "equity_net": (1 + port_ret_net).cumprod(),
        }
    )


def run_regime_switch_backtest(
    returns: pd.DataFrame,
    funding: pd.DataFrame,
    regimes: pd.DataFrame,
    weights_yaml: Dict,
    cfg: SmallAccountConfig,
) -> pd.DataFrame:
    """每次 rebalance 时：
    1. 读 regime 标签
    2. 查 regime_weights.yaml 得到 factor weights
    3. build_composite_score 只用这些 factors
    """
    factor_specs = weights_yaml["factor_specs"]
    reg_weights = weights_yaml["regime_weights"]
    default_spec_weights = reg_weights.get("ALL", {}).get("factors", {})
    sectors = get_sectors(list(returns.columns))

    # 缓存：为每个 unique 权重签名计算一次 score
    score_cache: Dict[Tuple, pd.DataFrame] = {}

    def get_score_df(weight_key: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
        if weight_key in score_cache:
            return score_cache[weight_key]
        specs = _specs_from_weights(dict(weight_key), factor_specs)
        if not specs:
            specs = _specs_from_weights(default_spec_weights, factor_specs)
        s = build_composite_score(
            returns,
            funding,
            specs,
            sectors,
            sector_neutral=cfg.sector_neutral,
            winsorize_pct=cfg.winsorize_pct,
        )
        score_cache[weight_key] = s
        return s

    reb_idx = returns.index[cfg.lookback_max :: cfg.hold_bars]
    port_ret_gross = pd.Series(0.0, index=returns.index)
    port_ret_net = pd.Series(0.0, index=returns.index)
    prev_w = pd.Series(0.0, index=returns.columns)
    regime_trace = []

    for i in range(len(reb_idx) - 1):
        t0 = reb_idx[i]
        t1 = reb_idx[i + 1]
        # 找到当前 regime（用 t0 前一 bar 已知信息避免用未来）
        reg = (
            regimes["collapsed"].loc[:t0].iloc[-1]
            if t0 in regimes.index or len(regimes.loc[:t0])
            else "range_normal"
        )
        reg_conf = reg_weights.get(reg, reg_weights.get("ALL", {}))
        fweights = reg_conf.get("factors", default_spec_weights)
        weight_key = tuple(sorted(fweights.items()))
        score = get_score_df(weight_key)
        regime_trace.append({"time": t0, "regime": reg, "n_factors": len(fweights)})

        s_row = score.loc[t0].dropna() if t0 in score.index else pd.Series(dtype=float)
        if len(s_row) < cfg.max_longs + cfg.max_shorts:
            continue
        ranked = s_row.sort_values(ascending=False)
        longs = ranked.head(cfg.max_longs).index.tolist()
        shorts = ranked.tail(cfg.max_shorts).index.tolist()
        w = pd.Series(0.0, index=returns.columns)
        w[longs] = 0.5 / max(len(longs), 1)
        w[shorts] = -0.5 / max(len(shorts), 1)

        seg = returns.loc[(returns.index > t0) & (returns.index <= t1)]
        if len(seg) == 0:
            continue
        position_w = w.copy()
        cum_pnl_frac = pd.Series(0.0, index=w.index)
        seg_port_ret = pd.Series(0.0, index=seg.index)
        for ts, row in seg.iterrows():
            leg_ret = row * np.sign(position_w)
            active = position_w != 0
            cum_pnl_frac[active] = cum_pnl_frac[active] + leg_ret[active]
            seg_port_ret.loc[ts] = (row * position_w).sum()
            triggered = cum_pnl_frac[
                active & (cum_pnl_frac < -cfg.stop_loss_per_leg)
            ].index.tolist()
            for sym in triggered:
                fee = abs(position_w[sym]) * cfg.fee_bps_per_side / 1e4
                seg_port_ret.loc[ts] -= fee
                position_w[sym] = 0.0
        port_ret_gross.loc[seg.index] = seg_port_ret
        port_ret_net.loc[seg.index] = seg_port_ret
        turnover = float((w - prev_w).abs().sum())
        cost = turnover * cfg.fee_bps_per_side / 1e4
        port_ret_net.loc[seg.index[0]] -= cost
        prev_w = position_w

    eq = pd.DataFrame(
        {
            "port_ret_gross": port_ret_gross,
            "port_ret_net": port_ret_net,
            "equity_gross": (1 + port_ret_gross).cumprod(),
            "equity_net": (1 + port_ret_net).cumprod(),
        }
    )
    eq.attrs["regime_trace"] = pd.DataFrame(regime_trace)
    return eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2026-03")
    ap.add_argument("--price-dir", default="data/parquet_data")
    ap.add_argument("--funding-dir", default="data/funding_rate/parquet")
    ap.add_argument(
        "--weights-yaml",
        default="reports/cross_section/exp05_regime_ic/regime_ic/regime_weights.yaml",
    )
    ap.add_argument(
        "--outdir", default="reports/cross_section/exp05_regime_ic/switch_backtest"
    )
    ap.add_argument("--hold-bars", type=int, default=HOLD_BARS_DEFAULT)
    ap.add_argument("--timeframe", default="1h")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    weights_yaml = yaml.safe_load(Path(args.weights_yaml).read_text())
    print(f"[1/4] 加载面板")
    panels = build_panels(
        LIQUID_POOL,
        args.start,
        args.end,
        Path(args.price_dir),
        Path(args.funding_dir),
        args.timeframe,
        0.5,
        verbose=False,
    )
    prices = panels["prices"]
    returns = panels["returns"].fillna(0.0)
    funding = panels["funding"].fillna(0.0)
    print(f"      aligned: {returns.shape[0]} bars, {returns.shape[1]} symbols")

    regimes = compute_regime_labels(prices, funding)

    # 统一 config
    def mk_cfg(lookback_max: int):
        return SmallAccountConfig(
            account_size_usd=ACCOUNT_SIZE_USD,
            max_longs=MAX_LONGS,
            max_shorts=MAX_SHORTS,
            hold_bars=args.hold_bars,
            fee_bps_per_side=FEE_BPS_PER_SIDE,
            stop_loss_per_leg=STOP_LOSS_PER_LEG,
            sector_neutral=True,
            lookback_max=lookback_max,
        )

    factor_specs = weights_yaml["factor_specs"]
    max_lb = max(
        int(v["lookback"]) + int(v.get("skip", 0)) for v in factor_specs.values()
    )
    cfg = mk_cfg(max_lb)

    results: Dict[str, pd.DataFrame] = {}

    # Baseline 1: static mom_only
    print("[2/4] Static mom_only baseline")
    specs_mom = PRESETS["mom_only"]
    eq1 = _execute(
        returns,
        build_composite_score(
            returns,
            funding,
            specs_mom,
            get_sectors(list(returns.columns)),
            sector_neutral=True,
            winsorize_pct=cfg.winsorize_pct,
        ),
        cfg,
        "mom_only",
    )
    results["static_mom_only"] = eq1

    # Baseline 2: static using ALL weights
    print("[2/4] Static ALL-regime weights baseline")
    all_w = weights_yaml["regime_weights"]["ALL"]["factors"]
    specs_all = _specs_from_weights(all_w, factor_specs)
    eq2 = _execute(
        returns,
        build_composite_score(
            returns,
            funding,
            specs_all,
            get_sectors(list(returns.columns)),
            sector_neutral=True,
            winsorize_pct=cfg.winsorize_pct,
        ),
        cfg,
        "all_weights",
    )
    results["static_all_weights"] = eq2

    # Regime switch
    print("[3/4] Regime-switch")
    eq3 = run_regime_switch_backtest(returns, funding, regimes, weights_yaml, cfg)
    results["regime_switch"] = eq3

    # 指标 + 保存
    bars_per_year = 24 * 365
    rows = []
    for name, eq in results.items():
        m = _metrics(eq, bars_per_year)
        m["strategy"] = name
        rows.append(m)
        eq_save = eq.copy()
        eq_save.attrs.clear()
        eq_save.to_parquet(outdir / f"equity_{name}.parquet")
        print(
            f"  {name:<25s} | Gross SR={m['gross_sharpe']:+.2f}  "
            f"Net SR={m['net_sharpe']:+.2f}  "
            f"AnnRet={m['net_ann_return']*100:+.1f}%  "
            f"MaxDD={m['net_max_dd']*100:.1f}%"
        )
    pd.DataFrame(rows).to_csv(outdir / "metrics.csv", index=False)

    # Regime trace
    if "regime_trace" in results["regime_switch"].attrs:
        rt = results["regime_switch"].attrs["regime_trace"]
        rt.to_csv(outdir / "regime_trace.csv", index=False)

    # 画图
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        for name, eq in results.items():
            eq["equity_net"].plot(ax=ax, label=name)
        if "BTCUSDT" in prices.columns:
            btc = prices["BTCUSDT"].dropna()
            (btc / btc.iloc[0]).plot(ax=ax, label="BTC_BH", alpha=0.4, color="orange")
        ax.axhline(1.0, color="k", lw=0.5)
        ax.set_title("Regime-switch vs static baselines (Net equity)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "equity_compare.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    print("[4/4] 写 summary")
    lines = [
        "# exp05 Part 3 - Regime-Switching Backtest\n",
        f"- Period: {args.start} -> {args.end}",
        f"- Hold bars: {args.hold_bars}",
        f"- WARNING: regime_weights.yaml 是用 full sample 学的，存在 look-ahead；此为 POC\n",
        "## 三种策略对比\n",
        "| strategy | gross SR | net SR | ann return | max DD |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['strategy']} | {r['gross_sharpe']:+.2f} | **{r['net_sharpe']:+.2f}** "
            f"| {r['net_ann_return']*100:+.1f}% | {r['net_max_dd']*100:.1f}% |"
        )
    lines.append("\n## 判读\n")
    lines.append(
        "- 若 static_all >> static_mom_only → IC-weighted combo 相比单因子显著更稳"
    )
    lines.append(
        "- 若 regime_switch ~ static_all → regime 切换换来的定向 alpha 被换因子噪声抵消"
    )
    lines.append("- 若 regime_switch > static_all → regime 信号真正有效")
    lines.append("\n**下一步**:")
    lines.append("1. walk-forward OOS 权重训练（用过去 N 天学到的权重只用于未来 M 天）")
    lines.append("2. 增加 stop_loss 放宽到 0.20 的 ablation")
    lines.append("3. 把 static_all_weights 作为 exp07 paper 的默认 preset 候选")
    (outdir / "summary.md").write_text("\n".join(lines))
    print(f"完成 -> {outdir.resolve()}")


if __name__ == "__main__":
    main()
