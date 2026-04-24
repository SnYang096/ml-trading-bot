"""exp05 v2 — Walk-forward OOS regime IC 权重（消除 look-ahead）。

对每个 rebalance 时点 t0：
    - 训练窗 [t0 - train_window_bars, t0] 内，IC 样本只取到 index <= t0 - horizon_bars，
      保证 14d forward return 在 t0 前已完全实现。
    - 在该窗内拟合 regime_weights（与 run_regime_ic 相同规则）。
    - OOS：用该权重在 (t0, t1] 持仓段交易（与 regime_switch 一致）。

产出:
    reports/cross_section/exp05_regime_ic/walk_forward/wf_weights.jsonl
    reports/cross_section/exp05_regime_ic/walk_forward/equity_wf_oos.parquet
    reports/cross_section/exp05_regime_ic/walk_forward/summary.md
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

from ..exp02_multi_factor.backtester import FactorSpec, build_composite_score
from ..exp02_multi_factor.data_loader import build_panels
from ..exp02_multi_factor.sectors import get_sectors
from ..exp04_small_account.backtester import SmallAccountConfig, _metrics
from ..exp04_small_account.config import (
    FEE_BPS_PER_SIDE,
    HOLD_BARS_DEFAULT,
    LIQUID_POOL,
    MAX_LONGS,
    MAX_SHORTS,
    STOP_LOSS_PER_LEG,
)
from ..exp04_small_account.run import PRESETS
from .regimes import compute_regime_labels
from .wf_ic_utils import (
    TARGET_HORIZON_BARS,
    factor_specs_dict,
    fit_weights_at_rebalance,
    max_lookback_bars,
    precompute_factors_fwd_regimes,
)


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


def _mom_only_fallback_weights() -> Dict[str, float]:
    return {s.name: s.weight for s in PRESETS["mom_only"]}


def run_wf_oos_backtest(
    returns: pd.DataFrame,
    funding: pd.DataFrame,
    regimes: pd.DataFrame,
    cfg: SmallAccountConfig,
    weights_by_t0: Dict[pd.Timestamp, Dict[str, Dict]],
    factor_specs: Dict[str, Dict],
) -> pd.DataFrame:
    """weights_by_t0[t0] = regime_weights 子 dict（含 ALL, bull_normal, ...）。"""
    sectors = get_sectors(list(returns.columns))
    default_w = _mom_only_fallback_weights()
    score_cache: Dict[Tuple, pd.DataFrame] = {}

    def get_score(weight_key: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
        if weight_key in score_cache:
            return score_cache[weight_key]
        specs = _specs_from_weights(dict(weight_key), factor_specs)
        if not specs:
            specs = _specs_from_weights(default_w, factor_specs)
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

    last_valid_rw: Dict[str, Dict] = {}

    for i in range(len(reb_idx) - 1):
        t0 = reb_idx[i]
        t1 = reb_idx[i + 1]
        rw = weights_by_t0.get(t0)
        if rw is None or not rw:
            rw = last_valid_rw
        else:
            last_valid_rw = rw

        reg = (
            regimes["collapsed"].loc[:t0].iloc[-1]
            if len(regimes.loc[:t0])
            else "range_normal"
        )
        reg_conf = rw.get(reg, rw.get("ALL", {})) if rw else {}
        fweights = reg_conf.get("factors", {}) if reg_conf else {}
        if not fweights:
            fweights = default_w
        weight_key = tuple(sorted(fweights.items()))
        score = get_score(weight_key)

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

    return pd.DataFrame(
        {
            "port_ret_gross": port_ret_gross,
            "port_ret_net": port_ret_net,
            "equity_gross": (1 + port_ret_gross).cumprod(),
            "equity_net": (1 + port_ret_net).cumprod(),
        }
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2026-03")
    ap.add_argument("--price-dir", default="data/parquet_data")
    ap.add_argument("--funding-dir", default="data/funding_rate/parquet")
    ap.add_argument(
        "--outdir", default="reports/cross_section/exp05_regime_ic/walk_forward"
    )
    ap.add_argument(
        "--train-window-bars",
        type=int,
        default=24 * 180,
        help="训练窗长度（bars），默认 180 天（减轻 IC 噪声）",
    )
    ap.add_argument(
        "--hold-bars",
        type=int,
        default=HOLD_BARS_DEFAULT,
        help="持仓周期（bars）；增大则换仓/重算决策点更稀",
    )
    ap.add_argument(
        "--refit-every-n",
        type=int,
        default=1,
        help="每 N 个 rebalance 点才重新拟合 IC 权重，中间 bar 沿用上一份权重（降换手/过拟合）",
    )
    ap.add_argument(
        "--all-weights-only",
        action="store_true",
        help="只做全截面 ALL 的 walk-forward 权重，不按 regime 切分（弱化 regime×因子维度过拟合）",
    )
    ap.add_argument("--min-regime-samples", type=int, default=50)
    ap.add_argument("--sample-every", type=int, default=24)
    ap.add_argument("--ic-threshold", type=float, default=0.02)
    ap.add_argument(
        "--skip-fit",
        action="store_true",
        help="跳过拟合，只读已有 wf_weights.jsonl 回测",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / "wf_weights.jsonl"

    print("[1/4] 加载面板")
    panels = build_panels(
        LIQUID_POOL,
        args.start,
        args.end,
        Path(args.price_dir),
        Path(args.funding_dir),
        "1h",
        0.5,
        verbose=False,
    )
    returns = panels["returns"].fillna(0.0)
    funding = panels["funding"].fillna(0.0)
    prices = panels["prices"]
    print(f"      aligned: {returns.shape[0]} bars")

    cfg = SmallAccountConfig(
        account_size_usd=10_000,
        max_longs=MAX_LONGS,
        max_shorts=MAX_SHORTS,
        hold_bars=args.hold_bars,
        fee_bps_per_side=FEE_BPS_PER_SIDE,
        stop_loss_per_leg=STOP_LOSS_PER_LEG,
        sector_neutral=True,
        lookback_max=max_lookback_bars(),
    )
    reb_idx = returns.index[cfg.lookback_max :: cfg.hold_bars]
    print(f"      rebalance points: {len(reb_idx)}")

    factor_specs = factor_specs_dict()
    regimes = compute_regime_labels(prices, funding)

    weights_by_t0: Dict[pd.Timestamp, Dict[str, Dict]] = {}

    if not args.skip_fit:
        print("[2/4] 预计算因子 + walk-forward 拟合权重（较慢）")
        pre = precompute_factors_fwd_regimes(
            returns, funding, prices, TARGET_HORIZON_BARS
        )
        n_ok = 0
        n_carry = 0
        last_rw: Dict[str, Dict] = {}
        with open(jsonl_path, "w") as fj:
            for j, t0 in enumerate(reb_idx):
                should_refit = j % max(1, args.refit_every_n) == 0
                if not should_refit and last_rw:
                    weights_by_t0[t0] = last_rw
                    rec = {
                        "t0": str(t0),
                        "regime_weights": last_rw,
                        "note": "carry_forward",
                        "refit_index": j,
                        "refit_every_n": args.refit_every_n,
                        "all_weights_only": args.all_weights_only,
                    }
                    fj.write(json.dumps(rec, default=str) + "\n")
                    n_carry += 1
                    continue
                try:
                    ic_mat, rw, dbg = fit_weights_at_rebalance(
                        returns,
                        t0,
                        args.train_window_bars,
                        TARGET_HORIZON_BARS,
                        args.min_regime_samples,
                        args.sample_every,
                        args.ic_threshold,
                        pre,
                        all_weights_only=args.all_weights_only,
                    )
                except Exception as e:
                    print(f"  [skip] {t0}  {e}")
                    continue
                if dbg.get("skip") or not rw:
                    rec = {
                        "t0": str(t0),
                        "regime_weights": {},
                        "debug": dbg,
                        "note": "skip_short_train",
                        "all_weights_only": args.all_weights_only,
                    }
                    fj.write(json.dumps(rec, default=str) + "\n")
                    continue
                last_rw = rw
                weights_by_t0[t0] = rw
                rec = {
                    "t0": str(t0),
                    "regime_weights": rw,
                    "debug": {
                        k: int(v) if isinstance(v, (int, np.integer)) else v
                        for k, v in dbg.items()
                    },
                    "note": "fit",
                    "all_weights_only": args.all_weights_only,
                    "refit_every_n": args.refit_every_n,
                }
                fj.write(json.dumps(rec, default=str) + "\n")
                n_ok += 1
                if n_ok % 20 == 0:
                    print(f"  ... fitted {n_ok} / {len(reb_idx)}  (carry {n_carry})")
        print(f"      wrote {jsonl_path}  (fit={n_ok}, carry_forward={n_carry})")
    else:
        print("[2/4] 读取已有 wf_weights.jsonl")
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("regime_weights"):
                weights_by_t0[pd.Timestamp(rec["t0"])] = rec["regime_weights"]

    print("[3/4] OOS 回测")
    eq = run_wf_oos_backtest(
        returns, funding, regimes, cfg, weights_by_t0, factor_specs
    )
    eq_save = eq.copy()
    eq_save.attrs.clear()
    eq_save.to_parquet(outdir / "equity_wf_oos.parquet")
    bars_per_year = 24 * 365
    m = _metrics(eq, bars_per_year)
    (outdir / "metrics_wf_oos.json").write_text(json.dumps(m, indent=2, default=str))

    # 对照：全样本 in-sample regime_switch 需要已有 yaml — 这里只写 wf 指标
    lines = [
        "# exp05 v2 Walk-Forward OOS\n",
        f"- Train window: {args.train_window_bars} bars (~{args.train_window_bars/24:.0f}d)",
        f"- IC horizon: {TARGET_HORIZON_BARS} bars (14d)",
        f"- Hold: {args.hold_bars} bars",
        f"- Refit every N rebalances: {args.refit_every_n}",
        f"- ALL-only (no regime-split in fit): {bool(args.all_weights_only)}",
        f"- IC threshold: {args.ic_threshold}\n",
        "## OOS (fitted weights strictly before each refit t0)\n",
        f"- Net Sharpe: **{m['net_sharpe']:+.3f}**",
        f"- Net ann return: {m['net_ann_return']*100:+.1f}%",
        f"- Max DD: {m['net_max_dd']*100:.1f}%",
        "\nCompare with `switch_backtest/metrics.csv` in-sample regime_switch / static_all.",
        "\n若 OOS SR 显著低于 in-sample static_all → 原结果主要为 look-ahead + 过拟合。",
    ]
    (outdir / "summary.md").write_text("\n".join(lines))
    print(
        f"      Net SR={m['net_sharpe']:+.3f}  AnnRet={m['net_ann_return']*100:+.1f}%"
    )

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 5))
        eq["equity_net"].plot(ax=ax, label="wf_oos_net")
        if "BTCUSDT" in prices.columns:
            btc = prices["BTCUSDT"].dropna()
            (btc / btc.iloc[0]).plot(ax=ax, label="BTC_BH", alpha=0.35)
        ax.axhline(1.0, color="k", lw=0.5)
        ttl = "Walk-forward OOS — net equity"
        if args.all_weights_only:
            ttl += " (ALL-only)"
        if args.refit_every_n > 1:
            ttl += f" (refit every {args.refit_every_n})"
        ax.set_title(ttl)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "equity_wf_oos.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] plot: {e}")

    print(f"[4/4] 完成 -> {outdir.resolve()}")


if __name__ == "__main__":
    main()
