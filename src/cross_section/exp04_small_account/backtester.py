"""exp04 小资金回测器。

区别于 exp02：
    1. 强制 top_k / bottom_k 很小（默认 2+2）
    2. 每腿独立跟踪，单腿累计亏损 > stop_loss_per_leg 时提前平该腿
    3. 报告实际换手次数、单次预计 notional 对 MIN_NOTIONAL 的可行性
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..exp02_multi_factor.backtester import FactorSpec, build_composite_score
from ..exp02_multi_factor.sectors import get_sectors


@dataclass
class SmallAccountConfig:
    account_size_usd: float = 10_000.0
    max_longs: int = 2
    max_shorts: int = 2
    hold_bars: int = 24 * 14
    fee_bps_per_side: float = 8.0
    stop_loss_per_leg: float = 0.15
    min_notional_usd: float = 200.0
    sector_neutral: bool = True
    winsorize_pct: float = 0.02
    lookback_max: int = 24 * 30


def run_small_account_backtest(
    returns: pd.DataFrame,
    funding: pd.DataFrame,
    specs: List[FactorSpec],
    cfg: SmallAccountConfig,
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """返回: equity_df, metrics, trades_df"""
    sectors = get_sectors(list(returns.columns))
    score = build_composite_score(
        returns,
        funding,
        specs,
        sectors,
        sector_neutral=cfg.sector_neutral,
        winsorize_pct=cfg.winsorize_pct,
    )

    reb_idx = returns.index[cfg.lookback_max :: cfg.hold_bars]
    port_ret_gross = pd.Series(0.0, index=returns.index)
    port_ret_net = pd.Series(0.0, index=returns.index)
    prev_w = pd.Series(0.0, index=returns.columns)
    trades: List[Dict] = []

    for i in range(len(reb_idx) - 1):
        t0 = reb_idx[i]
        t1 = reb_idx[i + 1]
        s_row = score.loc[t0].dropna()
        if len(s_row) < cfg.max_longs + cfg.max_shorts:
            continue
        ranked = s_row.sort_values(ascending=False)
        longs = ranked.head(cfg.max_longs).index.tolist()
        shorts = ranked.tail(cfg.max_shorts).index.tolist()

        # 等权、美元中性、sum(|w|)=1
        w = pd.Series(0.0, index=returns.columns)
        w[longs] = 0.5 / max(len(longs), 1)
        w[shorts] = -0.5 / max(len(shorts), 1)

        seg = returns.loc[(returns.index > t0) & (returns.index <= t1)]
        if len(seg) == 0:
            continue

        # 逐币累计收益，检查 stop loss
        # 思路：每个有仓位的币种累计对数收益，超过阈值就将其权重从当根后归零
        position_w = w.copy()
        cum_pnl_frac = pd.Series(
            0.0, index=w.index
        )  # 每腿累计 pnl 占该腿 notional 的比例
        seg_port_ret = pd.Series(0.0, index=seg.index)
        stop_hit = []

        for ts, row in seg.iterrows():
            leg_ret = row * np.sign(position_w)  # 多腿=收益, 空腿=-收益
            # 累计（对数近似为简单和）
            active_mask = position_w != 0
            cum_pnl_frac[active_mask] = cum_pnl_frac[active_mask] + leg_ret[active_mask]
            # 当前根的组合收益
            seg_port_ret.loc[ts] = (row * position_w).sum()
            # 触发止损的腿
            triggered = cum_pnl_frac[
                active_mask & (cum_pnl_frac < -cfg.stop_loss_per_leg)
            ].index.tolist()
            if triggered:
                for sym in triggered:
                    stop_hit.append(
                        {
                            "time": ts,
                            "symbol": sym,
                            "cum_pnl_frac": float(cum_pnl_frac[sym]),
                            "side": "long" if position_w[sym] > 0 else "short",
                        }
                    )
                    # 关闭该腿：立即计算该腿平仓成本（fee_bps * |w|），从下一根开始 w=0
                    fee = abs(position_w[sym]) * cfg.fee_bps_per_side / 1e4
                    seg_port_ret.loc[ts] -= fee
                    position_w[sym] = 0.0

        port_ret_gross.loc[seg.index] = seg_port_ret  # gross 也包含 stop 成本（保守）
        port_ret_net.loc[seg.index] = seg_port_ret

        # rebalance 换手成本（对比上次 weight）
        turnover = float((w - prev_w).abs().sum())
        cost = turnover * cfg.fee_bps_per_side / 1e4
        port_ret_net.loc[seg.index[0]] -= cost
        prev_w = position_w  # 下次 rebalance 时，止损后的仓位才是起点

        # 记录本次 rebalance 的持仓
        for sym in longs:
            trades.append(
                {
                    "time": t0,
                    "symbol": sym,
                    "side": "long",
                    "weight": float(w[sym]),
                    "notional_usd": float(abs(w[sym]) * cfg.account_size_usd),
                    "feasible": bool(
                        abs(w[sym]) * cfg.account_size_usd >= cfg.min_notional_usd
                    ),
                    "stopped_out": sym in [s["symbol"] for s in stop_hit],
                }
            )
        for sym in shorts:
            trades.append(
                {
                    "time": t0,
                    "symbol": sym,
                    "side": "short",
                    "weight": float(w[sym]),
                    "notional_usd": float(abs(w[sym]) * cfg.account_size_usd),
                    "feasible": bool(
                        abs(w[sym]) * cfg.account_size_usd >= cfg.min_notional_usd
                    ),
                    "stopped_out": sym in [s["symbol"] for s in stop_hit],
                }
            )

    eq_df = pd.DataFrame(
        {
            "port_ret_gross": port_ret_gross,
            "port_ret_net": port_ret_net,
            "equity_gross": (1 + port_ret_gross).cumprod(),
            "equity_net": (1 + port_ret_net).cumprod(),
        }
    )

    bars_per_year = 24 * 365
    metrics = _metrics(eq_df, bars_per_year)
    metrics.update(
        {
            "n_rebalances": len(reb_idx) - 1,
            "hold_bars": cfg.hold_bars,
            "max_longs": cfg.max_longs,
            "max_shorts": cfg.max_shorts,
            "fee_bps_per_side": cfg.fee_bps_per_side,
            "stop_loss_per_leg": cfg.stop_loss_per_leg,
            "sector_neutral": cfg.sector_neutral,
            "account_size_usd": cfg.account_size_usd,
            "factors": [
                f"{s.name}:{s.kind}(lb={s.lookback},w={s.weight})" for s in specs
            ],
        }
    )
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        metrics["n_trades_total"] = int(len(trades_df))
        metrics["n_trades_stopped"] = int(trades_df["stopped_out"].sum())
        metrics["avg_notional_usd"] = float(trades_df["notional_usd"].mean())
        metrics["pct_feasible"] = float(trades_df["feasible"].mean())
    return eq_df, metrics, trades_df


def _metrics(eq: pd.DataFrame, bars_per_year: int) -> Dict:
    def _s(r: pd.Series, prefix: str) -> Dict:
        ann_r = r.mean() * bars_per_year
        ann_v = r.std() * np.sqrt(bars_per_year)
        sr = ann_r / ann_v if ann_v > 0 else np.nan
        equity = (1 + r).cumprod()
        dd = (equity / equity.cummax() - 1).min()
        return {
            f"{prefix}_ann_return": float(ann_r),
            f"{prefix}_ann_vol": float(ann_v),
            f"{prefix}_sharpe": float(sr) if sr == sr else np.nan,
            f"{prefix}_max_dd": float(dd),
            f"{prefix}_final_equity": float(equity.iloc[-1]),
        }

    out = {}
    out.update(_s(eq["port_ret_gross"], "gross"))
    out.update(_s(eq["port_ret_net"], "net"))
    return out
