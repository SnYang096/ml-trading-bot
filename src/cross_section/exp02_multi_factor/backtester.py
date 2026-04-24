"""多因子横截面 L/S 回测器。

流程（每个 rebalance 时点 t）：
    1. 取每个因子的当前截面值
    2. 板块内 z-score 中性化（可选），然后跨截面 z-score
    3. 按权重线性合成得到 composite score
    4. Winsorize + 再次 z-score
    5. 映射为目标权重：
        - rank-based: top-K long / bottom-K short 等权
        - score-weighted: weight_i ∝ score_i / sum(|score_j|)，仅保留 score > threshold 的做多、< -threshold 的做空
    6. 等金额美元中性 (sum(w) = 0, sum(|w|) = 1)
    7. 持仓至下个 rebalance，应用交易成本（按 |Δw| 计）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import factors as F
from .sectors import (
    cross_sectional_zscore,
    get_sectors,
    sector_neutralize,
)


@dataclass
class FactorSpec:
    name: str
    weight: float
    kind: str  # 'mom' | 'reversal' | 'funding' | 'low_vol'
    lookback: int = 24 * 7
    skip: int = 0
    vol_normalize: bool = False


@dataclass
class BacktestConfig:
    lookback_max: int = 24 * 14
    hold_bars: int = 24
    top_k: Optional[int] = None  # 若为 None，用 score_weighted
    bottom_k: Optional[int] = None
    score_threshold: float = 0.5  # score-weighted 模式下的 |z| 阈值
    fee_bps_per_side: float = 5.0
    sector_neutral: bool = True
    winsorize_pct: float = 0.02
    min_symbols: int = 8  # 截面最少可用币种


def _compute_single_factor(
    spec: FactorSpec, returns: pd.DataFrame, funding: pd.DataFrame
) -> pd.DataFrame:
    if spec.kind == "mom":
        raw = F.momentum(returns, spec.lookback, spec.skip)
    elif spec.kind == "reversal":
        raw = F.short_term_reversal(returns, spec.lookback)
    elif spec.kind == "funding":
        raw = F.funding_factor(funding, spec.lookback)
    elif spec.kind == "low_vol":
        raw = F.low_vol_factor(returns, spec.lookback)
    else:
        raise ValueError(f"未知因子类型: {spec.kind}")
    if spec.vol_normalize:
        raw = F.vol_normalize(raw, returns)
    return raw


def build_composite_score(
    returns: pd.DataFrame,
    funding: pd.DataFrame,
    specs: List[FactorSpec],
    sectors: pd.Series,
    sector_neutral: bool = True,
    winsorize_pct: float = 0.02,
) -> pd.DataFrame:
    """合成复合分数：每个因子 -> (板块中性) z-score -> 加权求和 -> 再 z-score。"""
    composite = None
    total_w = 0.0
    for spec in specs:
        raw = _compute_single_factor(spec, returns, funding)
        raw = F.winsorize(raw, winsorize_pct, 1 - winsorize_pct)
        if sector_neutral:
            z = sector_neutralize(raw, sectors)
        else:
            z = cross_sectional_zscore(raw)
        composite = (
            z * spec.weight if composite is None else composite + z * spec.weight
        )
        total_w += abs(spec.weight)
    if total_w > 0:
        composite = composite / total_w
    return cross_sectional_zscore(composite)


def _score_to_weights(
    score_row: pd.Series,
    cfg: BacktestConfig,
) -> pd.Series:
    """把一行 score 转成目标权重（美元中性，sum(|w|) = 1）。"""
    w = pd.Series(0.0, index=score_row.index)
    s = score_row.dropna()
    if len(s) < cfg.min_symbols:
        return w
    if cfg.top_k is not None and cfg.bottom_k is not None:
        ranked = s.sort_values(ascending=False)
        longs = ranked.head(cfg.top_k).index.tolist()
        shorts = ranked.tail(cfg.bottom_k).index.tolist()
        w[longs] = 0.5 / max(len(longs), 1)
        w[shorts] = -0.5 / max(len(shorts), 1)
    else:
        longs_mask = s > cfg.score_threshold
        shorts_mask = s < -cfg.score_threshold
        if not longs_mask.any() or not shorts_mask.any():
            return w
        s_long = s[longs_mask]
        s_short = s[shorts_mask]
        if s_long.sum() > 0:
            w[s_long.index] = 0.5 * s_long / s_long.sum()
        if s_short.abs().sum() > 0:
            w[s_short.index] = -0.5 * s_short.abs() / s_short.abs().sum()
    # 规范化到 sum(|w|) = 1
    tot = w.abs().sum()
    if tot > 0:
        w = w / tot
    return w


def run_backtest(
    returns: pd.DataFrame,
    funding: pd.DataFrame,
    specs: List[FactorSpec],
    cfg: BacktestConfig,
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """执行回测。

    返回：
        equity_df (port_ret_gross, port_ret_net, equity_gross, equity_net)
        metrics dict
        weights_df (shape: [n_rebalances, n_symbols])
    """
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
    weight_records: List[Dict] = []

    for i in range(len(reb_idx) - 1):
        t0 = reb_idx[i]
        t1 = reb_idx[i + 1]
        score_row = score.loc[t0]
        w = _score_to_weights(score_row, cfg)
        weight_records.append({"time": t0, **w.to_dict()})

        seg = returns.loc[(returns.index > t0) & (returns.index <= t1)]
        if len(seg) == 0:
            continue
        # 线性收益，日内按持仓不变假设
        seg_ret = seg.mul(w, axis=1).sum(axis=1)
        port_ret_gross.loc[seg.index] = seg_ret
        port_ret_net.loc[seg.index] = seg_ret

        turnover = float((w - prev_w).abs().sum())
        cost = turnover * cfg.fee_bps_per_side / 1e4
        port_ret_net.loc[seg.index[0]] -= cost
        prev_w = w

    eq_df = pd.DataFrame(
        {
            "port_ret_gross": port_ret_gross,
            "port_ret_net": port_ret_net,
            "equity_gross": (1 + port_ret_gross).cumprod(),
            "equity_net": (1 + port_ret_net).cumprod(),
        }
    )

    bars_per_year = 24 * 365
    metrics = _compute_metrics(eq_df, bars_per_year)
    metrics.update(
        {
            "n_rebalances": len(weight_records),
            "hold_bars": cfg.hold_bars,
            "sector_neutral": cfg.sector_neutral,
            "fee_bps_per_side": cfg.fee_bps_per_side,
            "top_k": cfg.top_k,
            "bottom_k": cfg.bottom_k,
            "score_threshold": cfg.score_threshold,
            "factors": [
                f"{s.name}:{s.kind}(lb={s.lookback},w={s.weight})" for s in specs
            ],
        }
    )
    weights_df = (
        pd.DataFrame(weight_records).set_index("time")
        if weight_records
        else pd.DataFrame()
    )
    return eq_df, metrics, weights_df


def _compute_metrics(eq: pd.DataFrame, bars_per_year: int) -> Dict:
    def _stats(r: pd.Series, prefix: str) -> Dict:
        ann_r = r.mean() * bars_per_year
        ann_v = r.std() * np.sqrt(bars_per_year)
        sr = ann_r / ann_v if ann_v > 0 else np.nan
        equity = (1 + r).cumprod()
        peak = equity.cummax()
        dd = (equity / peak - 1).min()
        return {
            f"{prefix}_ann_return": float(ann_r),
            f"{prefix}_ann_vol": float(ann_v),
            f"{prefix}_sharpe": float(sr),
            f"{prefix}_max_dd": float(dd),
            f"{prefix}_final_equity": float(equity.iloc[-1]),
        }

    out = {}
    out.update(_stats(eq["port_ret_gross"], "gross"))
    out.update(_stats(eq["port_ret_net"], "net"))
    return out
