"""因子 IC (Information Coefficient) 分析。

IC 定义：每个时点 t，因子值截面与未来 H 根 K 线收益截面的 Spearman 相关系数。
- mean(IC) > 0.03, IR = mean/std > 0.5  通常视为可用
- 若某因子 IC 长期为负 -> 方向搞反了，需要取负号
"""

from __future__ import annotations

from typing import Dict, List

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr

warnings.filterwarnings("ignore", category=ConstantInputWarning)


def forward_returns(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """未来 horizon 根 K 线累计对数收益（不含当前行）。"""
    return returns.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))


def factor_ic_series(
    factor: pd.DataFrame, fwd_ret: pd.DataFrame, sample_every: int = 1
) -> pd.Series:
    """每行计算 factor vs fwd_ret 的 Spearman IC，返回 IC 时间序列。

    sample_every: 为性能每 N 行采样一次（1h bar 下设 24 表示每天）。
    """
    idx_all = factor.index.intersection(fwd_ret.index)
    if sample_every > 1:
        idx_all = idx_all[::sample_every]
    out = []
    f = factor.loc[idx_all]
    r = fwd_ret.loc[idx_all]
    for t in idx_all:
        fv = f.loc[t].dropna()
        rv = r.loc[t].dropna()
        common = fv.index.intersection(rv.index)
        if len(common) < 5:
            out.append(np.nan)
            continue
        rho, _ = spearmanr(fv.loc[common].values, rv.loc[common].values)
        out.append(rho)
    return pd.Series(out, index=idx_all, name="ic")


def ic_stats(ic: pd.Series) -> Dict[str, float]:
    ic = ic.dropna()
    if len(ic) == 0:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "ic_hit_rate": np.nan,
            "n_samples": 0,
        }
    mu = float(ic.mean())
    sd = float(ic.std())
    ir = mu / sd if sd > 0 else np.nan
    hit = float((ic > 0).mean())
    return {
        "ic_mean": mu,
        "ic_std": sd,
        "ic_ir": float(ir) if ir == ir else np.nan,
        "ic_hit_rate": hit,
        "n_samples": int(len(ic)),
    }


def factor_quantile_returns(
    factor: pd.DataFrame, fwd_ret: pd.DataFrame, n_q: int = 5, sample_every: int = 24
) -> pd.DataFrame:
    """按因子值分 n_q 分位，计算每个分位的平均 forward return。
    输出：index=quantile (1..n_q)，列=mean_fwd_return/hit/count。
    """
    idx_all = factor.index.intersection(fwd_ret.index)[::sample_every]
    rows = []
    for t in idx_all:
        fv = factor.loc[t].dropna()
        rv = fwd_ret.loc[t].dropna()
        common = fv.index.intersection(rv.index)
        if len(common) < n_q * 2:
            continue
        fv = fv.loc[common]
        rv = rv.loc[common]
        try:
            q = pd.qcut(fv.rank(method="first"), n_q, labels=False) + 1
        except Exception:
            continue
        for qi in range(1, n_q + 1):
            mask = q == qi
            if mask.any():
                rows.append({"quantile": qi, "fwd_ret": rv[mask].mean()})
    if not rows:
        return pd.DataFrame()
    df = (
        pd.DataFrame(rows)
        .groupby("quantile")
        .agg(
            mean_fwd_return=("fwd_ret", "mean"),
            std_fwd_return=("fwd_ret", "std"),
            count=("fwd_ret", "size"),
        )
    )
    return df
