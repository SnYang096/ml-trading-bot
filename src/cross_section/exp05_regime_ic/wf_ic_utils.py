"""Walk-forward IC 权重工具：仅用 train_end 之前的数据拟合 regime_weights（无 look-ahead）。"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..exp02_multi_factor import factors as F
from ..exp02_multi_factor.sectors import cross_sectional_zscore
from ..exp03_ic_and_grid.ic_analysis import factor_ic_series, forward_returns, ic_stats
from .regimes import COLLAPSED_REGIMES, compute_regime_labels
from .run_horizon_ic import CANDIDATES, _compute_raw

TARGET_HORIZON_BARS = 336  # 14d @ 1h
REGIME_ORDER = ["ALL"] + list(COLLAPSED_REGIMES)


def factor_specs_dict() -> Dict[str, Dict]:
    return {
        c["name"]: {"kind": c["kind"], "lookback": c["lookback"], "skip": c["skip"]}
        for c in CANDIDATES
    }


def max_lookback_bars() -> int:
    specs = factor_specs_dict()
    return max(int(v["lookback"]) + int(v.get("skip", 0)) for v in specs.values())


def precompute_factors_fwd_regimes(
    returns: pd.DataFrame,
    funding: pd.DataFrame,
    prices: pd.DataFrame,
    horizon_bars: int = TARGET_HORIZON_BARS,
) -> Dict[str, object]:
    """全样本一次性算因子截面 z、forward return、regime（walk-forward 每步只做切片）。"""
    fund = funding.reindex(returns.index).fillna(0.0)
    fwd = forward_returns(returns, horizon_bars)
    regimes = compute_regime_labels(prices.reindex(returns.index), fund)
    fac_xs: Dict[str, pd.DataFrame] = {}
    for cand in CANDIDATES:
        raw = _compute_raw(cand, returns, fund)
        raw_w = F.winsorize(raw, 0.02, 0.98)
        fac_xs[cand["name"]] = cross_sectional_zscore(raw_w)
    return {"fwd": fwd, "regimes": regimes, "fac_xs": fac_xs}


def _ic_matrix_for_slice(
    idx_slice: pd.DatetimeIndex,
    fac_xs: Dict[str, pd.DataFrame],
    fwd: pd.DataFrame,
    regimes: pd.DataFrame,
    horizon_bars: int,
    min_regime_samples: int,
    sample_every: int,
    ic_threshold: float,
    all_weights_only: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """在 idx_slice 上计算 IC mean，并生成 regime_weights 字典。

    all_weights_only=True：只算全截面 ALL 的 IC 并只产出 ``ALL`` 一档权重（OOS 不按 regime 切因子，减过拟合）。
    """
    sub_fwd = fwd.loc[idx_slice]
    sub_reg = None if all_weights_only else regimes.reindex(idx_slice).bfill().ffill()

    rows = []
    for name, xs_z in fac_xs.items():
        sub_xs = xs_z.loc[idx_slice]
        ic_all = factor_ic_series(sub_xs, sub_fwd, sample_every=sample_every)
        rows.append({"factor": name, "regime": "ALL", **ic_stats(ic_all)})

        if not all_weights_only and sub_reg is not None:
            for reg in COLLAPSED_REGIMES:
                reg_mask = sub_reg["collapsed"] == reg
                if int(reg_mask.sum()) < min_regime_samples:
                    rows.append(
                        {
                            "factor": name,
                            "regime": reg,
                            "ic_mean": np.nan,
                            "ic_std": np.nan,
                            "ic_ir": np.nan,
                            "ic_hit_rate": np.nan,
                            "n_samples": int(reg_mask.sum()),
                        }
                    )
                    continue
                xs_r = sub_xs.loc[reg_mask]
                fw_r = sub_fwd.reindex(xs_r.index)
                ic_r = factor_ic_series(xs_r, fw_r, sample_every=sample_every)
                rows.append({"factor": name, "regime": reg, **ic_stats(ic_r)})

    df = pd.DataFrame(rows)
    ic_mean_mat = df.pivot_table(index="factor", columns="regime", values="ic_mean")
    cols = (
        ["ALL"]
        if all_weights_only
        else [c for c in REGIME_ORDER if c in ic_mean_mat.columns]
    )
    ic_mean_mat = ic_mean_mat[[c for c in cols if c in ic_mean_mat.columns]]

    weights_yaml: Dict[str, Dict] = {}
    for reg in ic_mean_mat.columns:
        col = ic_mean_mat[reg].dropna()
        eligible = col[col > ic_threshold]
        if eligible.empty:
            fallback = ic_mean_mat["ALL"].dropna().sort_values(ascending=False)
            eligible = fallback[fallback > ic_threshold].head(3)
            note = "fallback_to_ALL_top3"
        else:
            note = (
                "wf_all_only"
                if all_weights_only and reg == "ALL"
                else "regime_conditional"
            )
        total = float(eligible.sum())
        if total <= 0 or eligible.empty:
            weights_yaml[reg] = {
                "factors": {},
                "n_factors": 0,
                "note": "empty_fallback",
            }
            continue
        fac_weights = {f: round(float(v / total), 4) for f, v in eligible.items()}
        weights_yaml[reg] = {
            "factors": fac_weights,
            "n_factors": len(fac_weights),
            "note": note,
        }

    return ic_mean_mat, weights_yaml


def fit_weights_at_rebalance(
    returns: pd.DataFrame,
    t0: pd.Timestamp,
    train_window_bars: int,
    horizon_bars: int,
    min_regime_samples: int,
    sample_every: int,
    ic_threshold: float,
    pre: Dict[str, object],
    all_weights_only: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Dict], Dict[str, int]]:
    """在决策时点 t0：训练窗内 IC 只用到 t0 - horizon 前（fwd 已完全实现）。"""
    idx = returns.index
    i0 = int(idx.searchsorted(t0, side="right")) - 1
    if i0 < 0:
        raise ValueError("t0 before returns start")
    i_train_end_ic = i0 - horizon_bars
    i_train_start = max(0, i0 - train_window_bars)
    idx_slice = idx[i_train_start : i_train_end_ic + 1]
    if len(idx_slice) < horizon_bars + 24:
        return (
            pd.DataFrame(),
            {},
            {
                "i_train_start": i_train_start,
                "i_train_end_ic": i_train_end_ic,
                "i0": i0,
                "n_slice": len(idx_slice),
                "skip": True,
            },
        )

    ic_mean_mat, w = _ic_matrix_for_slice(
        idx_slice,
        pre["fac_xs"],
        pre["fwd"],
        pre["regimes"],
        horizon_bars,
        min_regime_samples,
        sample_every,
        ic_threshold,
        all_weights_only,
    )
    dbg = {
        "i_train_start": i_train_start,
        "i_train_end_ic": i_train_end_ic,
        "i0": i0,
        "n_slice": len(idx_slice),
        "all_weights_only": all_weights_only,
    }
    return ic_mean_mat, w, dbg
