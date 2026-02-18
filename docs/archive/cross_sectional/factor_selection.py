from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def filter_panel_by_assets(
    panel: pd.DataFrame,
    *,
    min_assets: int = 2,
    timestamp_level: int = 0,
) -> pd.DataFrame:
    if min_assets <= 1:
        return panel
    counts = panel.groupby(level=timestamp_level).size()
    valid_index = counts[counts >= min_assets].index
    return panel.loc[pd.IndexSlice[valid_index, :], :]


def compute_cross_sectional_ic(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    target_col: str,
    *,
    min_assets: int = 2,
    timestamp_level: int = 0,
    symbol_level: int = 1,
) -> pd.DataFrame:
    """
    Compute per-factor rank IC statistics across timestamps.
    """
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("Panel must be a MultiIndex with (timestamp, symbol).")

    # Ensure factor list is stable-unique to avoid duplicate metric rows / non-unique indices.
    factor_cols = list(dict.fromkeys([str(c) for c in factor_cols if str(c)]))

    panel = filter_panel_by_assets(
        panel, min_assets=min_assets, timestamp_level=timestamp_level
    )

    columns = [target_col] + list(factor_cols)
    available_cols = [c for c in columns if c in panel.columns]
    subset = panel[available_cols].dropna(subset=[target_col])

    if subset.empty:
        return pd.DataFrame(columns=["ic_mean", "ic_std", "ic_ir", "ic_count"])

    grouped = list(subset.groupby(level=timestamp_level))
    results: List[Dict[str, float]] = []

    for factor in factor_cols:
        if factor not in subset.columns:
            continue

        ic_values: List[float] = []
        for _, group in grouped:
            if factor not in group.columns:
                continue
            pair = group[[factor, target_col]].dropna()
            if pair.empty or len(pair) < min_assets:
                continue

            x = pair[factor]
            y = pair[target_col]
            # Guard: if x is not a Series (e.g. duplicate columns / pandas edge cases),
            # reduce deterministically to a 1D Series.
            if isinstance(x, pd.DataFrame):
                x = x.iloc[:, 0]
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]

            if getattr(x, "nunique", None) is None or int(x.nunique()) < min(
                3, min_assets
            ):
                continue

            corr = x.corr(y, method="spearman")
            if pd.notna(corr):
                ic_values.append(float(corr))

        if not ic_values:
            continue

        arr = np.asarray(ic_values)
        ic_mean = float(arr.mean())
        ic_std = float(arr.std(ddof=0))
        ic_ir = ic_mean / ic_std if ic_std > 0 else float("nan")
        results.append(
            {
                "factor": factor,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_ir": ic_ir,
                "ic_count": float(len(arr)),
            }
        )

    if not results:
        return pd.DataFrame(columns=["ic_mean", "ic_std", "ic_ir", "ic_count"])

    metrics = pd.DataFrame(results).set_index("factor")
    metrics.sort_values(by="ic_mean", ascending=False, inplace=True)
    return metrics


def apply_factor_selection(
    metrics: pd.DataFrame,
    original_factors: Sequence[str],
    *,
    select_topk: int,
    ic_threshold: Optional[float],
    ir_threshold: Optional[float],
    ranking_stat: str,
) -> List[str]:
    if metrics.empty:
        return list(original_factors)

    selected = metrics.copy()

    if ic_threshold is not None:
        selected = selected[selected["ic_mean"].abs() >= float(ic_threshold)]

    if ir_threshold is not None and "ic_ir" in selected.columns:
        selected = selected[selected["ic_ir"].abs() >= float(ir_threshold)]

    if select_topk and select_topk > 0:
        key = (
            "ic_ir"
            if ranking_stat == "ir" and "ic_ir" in selected.columns
            else "ic_mean"
        )
        selected = selected.sort_values(by=key, ascending=False).head(select_topk)

    selected_factors = [
        factor for factor in selected.index if factor in original_factors
    ]
    return selected_factors
