from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def _group_ic(
    panel: pd.DataFrame,
    factor_col: str,
    target_col: str,
    timestamp_level: int = 0,
) -> pd.Series:

    def _ic(group: pd.DataFrame) -> float:
        if group[factor_col].nunique(
                dropna=True) < 2 or group[target_col].nunique(dropna=True) < 2:
            return np.nan
        return group[[factor_col, target_col]].corr(method="spearman").iloc[0,
                                                                            1]

    return panel.groupby(level=timestamp_level, group_keys=False).apply(_ic)


def rolling_ic(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    target_col: str,
    window: int = 120,
    timestamp_level: int = 0,
) -> Dict[str, pd.Series]:
    rolling: Dict[str, pd.Series] = {}
    for col in factor_cols:
        ic_series = _group_ic(panel,
                              col,
                              target_col,
                              timestamp_level=timestamp_level)
        rolling[col] = ic_series.rolling(window,
                                         min_periods=max(10,
                                                         window // 4)).mean()
    return rolling


def ic_stability_metrics(
    ic_series: pd.Series,
    half_life: Optional[int] = None,
) -> Dict[str, float]:
    ic_clean = ic_series.dropna()
    if ic_clean.empty:
        return {
            "mean_ic": np.nan,
            "std_ic": np.nan,
            "ic_ir": np.nan,
            "ewm_ic": np.nan
        }
    mean_ic = float(ic_clean.mean())
    std_ic = float(ic_clean.std(ddof=1))
    ic_ir = float(mean_ic / std_ic) if std_ic > 0 else np.nan
    if half_life:
        ewm_ic = float(ic_clean.ewm(halflife=half_life).mean().iloc[-1])
    else:
        ewm_ic = mean_ic
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ic_ir": ic_ir,
        "ewm_ic": ewm_ic
    }


def holm_bonferroni(p_values: Mapping[str, float]) -> Dict[str, float]:
    items = sorted(p_values.items(), key=lambda x: x[1])
    adjusted: Dict[str, float] = {}
    m = len(items)
    for i, (name, pv) in enumerate(items, start=1):
        adjusted[name] = min(1.0, pv * (m - i + 1))
    return adjusted


def governance_report(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    target_col: str,
    window: int = 120,
    timestamp_level: int = 0,
    half_life: Optional[int] = 90,
) -> pd.DataFrame:
    metrics_rows: List[Dict[str, float]] = []
    p_value_map: Dict[str, float] = {}
    for col in factor_cols:
        ic_series = _group_ic(panel,
                              col,
                              target_col,
                              timestamp_level=timestamp_level).dropna()
        if ic_series.empty:
            continue
        stats_dict = ic_stability_metrics(ic_series, half_life=half_life)
        t_stat, p_value = stats.ttest_1samp(ic_series.values,
                                            0.0,
                                            nan_policy="omit")
        p_value_map[col] = float(p_value) if np.isfinite(p_value) else 1.0
        row = {
            "factor": col,
            "observations": float(ic_series.count()),
            "mean_ic": stats_dict["mean_ic"],
            "std_ic": stats_dict["std_ic"],
            "ic_ir": stats_dict["ic_ir"],
            "ewm_ic": stats_dict["ewm_ic"],
            "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "p_value": p_value_map[col],
            "positive_rate": float((ic_series > 0).mean()),
        }
        metrics_rows.append(row)

    if not metrics_rows:
        return pd.DataFrame(columns=[
            "factor",
            "observations",
            "mean_ic",
            "std_ic",
            "ic_ir",
            "ewm_ic",
            "t_stat",
            "p_value",
            "p_value_holm",
            "positive_rate",
        ])

    metrics_df = pd.DataFrame(metrics_rows).set_index("factor")
    adjusted = holm_bonferroni(p_value_map)
    metrics_df["p_value_holm"] = pd.Series(adjusted)
    return metrics_df.sort_values("mean_ic", ascending=False)


def flag_unstable_factors(
    metrics: pd.DataFrame,
    min_ic: float = 0.02,
    min_ic_ir: float = 0.2,
    max_pvalue: float = 0.1,
) -> List[str]:
    unstable: List[str] = []
    for factor, row in metrics.iterrows():
        if (row.get("mean_ic", np.nan) < min_ic
                or row.get("ic_ir", np.nan) < min_ic_ir
                or row.get("p_value_holm", np.nan) > max_pvalue):
            unstable.append(factor)
    return unstable
