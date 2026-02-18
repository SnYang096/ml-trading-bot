from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


def _ensure_panel(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels != 2:
        raise ValueError(
            "Expected panel with MultiIndex (timestamp, symbol). "
            "Use FactorPanelBuilder to assemble the panel first."
        )
    return df


def winsorize_by_sigma(
    panel: pd.DataFrame,
    columns: Sequence[str],
    sigma: float = 3.0,
    timestamp_level: int = 0,
) -> pd.DataFrame:
    """
    Cross-sectionally winsorize factor columns at +/- sigma standard deviations.

    Args:
        panel: MultiIndex dataframe indexed by (timestamp, symbol).
        columns: Factor columns to clamp.
        sigma: Standard-deviation multiple for clipping.
        timestamp_level: Level index for the timestamp within the MultiIndex.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    panel = _ensure_panel(panel).copy()
    cols = [col for col in columns if col in panel.columns]
    if not cols:
        raise ValueError("winsorize_by_sigma: no valid columns supplied.")

    def _winsorize(group: pd.DataFrame) -> pd.DataFrame:
        for col in cols:
            series = group[col].astype(float)
            mean = series.mean()
            std = series.std(ddof=0)
            if std == 0 or np.isnan(std):
                continue
            upper = mean + sigma * std
            lower = mean - sigma * std
            group[col] = series.clip(lower=lower, upper=upper)
        return group

    level = timestamp_level
    return panel.groupby(level=level, group_keys=False).apply(_winsorize)


def cross_sectional_zscore(
    panel: pd.DataFrame,
    columns: Sequence[str],
    timestamp_level: int = 0,
    ddof: int = 0,
    clip_sigma: Optional[float] = None,
) -> pd.DataFrame:
    """
    Apply cross-sectional z-score normalization per timestamp.

    Args:
        panel: MultiIndex dataframe indexed by (timestamp, symbol).
        columns: Factor columns to normalize.
        timestamp_level: Level index for the timestamp within the MultiIndex.
        ddof: Delta degrees of freedom for standard deviation.
        clip_sigma: Optional post-zscore clipping bounds.
    """
    panel = _ensure_panel(panel).copy()
    cols = [col for col in columns if col in panel.columns]
    if not cols:
        raise ValueError("cross_sectional_zscore: no valid columns supplied.")

    def _zscore(group: pd.DataFrame) -> pd.DataFrame:
        for col in cols:
            series = group[col].astype(float)
            mean = series.mean()
            std = series.std(ddof=ddof)
            if std == 0 or np.isnan(std):
                group[col] = 0.0
            else:
                z = (series - mean) / std
                if clip_sigma:
                    clip_val = abs(float(clip_sigma))
                    z = z.clip(-clip_val, clip_val)
                group[col] = z
        return group

    return panel.groupby(level=timestamp_level, group_keys=False).apply(_zscore)


def cross_sectional_rank(
    panel: pd.DataFrame,
    columns: Sequence[str],
    timestamp_level: int = 0,
    pct: bool = True,
    method: str = "average",
) -> pd.DataFrame:
    """
    Cross-sectional rank transformation per timestamp.

    Args:
        panel: MultiIndex dataframe indexed by (timestamp, symbol).
        columns: Factor columns to rank.
        timestamp_level: Level index for the timestamp within the MultiIndex.
        pct: If True, scale to [0, 1] percentile ranks.
        method: Ranking method passed to pandas.Series.rank.
    """
    panel = _ensure_panel(panel).copy()
    cols = [col for col in columns if col in panel.columns]
    if not cols:
        raise ValueError("cross_sectional_rank: no valid columns supplied.")

    def _rank(group: pd.DataFrame) -> pd.DataFrame:
        for col in cols:
            series = group[col].astype(float)
            group[col] = series.rank(method=method, pct=pct)
        return group

    return panel.groupby(level=timestamp_level, group_keys=False).apply(_rank)


def neutralize_against(
    panel: pd.DataFrame,
    factor_cols: Sequence[str],
    control_cols: Sequence[str],
    timestamp_level: int = 0,
) -> pd.DataFrame:
    """
    Neutralize factor exposures against control columns per timestamp.

    Performs regression for each factor on the control columns and subtracts the fitted values.

    Args:
        panel: MultiIndex dataframe indexed by (timestamp, symbol).
        factor_cols: Factors to neutralize.
        control_cols: Control exposures (e.g. size, industry dummies).
        timestamp_level: Level index for timestamp within the MultiIndex.
    """
    from numpy.linalg import lstsq

    panel = _ensure_panel(panel).copy()
    factors = [c for c in factor_cols if c in panel.columns]
    controls = [c for c in control_cols if c in panel.columns]
    if not factors:
        raise ValueError("neutralize_against: factor_cols not present in panel.")
    if not controls:
        raise ValueError("neutralize_against: control_cols not present in panel.")

    def _neutralize(group: pd.DataFrame) -> pd.DataFrame:
        X = group[controls].values.astype(float)
        X = np.nan_to_num(X, nan=0.0)
        # Add intercept
        X_design = np.concatenate([np.ones((len(group), 1)), X], axis=1)
        for factor in factors:
            y = group[factor].values.astype(float)
            if np.allclose(y, y[0]):
                group[factor] = 0.0
                continue
            y = np.nan_to_num(y, nan=0.0)
            beta, *_ = lstsq(X_design, y, rcond=None)
            fitted = X_design @ beta
            resid = y - fitted
            group[factor] = resid
        return group

    return panel.groupby(level=timestamp_level, group_keys=False).apply(_neutralize)
