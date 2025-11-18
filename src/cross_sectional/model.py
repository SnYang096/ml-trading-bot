from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class FactorBacktestResult:
    """Container for cross-sectional regression diagnostics."""

    factor_returns: pd.DataFrame
    intercept: pd.Series
    residuals: pd.DataFrame
    r2: pd.Series
    information_coefficients: pd.DataFrame
    valid_timestamps: List[pd.Timestamp] = field(default_factory=list)

    def factor_summary(self, periods_per_year: int = 252) -> pd.DataFrame:
        """Return mean, std, t-stat and annualised metrics for factor returns."""
        stats = []
        for factor in self.factor_returns.columns:
            series = self.factor_returns[factor].dropna()
            if series.empty:
                stats.append(
                    {
                        "factor": factor,
                        "mean": np.nan,
                        "std": np.nan,
                        "t_stat": np.nan,
                        "ann_mean": np.nan,
                        "ann_vol": np.nan,
                        "ir": np.nan,
                    }
                )
                continue
            mean = float(series.mean())
            std = float(series.std(ddof=0))
            t_stat = mean / (std / np.sqrt(len(series))) if std > 0 else np.nan
            ann_mean = mean * periods_per_year
            ann_vol = std * np.sqrt(periods_per_year)
            ir = ann_mean / ann_vol if ann_vol > 0 else np.nan
            stats.append(
                {
                    "factor": factor,
                    "mean": mean,
                    "std": std,
                    "t_stat": t_stat,
                    "ann_mean": ann_mean,
                    "ann_vol": ann_vol,
                    "ir": ir,
                }
            )
        return pd.DataFrame(stats).set_index("factor")

    def ic_summary(
        self,
        periods_per_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return mean, std, t-statistics and IR-style metrics for the IC series."""
        stats = []
        for factor in self.information_coefficients.columns:
            series = self.information_coefficients[factor].dropna()
            if series.empty:
                stats.append(
                    {
                        "factor": factor,
                        "ic_mean": np.nan,
                        "ic_std": np.nan,
                        "ic_t": np.nan,
                        "ic_ir": np.nan,
                        "ic_ir_annual": np.nan,
                    }
                )
                continue
            mean = float(series.mean())
            std = float(series.std(ddof=0))
            t_stat = mean / (std / np.sqrt(len(series))) if std > 0 else np.nan
            ic_ir = mean / std if std > 0 else np.nan
            if periods_per_year and not np.isnan(ic_ir):
                ic_ir_annual = ic_ir * np.sqrt(periods_per_year)
            else:
                ic_ir_annual = np.nan
            stats.append(
                {
                    "factor": factor,
                    "ic_mean": mean,
                    "ic_std": std,
                    "ic_t": t_stat,
                    "ic_ir": ic_ir,
                    "ic_ir_annual": ic_ir_annual,
                }
            )
        return pd.DataFrame(stats).set_index("factor")

    def newey_west_summary(
        self,
        max_lag: int = 5,
        periods_per_year: int = 252,
    ) -> pd.DataFrame:
        """
        Compute Newey-West adjusted standard errors for factor premia.

        Args:
            max_lag: Bartlett kernel truncation lag.
            periods_per_year: Annualisation factor for means/vols.
        """
        if max_lag < 0:
            raise ValueError("max_lag must be non-negative")

        rows = []
        for column in self.factor_returns.columns:
            series = self.factor_returns[column].dropna()
            if len(series) < 2:
                rows.append(
                    {
                        "factor": column,
                        "mean": np.nan,
                        "nw_se": np.nan,
                        "nw_t": np.nan,
                        "ann_mean": np.nan,
                        "ann_vol": np.nan,
                        "ir": np.nan,
                    }
                )
                continue
            mean = float(series.mean())
            se = float(_newey_west_se(series.values, max_lag=max_lag))
            ann_mean = mean * periods_per_year
            ann_vol = float(series.std(ddof=0)) * np.sqrt(periods_per_year)
            t_stat = mean / se if se > 0 else np.nan
            ir = ann_mean / ann_vol if ann_vol > 0 else np.nan
            rows.append(
                {
                    "factor": column,
                    "mean": mean,
                    "nw_se": se,
                    "nw_t": t_stat,
                    "ann_mean": ann_mean,
                    "ann_vol": ann_vol,
                    "ir": ir,
                }
            )

        return pd.DataFrame(rows).set_index("factor")

    def combined_metrics(
        self,
        max_lag: int = 5,
        periods_per_year: int = 252,
    ) -> pd.DataFrame:
        """
        Aggregate Newey-West, factor IR, and IC metrics into a single table.
        """
        nw = self.newey_west_summary(max_lag=max_lag, periods_per_year=periods_per_year)
        ic = self.ic_summary(periods_per_year=periods_per_year)
        summary = nw.join(ic, how="outer")
        summary.index.name = "factor"
        return summary


class CrossSectionalRegressor:
    """
    Estimate factor premia via cross-sectional regressions.

    Implements rolling Fama-MacBeth style regressions where each timestamp is
    treated as an independent cross-section.
    """

    def __init__(
        self,
        add_intercept: bool = True,
        min_assets: int = 5,
        max_condition_number: Optional[float] = 1e6,
        timestamp_level: int = 0,
    ):
        self.add_intercept = add_intercept
        self.min_assets = min_assets
        self.max_condition_number = max_condition_number
        self.timestamp_level = timestamp_level
        self.factor_names: List[str] = []
        self.result_: Optional[FactorBacktestResult] = None

    def fit(
        self,
        panel: pd.DataFrame,
        factor_cols: Sequence[str],
        target_col: str,
    ) -> FactorBacktestResult:
        if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
            raise ValueError("panel must have MultiIndex (timestamp, symbol).")

        factors = [c for c in factor_cols if c in panel.columns]
        if not factors:
            raise ValueError("No factor columns present in panel.")
        if target_col not in panel.columns:
            raise ValueError(f"Target column '{target_col}' missing from panel.")

        timestamp_level = self.timestamp_level
        timestamps = panel.index.get_level_values(timestamp_level).unique()

        factor_returns_records: List[pd.Series] = []
        intercept_records: List[Tuple[pd.Timestamp, float]] = []
        residual_records: List[pd.Series] = []
        r2_records: List[Tuple[pd.Timestamp, float]] = []
        ic_records: Dict[str, List[Tuple[pd.Timestamp, float]]] = {
            f: [] for f in factors
        }
        valid_timestamps: List[pd.Timestamp] = []

        for ts in timestamps:
            cross_section = panel.xs(ts, level=timestamp_level).copy()

            cross_section = cross_section.replace([np.inf, -np.inf], np.nan).dropna(
                subset=[target_col] + factors, how="any"
            )
            if len(cross_section) < max(
                self.min_assets, len(factors) + int(self.add_intercept)
            ):
                continue

            X = cross_section[factors].values.astype(float)
            y = cross_section[target_col].values.astype(float)

            X = np.nan_to_num(X, nan=0.0)
            y = np.nan_to_num(y, nan=0.0)

            if self.add_intercept:
                X_design = np.concatenate([np.ones((len(X), 1)), X], axis=1)
            else:
                X_design = X

            cond_number = np.linalg.cond(X_design)
            if self.max_condition_number and cond_number > self.max_condition_number:
                continue

            beta, residuals, rank, sing_vals = np.linalg.lstsq(X_design, y, rcond=None)

            intercept = beta[0] if self.add_intercept else 0.0
            factor_returns = beta[1:] if self.add_intercept else beta

            fitted = X_design @ beta
            resid = y - fitted
            sse = float(np.sum(resid**2))
            sst = float(np.sum((y - y.mean()) ** 2))
            r_squared = 1.0 - sse / sst if sst > 0 else np.nan

            factor_returns_records.append(
                pd.Series(factor_returns, index=factors, name=ts)
            )
            intercept_records.append((ts, intercept))
            residual_records.append(
                pd.Series(resid, index=cross_section.index, name=ts)
            )
            r2_records.append((ts, r_squared))
            valid_timestamps.append(ts)

            for idx, factor in enumerate(factors):
                exposures = cross_section[factor].values
                if exposures.std(ddof=0) == 0:
                    ic = np.nan
                else:
                    ic = np.corrcoef(exposures, y)[0, 1]
                ic_records[factor].append((ts, float(ic)))

        if not factor_returns_records:
            raise RuntimeError("No valid cross-sectional regressions could be fit.")

        factor_returns_df = pd.DataFrame(factor_returns_records)
        intercept_series = pd.Series({ts: val for ts, val in intercept_records})
        residual_panel = pd.concat(residual_records, axis=0)
        r2_series = pd.Series({ts: val for ts, val in r2_records}).sort_index()
        ic_df = pd.DataFrame(
            {factor: pd.Series(dict(records)) for factor, records in ic_records.items()}
        ).sort_index()

        result = FactorBacktestResult(
            factor_returns=factor_returns_df.sort_index(),
            intercept=intercept_series.sort_index(),
            residuals=residual_panel.sort_index(),
            r2=r2_series,
            information_coefficients=ic_df,
            valid_timestamps=valid_timestamps,
        )
        self.result_ = result
        self.factor_names = factors
        return result

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        """
        Predict expected returns using average factor premia.

        Args:
            panel: Panel with factor exposures for a single timestamp slice or MultiIndex.

        Returns:
            pd.Series of expected returns indexed by the same index as the input panel.
        """
        if self.result_ is None:
            raise RuntimeError("Model must be fit before calling predict.")
        avg_factor_premia = self.result_.factor_returns.mean()
        avg_intercept = self.result_.intercept.mean()

        if isinstance(panel.index, pd.MultiIndex):
            exposures = panel[self.factor_names]
            predicted = exposures @ avg_factor_premia
            predicted = predicted + avg_intercept
            return predicted

        # Single timestamp exposures (DataFrame indexed by symbol)
        exposures = panel[self.factor_names]
        predicted = exposures @ avg_factor_premia
        return predicted + avg_intercept


def _newey_west_se(values: np.ndarray, max_lag: int = 5) -> float:
    """
    Compute Newey-West standard error for the sample mean.

    Args:
        values: 1D numpy array of factor returns.
        max_lag: Truncation lag for Bartlett kernel.
    """
    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    T = series.shape[0]
    if T <= 1:
        return np.nan

    mean = series.mean()
    residuals = series - mean
    gamma0 = np.dot(residuals, residuals) / T
    var = gamma0
    max_lag = int(min(max_lag, T - 1))
    if max_lag <= 0:
        return np.sqrt(var / T)

    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        cov = np.dot(residuals[lag:], residuals[:-lag]) / T
        var += 2.0 * weight * cov

    return np.sqrt(var / T)
