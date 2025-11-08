from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class PanelConfig:
    """
    Configuration for assembling a multi-asset factor panel.

    Attributes:
        timestamp_col: Name of the timestamp column in raw frames.
        symbol_col: Name of the symbol/asset identifier column.
        feature_cols: Optional explicit list of feature columns to retain.
        target_col: Column name for the forward return target.
        forward_return_horizon: Number of periods to look ahead for returns.
        min_assets_per_ts: Minimum distinct assets required per timestamp.
        fill_method: Missing data handling strategy ('ffill', 'bfill', 'zero', or None).
        align_intersection_only: If True, drop timestamps lacking complete asset coverage.
        check_duplicates: Whether to check for duplicate (timestamp, symbol) rows.
    """

    timestamp_col: str = "timestamp"
    symbol_col: str = "symbol"
    feature_cols: Optional[Sequence[str]] = None
    target_col: str = "future_return"
    forward_return_horizon: Optional[int] = None
    min_assets_per_ts: int = 2
    fill_method: Optional[str] = "ffill"
    align_intersection_only: bool = False
    check_duplicates: bool = True
    sort_index: bool = True
    dropna_after_fill: bool = True

    def feature_list(self, df_cols: Iterable[str]) -> List[str]:
        """Resolve feature columns given a dataframe column set."""
        if self.feature_cols:
            return [col for col in self.feature_cols if col in df_cols]
        exclude = {
            self.timestamp_col,
            self.symbol_col,
            self.target_col,
        }
        return [
            col for col in df_cols if col not in exclude and not col.startswith("future_return")
        ]


class FactorPanelBuilder:
    """Construct aligned factor panels for cross-sectional analysis."""

    def __init__(self, config: PanelConfig):
        self.config = config

    def from_symbol_frames(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Build a panel from a dict mapping symbol->dataframe.

        Each dataframe should be indexed by timestamp or contain the configured
        timestamp column. Feature columns are intersected across assets by default.
        """
        frames: List[pd.DataFrame] = []
        for symbol, df in data.items():
            if df is None or df.empty:
                continue
            frame = df.copy()
            if self.config.timestamp_col in frame.columns:
                frame = frame.set_index(self.config.timestamp_col)
            if not isinstance(frame.index, pd.DatetimeIndex):
                raise ValueError(
                    f"{symbol}: expected DatetimeIndex or column '{self.config.timestamp_col}'."
                )
            frame[self.config.symbol_col] = symbol
            frames.append(frame)

        if not frames:
            raise ValueError("No valid symbol data provided.")

        combined = pd.concat(frames, axis=0)
        combined.index.name = self.config.timestamp_col
        combined_reset = combined.reset_index()
        return self._finalize_panel(combined_reset)

    def from_concat_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build a panel from a dataframe already containing symbol + timestamp."""
        if self.config.timestamp_col not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": self.config.timestamp_col})
            else:
                raise ValueError(f"Dataframe must include '{self.config.timestamp_col}'.")
        if self.config.symbol_col not in df.columns:
            raise ValueError(f"Dataframe must include '{self.config.symbol_col}'.")
        return self._finalize_panel(df.copy())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _finalize_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        df = df.copy()

        # Ensure timestamp column type
        df[cfg.timestamp_col] = pd.to_datetime(df[cfg.timestamp_col], utc=True, errors="coerce")
        if df[cfg.timestamp_col].isna().any():
            raise ValueError("NaT values detected in timestamp column after coercion.")

        # Optional duplicate check
        if cfg.check_duplicates:
            duplicates = df.duplicated(subset=[cfg.timestamp_col, cfg.symbol_col])
            if duplicates.any():
                dup_rows = df.loc[duplicates, [cfg.timestamp_col, cfg.symbol_col]]
                raise ValueError(
                    f"Duplicate (timestamp, symbol) entries detected:\n{dup_rows.head()}"
                )

        # Apply forward return computation if requested
        if cfg.forward_return_horizon and cfg.target_col not in df.columns:
            df = self._compute_forward_return(df, cfg.forward_return_horizon)

        # Determine features
        feature_cols = cfg.feature_list(df.columns)
        if not feature_cols:
            raise ValueError("No feature columns detected for panel assembly.")

        # Sort to guarantee deterministic ordering
        if cfg.sort_index:
            df = df.sort_values([cfg.timestamp_col, cfg.symbol_col])

        # Pivot into MultiIndex panel
        df = df.set_index([cfg.timestamp_col, cfg.symbol_col])

        # Fill missing values per asset if requested
        if cfg.fill_method:
            df = self._apply_fill(df, feature_cols + [cfg.target_col])

        # Drop rows without enough assets
        panel = df.copy()
        if cfg.align_intersection_only:
            asset_counts = panel.groupby(level=cfg.timestamp_col).size()
            required_dates = asset_counts[asset_counts >= cfg.min_assets_per_ts].index
            panel = panel.loc[pd.IndexSlice[required_dates, :], :]
        else:
            panel = self._filter_by_min_assets(panel, cfg.min_assets_per_ts)

        if cfg.dropna_after_fill:
            panel = panel.dropna(subset=feature_cols, how="any")

        # Ensure target exists (may be NaN if horizon implies tail removal)
        if cfg.target_col in panel.columns:
            panel = panel.dropna(subset=[cfg.target_col], how="any")

        return panel

    def _apply_fill(self, panel: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        cfg = self.config
        filled = panel.copy()
        method = cfg.fill_method
        if method in {"ffill", "bfill"}:
            filled = filled.groupby(level=self.config.symbol_col).apply(
                lambda grp: grp.sort_index(level=self.config.timestamp_col).fillna(method=method)
            )
            # Groupby adds index level; remove it
            filled.index = filled.index.droplevel(0)
        elif method == "zero":
            filled[columns] = filled[columns].fillna(0.0)
        elif method is None:
            return filled
        else:
            raise ValueError(f"Unsupported fill method: {method}")
        return filled

    def _filter_by_min_assets(self, panel: pd.DataFrame, min_assets: int) -> pd.DataFrame:
        if min_assets <= 1:
            return panel
        counts = panel.groupby(level=self.config.timestamp_col).size()
        valid_index = counts[counts >= min_assets].index
        return panel.loc[pd.IndexSlice[valid_index, :], :]

    def _compute_forward_return(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        cfg = self.config
        if horizon <= 0:
            raise ValueError("forward_return_horizon must be positive.")
        df = df.copy()
        df = df.sort_values([cfg.symbol_col, cfg.timestamp_col])

        def _calc(symbol_df: pd.DataFrame) -> pd.DataFrame:
            if "close" not in symbol_df.columns:
                raise ValueError("Forward return computation requires 'close' column.")
            price = symbol_df["close"].astype(float)
            future_price = price.shift(-horizon)
            returns = future_price / price - 1.0
            symbol_df[cfg.target_col] = returns
            return symbol_df.iloc[:-horizon] if horizon > 0 else symbol_df

        grouped = df.groupby(cfg.symbol_col, group_keys=False).apply(_calc)
        if grouped[cfg.target_col].isna().any():
            grouped = grouped.dropna(subset=[cfg.target_col])
        return grouped

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def describe_panel(panel: pd.DataFrame) -> Dict[str, float]:
        """Return basic diagnostics for the assembled panel."""
        if not isinstance(panel.index, pd.MultiIndex):
            raise ValueError("Panel must have MultiIndex (timestamp, symbol).")
        timestamps = panel.index.get_level_values(0)
        symbols = panel.index.get_level_values(1)
        return {
            "num_observations": float(len(panel)),
            "num_timestamps": float(timestamps.nunique()),
            "num_symbols": float(symbols.nunique()),
            "mean_assets_per_timestamp": float(
                len(panel) / max(1, timestamps.nunique())
            ),
        }

