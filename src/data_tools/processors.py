"""
Data processors for consistent preprocessing.

This module provides a set of composable processors for data preprocessing,
ensuring consistency across training, backtesting, and live trading.

Usage:
    from src.data_tools.processors import ProcessorChain, FillNaProcessor, ClipOutlierProcessor

    chain = ProcessorChain([
        FillNaProcessor(method="ffill"),
        ClipOutlierProcessor(std_threshold=5.0),
        DtypeDowncastProcessor(),
    ])
    df_processed = chain.process(df)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any

import numpy as np
import pandas as pd


class Processor(ABC):
    """Base class for data processors."""

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Processed DataFrame
        """
        pass

    @property
    def name(self) -> str:
        """Return processor name for logging."""
        return self.__class__.__name__


class FillNaProcessor(Processor):
    """
    Fill NaN values in numeric columns.

    Supports multiple fill methods:
    - "ffill": Forward fill (carry last valid value forward)
    - "bfill": Backward fill
    - "zero": Fill with 0
    - "mean": Fill with column mean
    - "median": Fill with column median
    """

    def __init__(
        self,
        method: str = "ffill",
        columns: Optional[List[str]] = None,
        fill_value: Optional[float] = None,
    ):
        """
        Initialize FillNaProcessor.

        Args:
            method: Fill method ("ffill", "bfill", "zero", "mean", "median")
            columns: Specific columns to fill (None = all numeric columns)
            fill_value: Custom fill value (overrides method if provided)
        """
        self.method = method
        self.columns = columns
        self.fill_value = fill_value

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Determine which columns to process
        if self.columns:
            cols = [c for c in self.columns if c in df.columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if self.fill_value is not None:
                df[col] = df[col].fillna(self.fill_value)
            elif self.method == "ffill":
                df[col] = df[col].ffill()
            elif self.method == "bfill":
                df[col] = df[col].bfill()
            elif self.method == "zero":
                df[col] = df[col].fillna(0.0)
            elif self.method == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif self.method == "median":
                df[col] = df[col].fillna(df[col].median())

        return df


class ClipOutlierProcessor(Processor):
    """
    Clip outliers based on standard deviation or percentile.

    Modes:
    - "std": Clip values beyond mean ± std_threshold * std
    - "percentile": Clip values beyond [lower_pct, upper_pct] percentiles
    - "iqr": Clip values beyond Q1 - 1.5*IQR and Q3 + 1.5*IQR
    """

    def __init__(
        self,
        mode: str = "std",
        std_threshold: float = 5.0,
        lower_pct: float = 0.01,
        upper_pct: float = 99.99,
        columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
    ):
        """
        Initialize ClipOutlierProcessor.

        Args:
            mode: Clipping mode ("std", "percentile", "iqr")
            std_threshold: Number of std deviations for "std" mode
            lower_pct: Lower percentile for "percentile" mode
            upper_pct: Upper percentile for "percentile" mode
            columns: Specific columns to clip (None = all numeric columns)
            exclude_columns: Columns to exclude from clipping
        """
        self.mode = mode
        self.std_threshold = std_threshold
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.columns = columns
        self.exclude_columns = exclude_columns or []

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Determine which columns to process
        if self.columns:
            cols = [c for c in self.columns if c in df.columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove excluded columns
        cols = [c for c in cols if c not in self.exclude_columns]

        for col in cols:
            series = df[col]
            valid_data = series.dropna()

            if len(valid_data) == 0:
                continue

            if self.mode == "std":
                mean = valid_data.mean()
                std = valid_data.std()
                if std > 0:
                    lower = mean - self.std_threshold * std
                    upper = mean + self.std_threshold * std
                    df[col] = series.clip(lower=lower, upper=upper)

            elif self.mode == "percentile":
                lower = valid_data.quantile(self.lower_pct / 100)
                upper = valid_data.quantile(self.upper_pct / 100)
                df[col] = series.clip(lower=lower, upper=upper)

            elif self.mode == "iqr":
                q1 = valid_data.quantile(0.25)
                q3 = valid_data.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                df[col] = series.clip(lower=lower, upper=upper)

        return df


class DtypeDowncastProcessor(Processor):
    """
    Downcast numeric dtypes to reduce memory usage.

    Converts:
    - float64 → float32
    - int64 → int32 (if values fit)
    """

    def __init__(
        self,
        float_dtype: str = "float32",
        int_dtype: str = "int32",
        columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
    ):
        """
        Initialize DtypeDowncastProcessor.

        Args:
            float_dtype: Target float dtype
            int_dtype: Target int dtype
            columns: Specific columns to downcast (None = all numeric)
            exclude_columns: Columns to exclude from downcasting
        """
        self.float_dtype = float_dtype
        self.int_dtype = int_dtype
        self.columns = columns
        self.exclude_columns = exclude_columns or []

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Determine which columns to process
        if self.columns:
            cols = [c for c in self.columns if c in df.columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove excluded columns
        cols = [c for c in cols if c not in self.exclude_columns]

        for col in cols:
            if df[col].dtype == np.float64:
                df[col] = df[col].astype(self.float_dtype)
            elif df[col].dtype == np.int64:
                # Check if values fit in int32
                if (
                    df[col].min() >= np.iinfo(np.int32).min
                    and df[col].max() <= np.iinfo(np.int32).max
                ):
                    df[col] = df[col].astype(self.int_dtype)

        return df


class ReplaceInfProcessor(Processor):
    """
    Replace inf/-inf values with NaN or specified values.
    """

    def __init__(
        self,
        replace_with: str = "nan",
        pos_inf_value: Optional[float] = None,
        neg_inf_value: Optional[float] = None,
        columns: Optional[List[str]] = None,
    ):
        """
        Initialize ReplaceInfProcessor.

        Args:
            replace_with: What to replace inf with ("nan", "clip", "value")
            pos_inf_value: Custom value for +inf (when replace_with="value")
            neg_inf_value: Custom value for -inf (when replace_with="value")
            columns: Specific columns to process (None = all numeric)
        """
        self.replace_with = replace_with
        self.pos_inf_value = pos_inf_value
        self.neg_inf_value = neg_inf_value
        self.columns = columns

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Determine which columns to process
        if self.columns:
            cols = [c for c in self.columns if c in df.columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if self.replace_with == "nan":
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            elif self.replace_with == "clip":
                # Replace with max/min finite values
                finite_vals = df[col][np.isfinite(df[col])]
                if len(finite_vals) > 0:
                    max_val = finite_vals.max()
                    min_val = finite_vals.min()
                    df[col] = df[col].replace(np.inf, max_val)
                    df[col] = df[col].replace(-np.inf, min_val)
            elif self.replace_with == "value":
                if self.pos_inf_value is not None:
                    df[col] = df[col].replace(np.inf, self.pos_inf_value)
                if self.neg_inf_value is not None:
                    df[col] = df[col].replace(-np.inf, self.neg_inf_value)

        return df


class ProcessorChain(Processor):
    """
    Chain multiple processors together.

    Processors are applied in order, with each processor receiving
    the output of the previous one.
    """

    def __init__(self, processors: Optional[List[Processor]] = None):
        """
        Initialize ProcessorChain.

        Args:
            processors: List of processors to apply in order
        """
        self.processors = processors or []

    def add(self, processor: Processor) -> "ProcessorChain":
        """Add a processor to the chain."""
        self.processors.append(processor)
        return self

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all processors in sequence."""
        result = df
        for proc in self.processors:
            result = proc.process(result)
        return result

    @property
    def name(self) -> str:
        names = [p.name for p in self.processors]
        return f"ProcessorChain({' -> '.join(names)})"


# Pre-configured processor chains for common use cases
def get_default_processor_chain() -> ProcessorChain:
    """
    Get the default processor chain for training/backtesting.

    Chain:
    1. Replace inf with NaN
    2. Forward-fill NaN
    3. Clip outliers (5 std)
    4. Downcast to float32
    """
    return ProcessorChain(
        [
            ReplaceInfProcessor(replace_with="nan"),
            FillNaProcessor(method="ffill"),
            ClipOutlierProcessor(mode="std", std_threshold=5.0),
            DtypeDowncastProcessor(float_dtype="float32"),
        ]
    )


def get_live_processor_chain() -> ProcessorChain:
    """
    Get processor chain for live trading (more conservative).

    Chain:
    1. Replace inf with NaN
    2. Forward-fill NaN (limited lookback in production)
    3. Clip outliers (3 std - more aggressive)
    """
    return ProcessorChain(
        [
            ReplaceInfProcessor(replace_with="nan"),
            FillNaProcessor(method="ffill"),
            ClipOutlierProcessor(mode="std", std_threshold=3.0),
        ]
    )
