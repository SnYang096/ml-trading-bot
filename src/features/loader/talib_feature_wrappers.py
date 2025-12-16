"""
TA-Lib 特征包装函数

为单个 TA-Lib 指标创建包装函数，支持按需计算
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import talib


def _prepare_inputs(
    df: Optional[pd.DataFrame], kwargs: Dict[str, object]
) -> tuple[Dict[str, object], pd.Index]:
    """Convert pandas inputs to numpy arrays and infer index."""

    processed: Dict[str, object] = {}
    index: Optional[pd.Index] = None

    for key, value in kwargs.items():
        if isinstance(value, pd.Series):
            # TA-Lib expects float64 ("double") inputs for most indicators.
            processed[key] = (
                pd.to_numeric(value, errors="coerce").astype("float64").values
            )
            if index is None:
                index = value.index
        elif isinstance(value, pd.DataFrame):
            processed[key] = value.apply(pd.to_numeric, errors="coerce").astype(
                "float64"
            ).values
            if index is None:
                index = value.index
        else:
            processed[key] = value

    if index is None:
        if df is None:
            raise ValueError(
                "Unable to infer index for TA-Lib inputs: no pandas Series/DataFrame provided."
            )
        index = df.index

    return processed, index


def compute_talib_indicator(
    df: pd.DataFrame,
    indicator_name: str,
    *,
    output_column: Optional[str] = None,
    output_columns: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    通用 TA-Lib 指标计算函数，可通过 column_mappings 传入任意参数。

    YAML 中可通过 compute_params 指定 indicator_name/timeperiod，
    column_mappings 映射 DataFrame 列到 talib 函数参数（real/high/low/close 等）。
    """

    result = df.copy()

    talib_func = getattr(talib, indicator_name, None)
    if talib_func is None:
        raise ValueError(f"Unknown TA-Lib indicator: {indicator_name}")

    processed_kwargs, index = _prepare_inputs(df, kwargs)

    try:
        talib_result = talib_func(**processed_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Error computing TA-Lib indicator '{indicator_name}': {exc}"
        ) from exc

    if isinstance(talib_result, tuple):
        cols = output_columns or [
            f"{indicator_name.lower()}_{i}" for i in range(len(talib_result))
        ]
        if len(cols) != len(talib_result):
            raise ValueError(
                f"Output columns mismatch for {indicator_name}: "
                f"expected {len(talib_result)}, got {len(cols)}"
            )
        for col_name, series in zip(cols, talib_result):
            result[col_name] = pd.Series(series, index=index)
    else:
        col = output_column or indicator_name.lower()
        result[col] = pd.Series(talib_result, index=index)

    return result


def compute_talib_indicator_from_series(
    *,
    indicator_name: str,
    output_column: Optional[str] = None,
    output_columns: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Narrow-IO entrypoint for TA-Lib indicators.

    Feature pipeline can call this with `pass_full_df: false` + `column_mappings`
    so only required Series are passed (no wide DataFrame).
    """
    # Infer index from provided Series/DataFrame args, then call the canonical implementation.
    _, index = _prepare_inputs(None, kwargs)
    df = pd.DataFrame(index=index)
    return compute_talib_indicator(
        df,
        indicator_name,
        output_column=output_column,
        output_columns=output_columns,
        **kwargs,
    )


def compute_talib_sma(
    df: pd.DataFrame,
    period: int = 20,
    series: Optional[pd.Series] = None,
    output_column: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    if series is None:
        series = df["close"]
    if output_column is None:
        output_column = f"sma_{period}"

    return compute_talib_indicator(
        df,
        "SMA",
        real=series,
        timeperiod=period,
        output_column=output_column,
        **kwargs,
    )


def compute_talib_ema(
    df: pd.DataFrame,
    period: int = 20,
    series: Optional[pd.Series] = None,
    output_column: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    if series is None:
        series = df["close"]
    if output_column is None:
        output_column = f"ema_{period}"

    return compute_talib_indicator(
        df,
        "EMA",
        real=series,
        timeperiod=period,
        output_column=output_column,
        **kwargs,
    )


def compute_talib_rsi(
    df: pd.DataFrame,
    period: int = 14,
    series: Optional[pd.Series] = None,
    output_column: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    if series is None:
        series = df["close"]
    if output_column is None:
        output_column = f"rsi_{period}"

    return compute_talib_indicator(
        df,
        "RSI",
        real=series,
        timeperiod=period,
        output_column=output_column,
        **kwargs,
    )


def compute_talib_macd(
    df: pd.DataFrame,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
    series: Optional[pd.Series] = None,
    output_columns: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    if series is None:
        series = df["close"]
    if output_columns is None:
        output_columns = ["macd", "macd_signal", "macd_histogram"]

    return compute_talib_indicator(
        df,
        "MACD",
        real=series,
        fastperiod=fastperiod,
        slowperiod=slowperiod,
        signalperiod=signalperiod,
        output_columns=output_columns,
        **kwargs,
    )
