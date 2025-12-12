import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import iqr


EPS = 1e-9


@dataclass
class FootprintConfig:
    """Configuration for footprint feature calculation."""

    price_bin_size: Optional[float] = None
    price_bin_method: str = "fd"  # "fd" (Freedman–Diaconis) or "fixed_bins"
    price_bin_target_bins: int = 40  # only used when price_bin_method == "fixed_bins"
    value_area_pct: float = 0.7
    tick_size: Optional[float] = None  # optional per-symbol tick size


def _auto_price_bin_size(prices: pd.Series, cfg: FootprintConfig) -> float:
    """Determine a price bin size that adapts to the symbol and the local volatility.

    Priority:
    1) Explicit tick_size (per-symbol) if provided.
    2) Explicit price_bin_size if provided.
    3) Freedman–Diaconis rule on the price samples inside the bar.
    4) Fallback to an equally spaced bin width targeting price_bin_target_bins.
    """
    if cfg.tick_size and cfg.tick_size > 0:
        return float(cfg.tick_size)
    if cfg.price_bin_size and cfg.price_bin_size > 0:
        return float(cfg.price_bin_size)

    if cfg.price_bin_method == "fd":
        iqr_val = iqr(prices)
        if iqr_val <= 0 or prices.count() < 2:
            # low variation or too few points; fall back to target_bins
            price_range = prices.max() - prices.min()
            return float(price_range / max(cfg.price_bin_target_bins, 1)) if price_range > 0 else 1.0
        bin_width = 2 * iqr_val / (prices.count() ** (1 / 3))
        return float(bin_width) if bin_width > 0 else 1.0

    # fixed_bins mode
    price_range = prices.max() - prices.min()
    return float(price_range / max(cfg.price_bin_target_bins, 1)) if price_range > 0 else 1.0


def _build_bins(prices: pd.Series, bin_width: float) -> np.ndarray:
    """Build closed-open bin edges that cover the full price range."""
    p_min = prices.min()
    p_max = prices.max()
    # ensure at least one bin even if p_min == p_max
    if math.isclose(p_min, p_max):
        p_min -= bin_width
        p_max += bin_width
    start_edge = math.floor(p_min / bin_width) * bin_width
    end_edge = math.ceil(p_max / bin_width) * bin_width + bin_width
    return np.arange(start_edge, end_edge + bin_width, bin_width)


def _value_area_bounds(volume_by_bin: pd.Series, value_area_pct: float, bin_edges: np.ndarray) -> Tuple[float, float]:
    """Compute VAH/VAL that cover `value_area_pct` of volume."""
    if volume_by_bin.empty or volume_by_bin.sum() <= 0:
        return np.nan, np.nan
    sorted_bins = volume_by_bin.sort_values(ascending=False)
    cum = sorted_bins.cumsum() / sorted_bins.sum()
    selected_bins = sorted_bins.index[cum <= value_area_pct]
    # ensure at least the top bin is included
    if selected_bins.empty:
        selected_bins = sorted_bins.index[:1]
    bin_idx_min = min(selected_bins)
    bin_idx_max = max(selected_bins)
    return bin_edges[bin_idx_min], bin_edges[bin_idx_max + 1]


def compute_kline_footprint_features(
    ticks: pd.DataFrame,
    klines: pd.DataFrame,
    open_col: str = "open_time",
    close_col: str = "close_time",
    cfg: Optional[FootprintConfig] = None,
) -> pd.DataFrame:
    """Compute single-bar (e.g., 1H/4H) footprint features: POC/HVN/LVN/VAH/VAL and delta/imbalance.

    Args:
        ticks: DataFrame with columns ['price', 'volume', 'side'] (side ∈ {1, -1}) and a DateTimeIndex.
        klines: DataFrame with at least open/close time columns; index will be used for aligning outputs.
        open_col: column name for kline start timestamp (inclusive).
        close_col: column name for kline end timestamp (exclusive).
        cfg: FootprintConfig; if None, defaults will be used.

    Returns:
        DataFrame indexed as `klines.index` with columns:
            fp_poc, fp_hvn, fp_lvn, fp_vah, fp_val,
            fp_delta_poc, fp_max_imbalance_price, fp_max_imbalance_ratio,
            fp_volume_skew, fp_delta_skew,
            fp_exhaustion_price, fp_exhaustion_zscore,
            fp_delta_divergence

    Notes:
        - Bins are adaptive per bar using Freedman–Diaconis unless an explicit tick_size or price_bin_size is supplied.
        - Bars with no ticks return NaN for all footprint columns.
    """
    cfg = cfg or FootprintConfig()
    required_cols = {"price", "volume", "side"}
    missing = required_cols - set(ticks.columns)
    if missing:
        raise ValueError(f"ticks missing required columns: {missing}")

    result_rows = []
    tick_index = ticks.index

    for _, row in klines.iterrows():
        start_ts = row[open_col]
        end_ts = row[close_col]
        # select ticks in [start_ts, end_ts)
        mask = (tick_index >= start_ts) & (tick_index < end_ts)
        bar_ticks = ticks.loc[mask]
        if bar_ticks.empty:
            result_rows.append(
                {
                    "fp_poc": np.nan,
                    "fp_hvn": np.nan,
                    "fp_lvn": np.nan,
                    "fp_vah": np.nan,
                    "fp_val": np.nan,
                    "fp_delta_poc": np.nan,
                    "fp_max_imbalance_price": np.nan,
                    "fp_max_imbalance_ratio": np.nan,
                    "fp_volume_skew": np.nan,
                    "fp_delta_skew": np.nan,
                    "fp_exhaustion_price": np.nan,
                    "fp_exhaustion_zscore": np.nan,
                    "fp_delta_divergence": np.nan,
                }
            )
            continue

        bin_width = _auto_price_bin_size(bar_ticks["price"], cfg)
        bin_edges = _build_bins(bar_ticks["price"], bin_width)
        # bin_id starts at 0
        bin_id = np.digitize(bar_ticks["price"], bin_edges, right=False) - 1
        bar_ticks = bar_ticks.assign(_bin=bin_id)

        buy_vol = bar_ticks.loc[bar_ticks["side"] > 0].groupby("_bin")["volume"].sum()
        sell_vol = bar_ticks.loc[bar_ticks["side"] < 0].groupby("_bin")["volume"].sum()
        all_bins = pd.Index(np.arange(len(bin_edges) - 1))
        buy_vol = buy_vol.reindex(all_bins, fill_value=0.0)
        sell_vol = sell_vol.reindex(all_bins, fill_value=0.0)
        total_vol = buy_vol + sell_vol
        delta_vol = buy_vol - sell_vol

        # core levels
        if total_vol.sum() <= 0:
            result_rows.append(
                {
                    "fp_poc": np.nan,
                    "fp_hvn": np.nan,
                    "fp_lvn": np.nan,
                    "fp_vah": np.nan,
                    "fp_val": np.nan,
                    "fp_delta_poc": np.nan,
                    "fp_max_imbalance_price": np.nan,
                    "fp_max_imbalance_ratio": np.nan,
                    "fp_volume_skew": np.nan,
                    "fp_delta_skew": np.nan,
                    "fp_exhaustion_price": np.nan,
                    "fp_exhaustion_zscore": np.nan,
                    "fp_delta_divergence": np.nan,
                }
            )
            continue

        poc_bin = int(total_vol.idxmax())
        hvn_bin = poc_bin  # align with POC as the highest volume node
        # LVN: smallest positive volume bin
        positive_bins = total_vol[total_vol > 0]
        lvn_bin = int(positive_bins.idxmin()) if not positive_bins.empty else poc_bin

        poc_price = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2
        hvn_price = (bin_edges[hvn_bin] + bin_edges[hvn_bin + 1]) / 2
        lvn_price = (bin_edges[lvn_bin] + bin_edges[lvn_bin + 1]) / 2

        vah_price, val_price = _value_area_bounds(total_vol, cfg.value_area_pct, bin_edges)

        delta_poc = delta_vol.loc[poc_bin]
        imbalance_ratio = (np.maximum(buy_vol, sell_vol) + EPS) / (np.minimum(buy_vol, sell_vol) + EPS)
        max_imbalance_bin = int(imbalance_ratio.idxmax())
        max_imbalance_price = (bin_edges[max_imbalance_bin] + bin_edges[max_imbalance_bin + 1]) / 2
        max_imbalance_value = float(imbalance_ratio.loc[max_imbalance_bin])

        # Exhaustion spike: largest absolute delta z-score within the bar
        abs_delta = delta_vol.abs()
        abs_mean = abs_delta.mean()
        abs_std = abs_delta.std(ddof=0)
        if abs_std > 0:
            zscores = (abs_delta - abs_mean) / abs_std
            exhaustion_bin = int(zscores.idxmax())
            exhaustion_z = float(zscores.loc[exhaustion_bin])
            exhaustion_price = (bin_edges[exhaustion_bin] + bin_edges[exhaustion_bin + 1]) / 2
        else:
            exhaustion_bin = poc_bin
            exhaustion_z = 0.0
            exhaustion_price = poc_price

        # Delta divergence: price change vs delta_poc sign (1 = divergence, 0 = aligned, NaN if no price info)
        if {"open", "close"}.issubset(klines.columns):
            price_change = row.get("close", np.nan) - row.get("open", np.nan)
            if pd.isna(price_change) or price_change == 0 or delta_poc == 0:
                delta_divergence = 0.0
            else:
                delta_divergence = float(np.sign(price_change) != np.sign(delta_poc))
        else:
            delta_divergence = np.nan

        result_rows.append(
            {
                "fp_poc": poc_price,
                "fp_hvn": hvn_price,
                "fp_lvn": lvn_price,
                "fp_vah": vah_price,
                "fp_val": val_price,
                "fp_delta_poc": float(delta_poc),
                "fp_max_imbalance_price": max_imbalance_price,
                "fp_max_imbalance_ratio": max_imbalance_value,
                "fp_volume_skew": float(total_vol.replace(0, np.nan).skew()),
                "fp_delta_skew": float(delta_vol.replace(0, np.nan).skew()),
                "fp_exhaustion_price": exhaustion_price,
                "fp_exhaustion_zscore": exhaustion_z,
                "fp_delta_divergence": delta_divergence,
            }
        )

    return pd.DataFrame(result_rows, index=klines.index)

