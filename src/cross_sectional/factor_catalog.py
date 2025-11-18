from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

DEFAULT_EXCLUDE_PREFIXES = (
    "future_return",
    "signal",
    "binary_signal",
)

DEFAULT_EXCLUDE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "timestamp",
    "symbol",
    "has_trade",
}


@dataclass(frozen=True)
class FactorCategory:
    name: str
    patterns: Iterable[str]


CATEGORY_DEFINITIONS: List[FactorCategory] = [
    FactorCategory(
        "baseline_structure",
        (
            "sr_dist_",
            "channel_",
            "compression_",
            "atr_percentile",
            "atr_compression_ratio",
            "price_entropy",
            "internal_price_density",
        ),
    ),
    FactorCategory(
        "volatility",
        (
            "_volatility",
            "volatility",
            "atr_",
            "natr",
            "std",
            "var",
            "bb_",
            "_hv",
        ),
    ),
    FactorCategory(
        "momentum_returns",
        (
            "_return",
            "momentum",
            "mom_",
            "roc",
            "rsi",
            "stoch",
            "macd",
            "cci",
            "willr",
            "ultosc",
        ),
    ),
    FactorCategory(
        "orderflow",
        (
            "cvd",
            "taker_buy",
            "buy_qty",
            "sell_qty",
            "orderflow",
        ),
    ),
    FactorCategory("hurst", ("_hurst",)),
    FactorCategory("wavelet", ("wpt_",)),
    FactorCategory("hilbert", ("hilbert_",)),
    FactorCategory("spectral", ("spectral_", "spectrum_", "freq_")),
    FactorCategory("dl_sequence", ("dl_seq", "mamba", "transformer", "ts_encoder")),
    FactorCategory("crypto_cross", ("cs_crypto_",)),
    FactorCategory("normalization", ("normalized", "zscore", "rank_")),
]


def categorize_columns(
    columns: Iterable[str],
    *,
    exclude_columns: Iterable[str] | None = None,
    exclude_prefixes: Iterable[str] | None = None,
) -> Dict[str, List[str]]:
    """
    Categorise factor columns into pre-defined buckets based on name patterns.
    """
    exclude_cols = set(DEFAULT_EXCLUDE_COLUMNS)
    if exclude_columns:
        exclude_cols.update(exclude_columns)

    exclude_pref = tuple(DEFAULT_EXCLUDE_PREFIXES)
    if exclude_prefixes:
        exclude_pref = tuple(set(DEFAULT_EXCLUDE_PREFIXES).union(set(exclude_prefixes)))

    buckets: Dict[str, List[str]] = defaultdict(list)
    buckets_order = [cat.name for cat in CATEGORY_DEFINITIONS] + ["other"]

    for col in columns:
        if col in exclude_cols:
            continue
        if col.startswith(exclude_pref):
            continue

        matched = False
        for cat in CATEGORY_DEFINITIONS:
            if any(pattern in col for pattern in cat.patterns):
                buckets[cat.name].append(col)
                matched = True
        if not matched:
            buckets["other"].append(col)

    # Sort lists for stability and ensure all categories exist
    ordered_result: Dict[str, List[str]] = {}
    for name in buckets_order:
        if buckets.get(name):
            ordered_result[name] = sorted(set(buckets[name]))
    return ordered_result


def format_summary(categories: Mapping[str, Iterable[str]]) -> str:
    """
    Create a human-readable summary of category sizes.
    """
    lines = ["Factor category summary:"]
    for name, cols in categories.items():
        size = len(list(cols))
        lines.append(f"  - {name}: {size}")
    return "\n".join(lines)
