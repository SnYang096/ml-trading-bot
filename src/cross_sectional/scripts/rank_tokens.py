#!/usr/bin/env python3
"""
Rank tokens cross-sectionally on a given date using a factor column from FeatureStore.

This is designed to support workflows like:
  - precompute features monthly in FeatureStore (per symbol / per timeframe)
  - query a specific day: rank all tokens by a factor (e.g., momentum, volatility, VPIN score)

Example:
  python3 src/cross_sectional/scripts/rank_tokens.py \
    --date 2025-10-10 \
    --factor rsi_f \
    --universe-config config/download/crypto_4h_token_universe_groups.yaml \
    --universe-set starter_a \
    --features-store-root feature_store \
    --features-store-layer features_83f12ecc5e \
    --timeframe 240T \
    --top 50
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_tools.universe_config import load_universe_config  # noqa: E402


@dataclass(frozen=True)
class RankRow:
    symbol: str
    value: float
    asof_ts: pd.Timestamp


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rank tokens cross-sectionally on a given date using FeatureStore."
    )
    p.add_argument("--date", required=True, help="Date (YYYY-MM-DD)")
    p.add_argument(
        "--factor", required=True, help="Factor/feature column name to rank by"
    )
    p.add_argument(
        "--factor-set-yaml",
        default=None,
        help="Optional YAML containing factor_sets. If set, validate that --factor belongs to the chosen set.",
    )
    p.add_argument(
        "--factor-set",
        default=None,
        help="Factor set name inside --factor-set-yaml.",
    )

    p.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT). If set, overrides universe config.",
    )
    p.add_argument(
        "--universe-config",
        default=None,
        help="Universe config YAML (e.g., config/download/crypto_4h_token_universe_groups.yaml).",
    )
    p.add_argument("--universe-set", default="starter_a", help="Universe set name")
    p.add_argument(
        "--universe-groups",
        default=None,
        help="Comma-separated groups to include (e.g., highcap,alt). Default: all groups.",
    )

    p.add_argument(
        "--features-store-root", default="feature_store", help="FeatureStore root dir"
    )
    p.add_argument(
        "--features-store-layer",
        required=True,
        help="FeatureStore layer (e.g., features_83f12ecc5e)",
    )
    p.add_argument("--timeframe", default="240T", help="Timeframe (e.g., 240T for 4H)")
    p.add_argument(
        "--bar",
        choices=["last", "first"],
        default="last",
        help="Which bar within the day to use (default: last).",
    )
    p.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending (default: descending, highest value ranks first).",
    )
    p.add_argument("--top", type=int, default=50, help="Top-N rows to print")
    p.add_argument(
        "--output",
        default=None,
        help="Optional output path (.csv or .json). If omitted, prints to stdout only.",
    )
    return p.parse_args()


def _validate_factor_in_set(
    factor: str, *, factor_set_yaml: str, factor_set: str
) -> None:
    import yaml

    obj = yaml.safe_load(Path(factor_set_yaml).read_text(encoding="utf-8")) or {}
    sets = (obj.get("factor_sets") or {}) if isinstance(obj, dict) else {}
    if factor_set not in sets:
        raise KeyError(f"factor_set '{factor_set}' not found in {factor_set_yaml}")
    candidates = [
        str(x).strip() for x in (sets.get(factor_set) or []) if str(x).strip()
    ]
    if not candidates:
        raise ValueError(f"factor_set '{factor_set}' is empty in {factor_set_yaml}")

    # tolerate _f suffix mismatch
    f = str(factor).strip()
    ok = (
        (f in candidates)
        or (f.endswith("_f") and f[:-2] in candidates)
        or ((f + "_f") in candidates)
    )
    if not ok:
        raise ValueError(
            f"Factor '{factor}' not in factor_set '{factor_set}'. "
            f"Example factors: {candidates[:10]}"
        )


def _resolve_symbols(args: argparse.Namespace) -> List[str]:
    if args.symbols:
        return [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    if args.universe_config:
        groups = (
            [g.strip() for g in str(args.universe_groups).split(",") if g.strip()]
            if args.universe_groups
            else None
        )
        cfg = load_universe_config(args.universe_config)
        return cfg.resolve_symbols_usdt(universe_set=args.universe_set, groups=groups)
    raise ValueError("Must provide either --symbols or --universe-config.")


def _month_key(ts: pd.Timestamp) -> str:
    return f"{ts.year:04d}-{ts.month:02d}"


def _feature_store_month_path(
    *,
    root: Path,
    layer: str,
    symbol: str,
    timeframe: str,
    month_key: str,
) -> Path:
    return root / layer / symbol / timeframe / f"{month_key}.parquet"


def _infer_day_target_timestamp(
    day_start: pd.Timestamp, timeframe: str, which: str
) -> Optional[pd.Timestamp]:
    """
    For fixed-minute timeframe like '240T', infer the first/last bar timestamp within the day.
    Returns None if cannot infer safely.
    """
    try:
        offset = pd.tseries.frequencies.to_offset(timeframe)
        seconds = getattr(offset, "delta", None)
        if seconds is None:
            return None
        step_s = int(seconds.total_seconds())
        if step_s <= 0 or 86400 % step_s != 0:
            return None
        n = 86400 // step_s
        if which == "first":
            return day_start
        return day_start + pd.Timedelta(seconds=step_s * (n - 1))
    except Exception:
        return None


def _pick_asof_value_for_day(
    df: pd.DataFrame,
    *,
    factor: str,
    day_start: pd.Timestamp,
    day_end: pd.Timestamp,
    target_ts: pd.Timestamp,
) -> Optional[RankRow]:
    # Ensure datetime index
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "Expected DatetimeIndex or a 'timestamp' column in FeatureStore frame."
        )

    # Filter day window
    idx = df.index
    mask = (idx >= day_start) & (idx < day_end)
    if not mask.any():
        return None
    df_day = df.loc[mask]

    use_factor = factor
    if use_factor not in df_day.columns:
        # Common convention mismatch: FeatureStore columns may be saved without "_f" suffix
        if use_factor.endswith("_f") and use_factor[:-2] in df_day.columns:
            use_factor = use_factor[:-2]
        elif (use_factor + "_f") in df_day.columns:
            use_factor = use_factor + "_f"
        else:
            # Give a small hint for debugging
            sample = [c for c in df_day.columns[:50]]
            raise KeyError(
                f"Factor column not found: {factor}. " f"Sample columns: {sample}"
            )

    # Prefer exact target_ts, otherwise asof <= target_ts inside day
    if target_ts in df_day.index:
        asof_ts = target_ts
        value = df_day.loc[asof_ts, use_factor]
    else:
        df_day_le = df_day.loc[df_day.index <= target_ts]
        if df_day_le.empty:
            # fallback: first available in day
            asof_ts = df_day.index.min()
            value = df_day.loc[asof_ts, use_factor]
        else:
            asof_ts = df_day_le.index.max()
            value = df_day_le.loc[asof_ts, use_factor]

    if pd.isna(value):
        return None
    # symbol is passed separately by caller; keep dataclass for completeness
    return RankRow(
        symbol=str(df_day.iloc[0].get("symbol", "")),
        value=float(value),
        asof_ts=asof_ts,
    )


def rank_tokens_for_date(
    *,
    date_str: str,
    factor: str,
    symbols: List[str],
    features_store_root: str,
    features_store_layer: str,
    timeframe: str,
    which_bar: str,
    ascending: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    day = pd.Timestamp(date_str)
    day_start = day.normalize()
    day_end = day_start + pd.Timedelta(days=1)

    inferred_target = _infer_day_target_timestamp(day_start, timeframe, which_bar)
    # If can't infer (odd timeframe), use last timestamp available per symbol within day
    target_ts = inferred_target or (day_end - pd.Timedelta(microseconds=1))

    root = Path(features_store_root)
    month_key = _month_key(day_start)

    rows: List[dict] = []
    missing_symbols: List[str] = []

    for sym in symbols:
        p = _feature_store_month_path(
            root=root,
            layer=features_store_layer,
            symbol=sym,
            timeframe=timeframe,
            month_key=month_key,
        )
        if not p.exists():
            missing_symbols.append(sym)
            continue

        df = pd.read_parquet(p)
        # FeatureStore parquet uses timestamp as index
        if isinstance(df.index, pd.DatetimeIndex) and df.index.name == "timestamp":
            df = df.copy()
            df["timestamp"] = df.index
        if "symbol" not in df.columns:
            df["symbol"] = sym

        r = _pick_asof_value_for_day(
            df,
            factor=factor,
            day_start=day_start,
            day_end=day_end,
            target_ts=target_ts,
        )
        if r is None:
            continue
        rows.append({"symbol": sym, "value": r.value, "asof_ts": r.asof_ts})

    out = pd.DataFrame(rows)
    if out.empty:
        return out, missing_symbols

    out = out.sort_values("value", ascending=bool(ascending)).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out[["rank", "symbol", "value", "asof_ts"]], missing_symbols


def _print_table(df: pd.DataFrame, top: int) -> None:
    if df.empty:
        print("No ranking rows produced.")
        return
    view = df.head(int(top)) if top and top > 0 else df
    # Pretty print without requiring tabulate
    print(view.to_string(index=False))


def _write_output(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".json":
        p.write_text(df.to_json(orient="records", date_format="iso"), encoding="utf-8")
        return
    # default csv
    df.to_csv(p, index=False)


def main() -> None:
    args = _parse_args()
    if args.factor_set_yaml or args.factor_set:
        if not args.factor_set_yaml or not args.factor_set:
            raise ValueError("Must provide both --factor-set-yaml and --factor-set")
        _validate_factor_in_set(
            str(args.factor),
            factor_set_yaml=str(args.factor_set_yaml),
            factor_set=str(args.factor_set),
        )
    symbols = _resolve_symbols(args)

    df, missing = rank_tokens_for_date(
        date_str=args.date,
        factor=args.factor,
        symbols=symbols,
        features_store_root=args.features_store_root,
        features_store_layer=args.features_store_layer,
        timeframe=args.timeframe,
        which_bar=args.bar,
        ascending=args.ascending,
    )

    if missing:
        print(
            f"⚠️  Missing FeatureStore month parquet for {len(missing)} symbol(s): "
            f"{', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}"
        )

    _print_table(df, top=args.top)

    if args.output:
        _write_output(df, args.output)
        print(f"✅ Wrote: {args.output}")


if __name__ == "__main__":
    main()
