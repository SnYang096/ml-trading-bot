from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data_tools.data_loader import MarketDataLoader
from data_tools.comprehensive_feature_engineering import ComprehensiveFeatureEngineer
from data_tools.baseline_features import BaselineFeatureEngineer
from data_tools.rolling_data import (
    load_and_process_file,
    add_order_flow_features,
)


@dataclass
class PanelGenerationConfig:
    """
    Configuration for building a cross-sectional feature panel from raw OHLCV data.

    Attributes:
        symbols: Iterable of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"]).
        timeframe: Resample frequency (pandas offset alias, e.g., "15T").
        horizon: Forward return horizon in bars (creates `future_return_{horizon}`).
        data_path: Optional root directory for MarketDataLoader (if None, loader auto-detects).
        start_date: Optional start date filter (YYYY-MM-DD).
        end_date: Optional end date filter (YYYY-MM-DD).
        feature_type: One of {"baseline", "comprehensive"}.
        dropna: Drop rows with NaNs after engineering & target creation.
        save_path: Optional file path to save panel parquet.
        engine_kwargs: Extra kwargs forwarded to feature engineer constructors.
    """

    symbols: Sequence[str]
    timeframe: str = "15T"
    horizon: int = 12
    data_path: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    feature_type: str = "baseline"
    dropna: bool = True
    save_path: Optional[str] = None
    engine_kwargs: dict = field(default_factory=dict)
    include_order_flow: bool = True

    def validate(self) -> None:
        if not self.symbols:
            raise ValueError("At least one symbol must be provided.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.feature_type not in {"baseline", "comprehensive"}:
            raise ValueError("feature_type must be 'baseline' or 'comprehensive'.")


def generate_cross_sectional_panel(config: PanelGenerationConfig) -> Tuple[pd.DataFrame, str]:
    """
    Build a cross-sectional feature panel suitable for downstream modelling.

    Returns:
        panel: MultiIndex DataFrame indexed by (timestamp, symbol).
        target_col: Name of the generated forward return column.
    """
    config.validate()

    frames: List[pd.DataFrame] = []
    target_col = f"future_return_{config.horizon}"
    data_dir = Path(config.data_path or "data/parquet_data")
    start_ts = _to_utc_timestamp(config.start_date)
    end_ts = _to_utc_timestamp(config.end_date)

    for symbol in config.symbols:
        print(f"📊 Generating features for {symbol}...")
        df_resampled = _load_symbol_data(
            symbol=symbol,
            data_dir=data_dir,
            timeframe=config.timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
            include_order_flow=config.include_order_flow,
            loader_path=config.data_path,
            loader_start=config.start_date,
            loader_end=config.end_date,
        )

        if df_resampled is None or df_resampled.empty:
            print(f"   ⚠️  Skipping {symbol}: no data available.")
            continue

        df_resampled = df_resampled.sort_index()
        df_resampled["timestamp"] = df_resampled.index
        df_resampled["symbol"] = symbol

        if config.feature_type == "baseline":
            engineer = BaselineFeatureEngineer(**config.engine_kwargs)
            features = engineer.engineer_features(df_resampled, fit=True)
        else:
            engineer = ComprehensiveFeatureEngineer(**config.engine_kwargs)
            features = engineer.engineer_all_features(df_resampled, fit=True)

        if "timestamp" not in features.columns:
            features["timestamp"] = df_resampled["timestamp"].values
        features["symbol"] = symbol

        features = features.set_index(pd.to_datetime(features["timestamp"], utc=True))
        features.index.name = "timestamp"
        if "timestamp" in features.columns:
            features = features.drop(columns=["timestamp"])
        if target_col not in features.columns:
            if "close" not in features.columns:
                raise ValueError(f"{symbol}: 'close' column missing; cannot compute forward return.")
            features[target_col] = features["close"].shift(-config.horizon) / features["close"] - 1.0

        if config.dropna:
            features = features.dropna(subset=[target_col])

        frames.append(features)
        print(f"   ✅ {symbol}: {len(features):,} rows")

    if not frames:
        raise RuntimeError("No features were generated for any symbol.")

    panel = pd.concat(frames, axis=0)
    panel = panel.reset_index().set_index(["timestamp", "symbol"]).sort_index()

    if config.dropna:
        panel = panel.dropna(axis=0, how="any")

    if config.save_path:
        path = Path(config.save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(path)
        print(f"💾 Saved panel to {path}")

    return panel, target_col


def _load_symbol_data(
    symbol: str,
    data_dir: Path,
    timeframe: str,
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
    include_order_flow: bool,
    loader_path: Optional[str],
    loader_start: Optional[str],
    loader_end: Optional[str],
) -> Optional[pd.DataFrame]:
    # First try raw files (zip/parquet) in data_dir
    df = _load_symbol_from_files(
        symbol=symbol,
        data_dir=data_dir,
        timeframe=timeframe,
        start_ts=start_ts,
        end_ts=end_ts,
        include_order_flow=include_order_flow,
    )
    if df is not None and not df.empty:
        return df

    # Fallback to MarketDataLoader (expects pre-aggregated OHLCV directory or single file)
    try:
        loader = MarketDataLoader(data_path=loader_path)
        df_raw = loader.load_data(symbol=symbol, start_date=loader_start, end_date=loader_end)
        if df_raw is None or df_raw.empty:
            return None
        df_resampled = loader.resample_data(timeframe)
        df_resampled.index = pd.to_datetime(df_resampled.index, utc=True)
        if start_ts:
            df_resampled = df_resampled[df_resampled.index >= start_ts]
        if end_ts:
            df_resampled = df_resampled[df_resampled.index <= end_ts]
        return df_resampled
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  Loader fallback failed for {symbol}: {exc}")
        return None


def _load_symbol_from_files(
    symbol: str,
    data_dir: Path,
    timeframe: str,
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
    include_order_flow: bool,
) -> Optional[pd.DataFrame]:
    if not data_dir or not data_dir.exists():
        return None

    files = _collect_symbol_files(symbol, data_dir, start_ts=start_ts, end_ts=end_ts)
    if not files:
        return None

    frames: List[pd.DataFrame] = []
    for file_path in files:
        df_file = load_and_process_file(str(file_path), freq=timeframe)
        if df_file is None or df_file.empty:
            continue
        if include_order_flow and file_path.suffix.lower() == ".zip":
            df_file = add_order_flow_features(str(file_path), df_file)
        frames.append(df_file)

    if not frames:
        return None

    combined = pd.concat(frames, axis=0).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.index = pd.to_datetime(combined.index, utc=True, errors="coerce")
    combined = combined[combined.index.notna()]
    if start_ts:
        combined = combined[combined.index >= start_ts]
    if end_ts:
        combined = combined[combined.index <= end_ts]
    return combined


def _collect_symbol_files(
    symbol: str,
    data_dir: Path,
    *,
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
) -> List[Path]:
    normalized = symbol.upper().replace("-", "").replace("/", "")
    if not normalized.endswith("USDT"):
        normalized = f"{normalized}USDT"
    prefixes = {
        normalized,
        normalized.replace("USDT", "-USD"),
        normalized.replace("USDT", "_USD"),
        normalized.replace("USDT", ""),
    }

    start_period = start_ts.to_period("M") if start_ts is not None else None
    end_period = end_ts.to_period("M") if end_ts is not None else None

    files: List[Path] = []
    for ext in (".parquet", ".zip"):
        for path in data_dir.rglob(f"*{ext}"):
            name = path.name.upper()
            if any(name.startswith(prefix) for prefix in prefixes):
                file_period = _extract_file_period(name)
                if start_period and file_period and file_period < start_period:
                    continue
                if end_period and file_period and file_period > end_period:
                    continue
                files.append(path)

    files.sort()
    return files


def _extract_file_period(name: str) -> Optional[pd.Period]:
    match = None
    for pattern in [
        r"(20\d{2})[-_](\d{2})",
        r"(20\d{2})(\d{2})",
    ]:
        m = re.search(pattern, name)
        if m:
            match = m
            break
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2))
    try:
        return pd.Period(year=year, month=month, freq="M")
    except Exception:
        return None


def _to_utc_timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    ts = pd.to_datetime(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts

