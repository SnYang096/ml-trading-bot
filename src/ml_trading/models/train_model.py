"""CLI for training production-ready LightGBM models with comprehensive features."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ml_trading.data_tools.comprehensive_feature_engineering import (
    ComprehensiveFeatureEngineer, )
from ml_trading.data_tools.data_loader import MarketDataLoader
from ml_trading.strategies.ml_strategy import MLTradingStrategy
from ml_trading.pipeline.dimensionality.integration import DimensionalityIntegrationEngine  # type: ignore
from ml_trading.pipeline.dimensionality.utils import (
    load_top_factors_list,
    filter_engineered_by_topk,
)
from ml_trading.models.autoencoder import UnifiedAutoencoder  # type: ignore

# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------


class MultiTimeframeComprehensiveEngineer:
    """Wrapper that applies ComprehensiveFeatureEngineer per timeframe."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.engineers: Dict[str, ComprehensiveFeatureEngineer] = {}

    def engineer_features(self,
                          multi_tf_data: Dict[str, pd.DataFrame],
                          fit: bool = True) -> Dict[str, pd.DataFrame]:
        engineered: Dict[str, pd.DataFrame] = {}
        for timeframe, df in multi_tf_data.items():
            engineer = self.engineers.get(timeframe)
            if engineer is None or fit:
                engineer = ComprehensiveFeatureEngineer(**self.kwargs)
                self.engineers[timeframe] = engineer
            engineered[timeframe] = engineer.engineer_all_features(df, fit=fit)
        return engineered

    def save_scalers(self, path: str) -> None:
        with open(path, "wb") as handle:
            pickle.dump(self.engineers, handle)

    def load_scalers(self, path: str) -> None:
        with open(path, "rb") as handle:
            self.engineers = pickle.load(handle)


@dataclass
class SymbolTrainingConfig:
    symbol: str
    data_paths: List[str]
    start_ts: Optional[pd.Timestamp]
    end_ts: Optional[pd.Timestamp]


@dataclass
class SymbolTrainingResult:
    symbol: str
    model_path: Path
    scaler_path: Path
    info_path: Path
    start: pd.Timestamp
    end: pd.Timestamp
    metrics: Dict[str, Dict[str, Dict[str, float]]]
    total_bars: int


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=
        "Train multi-timeframe LightGBM strategy with comprehensive features",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=os.environ.get("SYMBOL", "BTCUSDT"),
        help="Default trading symbol (used when --symbols not provided)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="One or more trading symbols to train (e.g. BTCUSDT ETHUSDT)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=os.environ.get("START_DATE"),
        help=
        "Start date (YYYY-MM-DD). If omitted, use earliest available data.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=os.environ.get("END_DATE"),
        help="End date (YYYY-MM-DD). If omitted, use latest available data.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("DATA_DIR", "data/parquet_data"),
        help="Directory containing aggregated trade parquet/zip files",
    )
    parser.add_argument(
        "--train-data",
        nargs="+",
        default=None,
        help=
        ("Explicit data files to use (parquet/csv/zip). "
         "For multiple symbols use SYMBOL:path format, e.g. BTCUSDT:/data/a.parquet"
         ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("MODEL_DIR", "models"),
        help="Directory to store trained model artifacts",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("MODEL_NAME", "trained_model"),
        help=("Base name for saved model/scaler/info files. "
              "Supports {symbol} and {period} placeholders."),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing model/scaler files if they already exist",
    )
    parser.add_argument(
        "--forward-bars",
        type=int,
        default=1,
        help=
        "Number of bars ahead for label prediction (default: 1). Use 1, 5, 10, or 15 for different horizons.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable additional logging output",
    )
    # Dimensionality options
    parser.add_argument(
        "--use-top-factors",
        type=str,
        default=None,
        help=(
            "Path to top_factors JSON (from dimensionality-real). If provided, "
            "filters engineered features to this Top-K list per timeframe."),
    )
    parser.add_argument(
        "--use-autoencoder",
        type=str,
        default=None,
        help=
        ("Path to a trained autoencoder .pth (UnifiedAutoencoder). If provided, "
         "engineered features will be transformed to compressed embeddings before training."
         ),
    )
    parser.add_argument(
        "--encoding-dim",
        type=int,
        default=None,
        help=
        "Encoding dimension of the provided autoencoder (required with --use-autoencoder)",
    )
    return parser


def _parse_symbol_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    tokens = re.split(r"[\s,;]+", value.strip())
    result = [token for token in tokens if token]
    return result or None


def _parse_env_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    tokens = re.split(r"[\s,;]+", value.strip())
    entries = [token for token in tokens if token]
    return entries or None


def _parse_train_data_mapping(train_data_entries: Optional[Sequence[str]],
                              symbols: Sequence[str]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {sym: [] for sym in symbols}
    if not train_data_entries:
        return mapping

    use_prefixed = any(":" in entry for entry in train_data_entries)
    if use_prefixed:
        for entry in train_data_entries:
            if ":" not in entry:
                raise ValueError(
                    "When mixing prefixed and unprefixed --train-data entries, "
                    "ensure all entries follow SYMBOL:path format.")
            sym_key, path = entry.split(":", 1)
            sym_key = sym_key.strip().upper()
            mapping.setdefault(sym_key, [])
            mapping[sym_key].append(path.strip())
        return mapping

    if len(symbols) > 1:
        raise ValueError(
            "Provide symbol-prefixed entries (e.g. BTCUSDT:/path) when training multiple symbols."
        )

    mapping[symbols[0]].extend(entry.strip() for entry in train_data_entries)
    return mapping


def collect_data_paths(
    data_dir: str,
    symbol: str,
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
) -> List[str]:
    base_path = Path(data_dir)
    if not base_path.exists():
        return []

    start_period = start_ts.to_period("M") if start_ts is not None else None
    end_period = end_ts.to_period("M") if end_ts is not None else None

    # Symbol mapping: BTCUSDT -> BTC-USD, ETHUSDT -> ETH-USD, etc.
    symbol_mapping = {
        "BTCUSDT": "BTC-USD",
        "ETHUSDT": "ETH-USD",
        "BNBUSDT": "BNB-USD",
        "ADAUSDT": "ADA-USD",
        "SOLUSDT": "SOL-USD",
    }
    file_symbol = symbol_mapping.get(symbol, symbol)

    # Try multiple patterns
    patterns_to_try = [
        f"{symbol}-aggTrades-*",  # Original format: BTCUSDT-aggTrades-2024-10.parquet
        f"{file_symbol}_*",  # Alternative format: BTC-USD_2021-01.parquet
        f"{file_symbol}-*",  # Another alternative: BTC-USD-2021-01.parquet
    ]

    candidates: List[Path] = []
    for pattern in patterns_to_try:
        found = sorted(base_path.glob(pattern))
        candidates.extend(found)

    # Remove duplicates and sort
    candidates = sorted(set(candidates))

    data_paths: List[str] = []
    for candidate in candidates:
        if candidate.suffix.lower() not in {".parquet", ".zip", ".csv"}:
            continue

        # Try multiple regex patterns to extract date
        date_patterns = [
            # Pattern 1: BTCUSDT-aggTrades-2024-10
            rf"{re.escape(symbol)}-aggTrades-(?P<year>\d{{4}})-(?P<month>\d{{2}})",
            # Pattern 2: BTC-USD_2021-01
            rf"{re.escape(file_symbol)}_(?P<year>\d{{4}})-(?P<month>\d{{2}})",
            # Pattern 3: BTC-USD-2021-01
            rf"{re.escape(file_symbol)}-(?P<year>\d{{4}})-(?P<month>\d{{2}})",
        ]

        match = None
        for pattern in date_patterns:
            match = re.search(pattern, candidate.stem)
            if match:
                break

        if not match:
            continue

        try:
            year = int(match.group("year"))
            month = int(match.group("month"))
            file_period = pd.Period(year=year, month=month, freq="M")

            if start_period is not None and file_period < start_period:
                continue
            if end_period is not None and file_period > end_period:
                continue

            data_paths.append(str(candidate))
        except (ValueError, KeyError):
            continue

    return data_paths


def resolve_symbol_configs(
    args: argparse.Namespace,
) -> Tuple[List[SymbolTrainingConfig], Optional[pd.Timestamp],
           Optional[pd.Timestamp]]:
    env_symbols = _parse_symbol_list(os.environ.get("SYMBOLS"))
    symbols_input = args.symbols or env_symbols or [args.symbol]
    symbols = [sym.strip().upper() for sym in symbols_input if sym.strip()]
    if not symbols:
        raise ValueError("No symbols provided for training.")

    train_data_entries = args.train_data or _parse_env_list(
        os.environ.get("TRAIN_DATA"))
    train_data_map = _parse_train_data_mapping(train_data_entries, symbols)

    start_ts = pd.to_datetime(args.start_date) if args.start_date else None
    end_ts = pd.to_datetime(args.end_date) if args.end_date else None

    configs: List[SymbolTrainingConfig] = []
    for symbol in symbols:
        explicit_paths = [p for p in train_data_map.get(symbol, []) if p]
        if explicit_paths:
            data_paths = explicit_paths
        else:
            data_paths = collect_data_paths(args.data_dir, symbol, start_ts,
                                            end_ts)
        configs.append(
            SymbolTrainingConfig(
                symbol=symbol,
                data_paths=data_paths,
                start_ts=start_ts,
                end_ts=end_ts,
            ))

    return configs, start_ts, end_ts


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def load_trade_dataframe(
        data_path: str) -> Tuple[pd.DataFrame, Optional[Path]]:
    path = Path(data_path)
    suffix = path.suffix.lower()
    temp_dir: Optional[Path] = None

    if suffix == ".zip":
        temp_dir = path.parent / f"temp_extract_{path.stem}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        csv_files = list(temp_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV file found in the zip archive")
        df = pd.read_csv(csv_files[0])
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. Expected .zip, .csv, or .parquet"
        )

    return df, temp_dir


def prepare_trade_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "transact_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError("No timestamp column found in data")

    df.set_index("timestamp", inplace=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["price", "quantity"])
    return df


def prepare_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize pre-aggregated OHLCV datasets for downstream processing."""

    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "No timestamp column found in OHLCV data. Expected 'timestamp' or a datetime index."
        )

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("OHLCV dataset missing required columns: " +
                         ", ".join(sorted(missing)))

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=list(required))

    # Drop redundant timestamp column if parquet stored it both as index and column
    if "timestamp" in df.columns and df.index.equals(df["timestamp"]):
        df = df.drop(columns=["timestamp"])

    return df


def create_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    resampled = df.groupby(pd.Grouper(freq="1s")).agg({
        "price": ["first", "max", "min", "last"],
        "quantity":
        "sum"
    })
    resampled.columns = ["open", "high", "low", "close", "volume"]
    resampled = resampled.dropna().ffill()
    return resampled


def _format_period_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_label = start_ts.strftime("%Y%m%d")
    end_label = end_ts.strftime("%Y%m%d")
    if start_label == end_label:
        return start_label
    return f"{start_label}-{end_label}"


def add_order_flow_features(processed_trades: pd.DataFrame,
                            ohlcv_data: pd.DataFrame) -> pd.DataFrame:
    agg = processed_trades.copy()

    if "is_buyer_maker" in agg.columns:
        agg["is_buyer_maker"] = pd.to_numeric(agg["is_buyer_maker"],
                                              errors="coerce").fillna(0.5)
        agg["taker_buy"] = (
            ~agg["is_buyer_maker"].round().astype(bool)).astype(int)
    else:
        agg["taker_buy"] = 0

    agg["buy_qty"] = np.where(agg["taker_buy"] == 1, agg["quantity"], 0.0)
    agg["sell_qty"] = np.where(agg["taker_buy"] == 1, 0.0, agg["quantity"])

    per_sec = agg.groupby(pd.Grouper(freq="1s")).agg({
        "buy_qty": "sum",
        "sell_qty": "sum"
    })
    per_sec["taker_buy_ratio"] = per_sec["buy_qty"] / (
        per_sec["buy_qty"] + per_sec["sell_qty"]).replace(0, np.nan)
    per_sec["taker_buy_ratio"] = per_sec["taker_buy_ratio"].fillna(0.5)
    per_sec["cvd"] = (per_sec["buy_qty"] - per_sec["sell_qty"]).cumsum()

    enriched = (ohlcv_data.join(
        per_sec[["buy_qty", "sell_qty", "taker_buy_ratio", "cvd"]],
        how="left",
    ).ffill().fillna(0))
    return enriched


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------


def build_model_basename(base: str, symbol: str, period_label: str) -> str:
    base = base.strip() or "trained_model"
    symbol_token = symbol.lower()
    period_token = period_label

    if "{symbol}" in base or "{period}" in base:
        return base.format(symbol=symbol_token,
                           period=period_token).replace(" ", "_")

    name = base
    if symbol_token not in name.lower():
        name = f"{name}_{symbol_token}"
    if period_token not in name:
        name = f"{name}_{period_token}"
    return name.replace(" ", "_")


def train_symbol(
        args: argparse.Namespace,
        config: SymbolTrainingConfig) -> Optional[SymbolTrainingResult]:
    symbol = config.symbol
    print(f"\n=== Training {symbol} ===")

    if not config.data_paths:
        print(
            f"❌ No data files found for {symbol}. Provide --train-data or populate {args.data_dir}."
        )
        return None

    combined_frames: List[pd.DataFrame] = []
    temp_dirs: List[Path] = []

    try:
        for data_path in config.data_paths:
            if not os.path.exists(data_path):
                print(f"   ⚠️  Skipping missing file: {data_path}")
                continue
            raw_trades, temp_dir = load_trade_dataframe(data_path)
            combined_frames.append(raw_trades)
            if temp_dir is not None:
                temp_dirs.append(temp_dir)

        if not combined_frames:
            print(f"❌ Unable to load any data files for {symbol}.")
            return None

        raw_trades = pd.concat(combined_frames, ignore_index=True)

        has_trade_cols = {"price", "quantity"}.issubset(raw_trades.columns)
        has_ohlcv_cols = {"open", "high", "low", "close",
                          "volume"}.issubset(raw_trades.columns)

        processed_trades: Optional[pd.DataFrame]

        if has_trade_cols:
            processed_trades = prepare_trade_dataframe(raw_trades.copy())

            if config.start_ts is not None:
                processed_trades = processed_trades[processed_trades.index >=
                                                    config.start_ts]
            if config.end_ts is not None:
                processed_trades = processed_trades[processed_trades.index <=
                                                    config.end_ts]

            if processed_trades.empty:
                print(
                    f"❌ No trades remain after applying date filters for {symbol}."
                )
                return None

            actual_start = processed_trades.index.min()
            actual_end = processed_trades.index.max()

            ohlcv_data = create_ohlcv_data(processed_trades)
            ohlcv_data = add_order_flow_features(processed_trades, ohlcv_data)

        elif has_ohlcv_cols:
            ohlcv_data = prepare_ohlcv_dataframe(raw_trades.copy())

            if config.start_ts is not None:
                ohlcv_data = ohlcv_data[ohlcv_data.index >= config.start_ts]
            if config.end_ts is not None:
                ohlcv_data = ohlcv_data[ohlcv_data.index <= config.end_ts]

            if ohlcv_data.empty:
                print(
                    f"❌ No bars remain after applying date filters for {symbol}."
                )
                return None

            actual_start = ohlcv_data.index.min()
            actual_end = ohlcv_data.index.max()

            # Ensure optional order-flow columns exist for downstream processing
            if "buy_qty" not in ohlcv_data.columns:
                ohlcv_data["buy_qty"] = 0.0
            if "sell_qty" not in ohlcv_data.columns:
                ohlcv_data["sell_qty"] = 0.0
            if "taker_buy_ratio" not in ohlcv_data.columns:
                ohlcv_data["taker_buy_ratio"] = 0.5
            if "cvd" not in ohlcv_data.columns:
                ohlcv_data["cvd"] = (ohlcv_data["buy_qty"] -
                                     ohlcv_data["sell_qty"]).cumsum()

        else:
            raise ValueError(
                "Input data is missing required columns. Expected either raw trade "
                "fields {price, quantity} or OHLCV fields {open, high, low, close, volume}. "
                f"Columns present: {sorted(raw_trades.columns.tolist())}")

        data_loader = MarketDataLoader()
        data_loader.raw_data = ohlcv_data
        multi_tf_data = data_loader.get_multi_timeframe_data()

        feature_engineer = MultiTimeframeComprehensiveEngineer(
            scaler_type="standard")
        engineered_data = feature_engineer.engineer_features(multi_tf_data,
                                                             fit=True)

        # Optionally filter features by Top-K list
        if args.use_top_factors:
            try:
                top_list = load_top_factors_list(args.use_top_factors)
                if not top_list:
                    print(
                        "   ⚠️ Top factors list is empty; skipping filtering")
                else:
                    print(
                        f"   🔎 Applying Top-K filter with {len(top_list)} factors"
                    )
                    engineered_data = filter_engineered_by_topk(
                        engineered_data, top_list)
            except Exception as exc:  # noqa: BLE001
                print(f"   ⚠️ Failed to apply Top-K filter: {exc}")

        # Optionally compress features using a provided autoencoder
        if args.use_autoencoder:
            if not args.encoding_dim:
                print(
                    "   ❌ --encoding-dim is required when --use-autoencoder is provided"
                )
                return None
            try:
                # Derive input_dim from a representative timeframe
                any_tf = next(iter(engineered_data.values()))
                input_dim = any_tf.shape[1]
                encoding_dim = int(args.encoding_dim)
                autoencoder = UnifiedAutoencoder(
                    input_dim,
                    encoding_dim,
                    architecture="production",
                )
                import torch  # local import to avoid unnecessary dependency at import time

                state = torch.load(args.use_autoencoder, map_location="cpu")
                autoencoder.load_state_dict(state)
                autoencoder.eval()

                def _transform_df(df: pd.DataFrame) -> pd.DataFrame:
                    with torch.no_grad():
                        X = torch.as_tensor(df.values, dtype=torch.float32)
                        _, Z = autoencoder(X)
                        z_np = Z.numpy()
                    cols = [
                        f"compressed_feature_{i}" for i in range(z_np.shape[1])
                    ]
                    return pd.DataFrame(z_np, index=df.index, columns=cols)

                transformed: Dict[str, pd.DataFrame] = {}
                for tf, df in engineered_data.items():
                    transformed[tf] = _transform_df(df)
                engineered_data = transformed
                print(
                    f"   ✅ Applied autoencoder compression to {len(engineered_data)} timeframes"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"   ❌ Failed to apply autoencoder compression: {exc}")
                return None

        strategy = MLTradingStrategy(forward_bars=args.forward_bars)
        strategy.data_loader = data_loader
        strategy.feature_engineer = feature_engineer

        metrics = strategy.train_strategy()

        period_label = _format_period_label(actual_start, actual_end)
        model_basename = build_model_basename(args.model_name, symbol,
                                              period_label)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"{model_basename}.pkl"
        scaler_path = output_dir / f"{model_basename}_scalers.pkl"
        info_path = output_dir / f"{model_basename}_info.json"

        if (not args.overwrite
                and any(path.exists()
                        for path in (model_path, scaler_path, info_path))):
            print(
                f"❌ Model artifacts already exist for {symbol}. Use --overwrite to replace them."
            )
            return None

        model_data = {
            "strategy": strategy,
            "data_loader": data_loader,
            "feature_engineer": feature_engineer,
            "engineered_data": engineered_data,
            "metrics": metrics,
            "training_date": datetime.now(),
            "symbol": symbol,
            "data_files": config.data_paths,
            "date_range": (actual_start, actual_end),
            "data_info": {
                "total_bars":
                len(ohlcv_data),
                "timeframes": {
                    tf: len(df)
                    for tf, df in multi_tf_data.items()
                },
                "price_range": (
                    float(ohlcv_data["close"].min()),
                    float(ohlcv_data["close"].max()),
                ),
            },
        }

        with open(model_path, "wb") as handle:
            pickle.dump(model_data, handle)

        feature_engineer.save_scalers(str(scaler_path))

        model_info = {
            "model_path":
            str(model_path),
            "scaler_path":
            str(scaler_path),
            "training_date":
            datetime.now().isoformat(),
            "symbol":
            symbol,
            "data_files":
            config.data_paths,
            "actual_start":
            actual_start.isoformat(),
            "actual_end":
            actual_end.isoformat(),
            "total_bars":
            len(ohlcv_data),
            "timeframes": {
                tf: len(df)
                for tf, df in multi_tf_data.items()
            },
            "price_range": (
                float(ohlcv_data["close"].min()),
                float(ohlcv_data["close"].max()),
            ),
            "metrics":
            metrics,
            "feature_engineering":
            "comprehensive",
        }

        with open(info_path, "w") as handle:
            json.dump(model_info, handle, indent=2, default=str)

        print(f"   ✓ Model saved to {model_path}")
        print(f"   ✓ Scalers saved to {scaler_path}")
        print(f"   ✓ Model info saved to {info_path}")

        # Generate HTML report
        try:
            from ml_trading.pipeline.dimensionality.report_generator import write_training_report
            report_path = str(info_path).replace(".json", "_report.html")
            write_training_report(str(info_path), report_path)
            print(f"   ✓ Training report saved to {report_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"   ⚠️  Failed to generate HTML report: {exc}")

        return SymbolTrainingResult(
            symbol=symbol,
            model_path=model_path,
            scaler_path=scaler_path,
            info_path=info_path,
            start=actual_start,
            end=actual_end,
            metrics=metrics,
            total_bars=len(ohlcv_data),
        )

    finally:
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(args=list(argv) if argv is not None else None)

    try:
        configs, start_ts, end_ts = resolve_symbol_configs(args)
    except ValueError as err:
        print(f"❌ {err}")
        return

    summaries: List[SymbolTrainingResult] = []
    for config in configs:
        result = train_symbol(args, config)
        if result is not None:
            summaries.append(result)

    if not summaries:
        print("❌ No models were trained.")
        return

    print("\n=== Training Summary ===")
    for summary in summaries:
        print(
            f"{summary.symbol}: {summary.start.strftime('%Y-%m-%d')} → {summary.end.strftime('%Y-%m-%d')} "
            f"| samples={summary.total_bars:,} | model={summary.model_path.name}"
        )


if __name__ == "__main__":
    main()
