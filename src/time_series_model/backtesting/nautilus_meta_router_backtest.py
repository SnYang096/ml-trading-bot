#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml

try:
    from nautilus_trader.backtest.node import BacktestNode
    from nautilus_trader.backtest.config import (
        BacktestRunConfig,
        BacktestVenueConfig,
        BacktestDataConfig,
        BacktestEngineConfig,
    )
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.enums import (
        BarAggregation,
        PriceType,
        AggregationSource,
        OmsType,
        AccountType,
        BookType,
        AggressorSide,
    )
    from nautilus_trader.model import BarType, BarSpecification
    from nautilus_trader.model.data import Bar, TradeTick
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
    from nautilus_trader.trading.config import ImportableStrategyConfig
    from nautilus_trader.adapters.binance import (
        BINANCE,
        BINANCE_CLIENT_ID,
        BinanceAccountType,
        BinanceDataClientConfig,
        BinanceLiveDataClientFactory,
    )

    NAUTILUS_AVAILABLE = True
except Exception as e:  # pragma: no cover
    NAUTILUS_AVAILABLE = False
    _IMPORT_ERR = e


def parse_timeframe(timeframe: str) -> Tuple[BarSpecification, str]:
    tf = str(timeframe).upper()
    if tf.endswith("T"):
        minutes = int(tf[:-1])
        if minutes % 60 == 0:
            spec = BarSpecification(
                int(minutes / 60), BarAggregation.HOUR, PriceType.LAST
            )
            return spec, f"{int(minutes/60)}H"
        spec = BarSpecification(minutes, BarAggregation.MINUTE, PriceType.LAST)
        return spec, f"{minutes}T"
    if tf.endswith("H"):
        hours = int(tf[:-1])
        spec = BarSpecification(hours, BarAggregation.HOUR, PriceType.LAST)
        return spec, f"{hours}H"
    if tf.endswith("D"):
        days = int(tf[:-1])
        spec = BarSpecification(days, BarAggregation.DAY, PriceType.LAST)
        return spec, f"{days}D"
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _load_tick_df(
    data_dir: Path, symbol: str, *, max_files: int | None = None
) -> pd.DataFrame:
    files = sorted(data_dir.glob(f"{symbol}_*.parquet"))
    if max_files is not None and max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No parquet files found for {symbol} in {data_dir}")
    dfs = [pd.read_parquet(p) for p in files]
    df = pd.concat(dfs, ignore_index=True)
    if "timestamp" not in df.columns:
        raise ValueError("tick data must contain timestamp column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    # Nautilus raw builders expect tz-naive nanoseconds
    if hasattr(df["timestamp"].dt, "tz_localize"):
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.dropna(subset=["timestamp"])
    return df


def _filter_time(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()


def _load_time_windows(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "windows" in payload:
        return list(payload["windows"] or [])
    if isinstance(payload, list):
        return payload
    raise ValueError("time windows json must be a list or {'windows': [...]} object")


def _normalize_time_windows(windows: List[dict]) -> List[dict]:
    out = []
    for w in windows or []:
        if not isinstance(w, dict):
            continue
        start = pd.to_datetime(w.get("start"), errors="coerce", utc=True)
        end = pd.to_datetime(w.get("end"), errors="coerce", utc=True)
        if pd.isna(start) or pd.isna(end):
            continue
        start = start.tz_convert(None)
        end = end.tz_convert(None)
        if end < start:
            continue
        out.append(
            {
                "start": start,
                "end": end,
                "symbol": w.get("symbol"),
                "source": w.get("source"),
            }
        )
    return out


def _filter_time_windows(
    df: pd.DataFrame, windows: List[dict], *, symbol: str
) -> pd.DataFrame:
    if not windows:
        return df
    sym = str(symbol)
    applicable = []
    for w in windows:
        w_sym = w.get("symbol")
        if w_sym is None or str(w_sym) == sym:
            applicable.append((w["start"], w["end"]))
    if not applicable:
        return df.iloc[0:0].copy()
    mask = np.zeros(len(df), dtype=bool)
    ts = df["timestamp"]
    for start, end in applicable:
        mask |= (ts >= start) & (ts <= end)
    return df.loc[mask].copy()


def _summarize_windows(windows: List[dict]) -> dict:
    if not windows:
        return {"total_windows": 0}
    starts = [w["start"] for w in windows]
    ends = [w["end"] for w in windows]
    return {
        "total_windows": int(len(windows)),
        "min_start": min(starts).isoformat(),
        "max_end": max(ends).isoformat(),
    }


def _infer_currencies(symbol: str) -> tuple[str, str]:
    sym = str(symbol).upper()
    if sym.endswith("USDT"):
        return sym[:-4], "USDT"
    if sym.endswith("USD"):
        return sym[:-3], "USD"
    if sym.endswith("BTC"):
        return sym[:-3], "BTC"
    return sym, "USD"


def _build_instrument(
    *,
    instrument_id: InstrumentId,
    symbol: str,
    price_prec: int,
    size_prec: int,
    ts_event: int,
) -> "CryptoPerpetual":
    from nautilus_trader.model.instruments import CryptoPerpetual
    from nautilus_trader.model.objects import Price, Quantity
    from nautilus_trader.model.identifiers import Symbol
    from nautilus_trader.model.currencies import Currency

    base, quote = _infer_currencies(symbol)
    price_increment = Price.from_str("0." + "0" * (price_prec - 1) + "1")
    size_increment = Quantity.from_str("0." + "0" * (size_prec - 1) + "1")
    return CryptoPerpetual(
        instrument_id,
        Symbol(symbol),
        Currency.from_str(base),
        Currency.from_str(quote),
        Currency.from_str(quote),
        False,
        price_prec,
        size_prec,
        price_increment,
        size_increment,
        ts_event,
        ts_event,
    )


if TYPE_CHECKING:
    from nautilus_trader.model.instruments import CryptoPerpetual


def _build_trade_ticks(
    df: pd.DataFrame, instrument_id: InstrumentId, price_prec: int, size_prec: int
) -> List[TradeTick]:
    price_raw = (
        (df["price"].astype(float) * (10**price_prec))
        .round()
        .astype("int64")
        .to_numpy(dtype="int64")
    )
    size_raw = (
        (df["volume"].astype(float) * (10**size_prec))
        .round()
        .astype("int64")
        .to_numpy(dtype="int64")
    )
    ts_event = df["timestamp"].astype("datetime64[ns]").astype("int64").to_numpy()
    ts_init = ts_event
    aggressor = df.get("side", 1).astype(int).to_numpy()
    buy_val = int(AggressorSide.BUYER)
    sell_val = int(AggressorSide.SELLER)
    aggressor_side = np.where(aggressor > 0, buy_val, sell_val).astype(np.int8)
    trade_id = [str(i) for i in range(len(df))]
    return TradeTick.from_raw_arrays_to_list(
        instrument_id,
        price_prec,
        size_prec,
        price_raw.astype("float64"),
        size_raw.astype("float64"),
        aggressor_side.astype("uint8"),
        trade_id,
        ts_event.astype("uint64"),
        ts_init.astype("uint64"),
    )


def _build_bars(
    df: pd.DataFrame,
    bar_type: BarType,
    price_prec: int,
    size_prec: int,
    resample_rule: str,
) -> List[Bar]:
    bar_df = (
        df.set_index("timestamp")
        .resample(resample_rule)
        .agg(
            {
                "price": ["first", "max", "min", "last"],
                "volume": "sum",
            }
        )
        .dropna()
    )
    bar_df.columns = ["open", "high", "low", "close", "volume"]
    opens = bar_df["open"].to_numpy(dtype="float64")
    highs = bar_df["high"].to_numpy(dtype="float64")
    lows = bar_df["low"].to_numpy(dtype="float64")
    closes = bar_df["close"].to_numpy(dtype="float64")
    volumes = bar_df["volume"].to_numpy(dtype="float64")
    ts_events = (
        bar_df.index.astype("datetime64[ns]").astype("int64").to_numpy(dtype="uint64")
    )
    ts_inits = ts_events
    return Bar.from_raw_arrays_to_list(
        bar_type,
        price_prec,
        size_prec,
        opens,
        highs,
        lows,
        closes,
        volumes,
        ts_events,
        ts_inits,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Nautilus backtest for MetaRouterStrategy."
    )
    ap.add_argument("--data-dir", default="data/parquet_data")
    ap.add_argument("--symbol", required=False)
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--timeframe", default="240T")
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--trade-size", type=float, default=0.001)
    ap.add_argument(
        "--live-config",
        default="config/nnmultihead/live/meta_router_live_config.yaml",
    )
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--config-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--live-feature-plan", default=None)
    ap.add_argument("--evidence-quantiles", default=None)
    ap.add_argument("--output-dir", default="results/backtest")
    ap.add_argument(
        "--env-file",
        default=None,
        help="Load environment variables from a file (KEY=VALUE per line)",
    )
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument(
        "--no-trade-ticks",
        action="store_true",
        help="Disable trade ticks (bar-only smoke runs)",
    )
    ap.add_argument(
        "--use-adapter-data",
        action="store_true",
        help="Use Binance adapter data client (requires API keys in env)",
    )
    ap.add_argument(
        "--adapter-testnet",
        action="store_true",
        help="Use Binance testnet credentials",
    )
    ap.add_argument(
        "--adapter-account-type",
        default="USDT_FUTURES",
        choices=["SPOT", "USDT_FUTURES"],
        help="Binance account type for adapter data client",
    )
    ap.add_argument(
        "--adapter-api-key-env",
        default=None,
        help="Env var name for Binance API key (optional override)",
    )
    ap.add_argument(
        "--adapter-api-secret-env",
        default=None,
        help="Env var name for Binance API secret (optional override)",
    )
    ap.add_argument("--catalog-dir", default=None)
    ap.add_argument("--reuse-catalog", action="store_true")
    ap.add_argument("--price-prec", type=int, default=8)
    ap.add_argument("--size-prec", type=int, default=8)
    ap.add_argument(
        "--time-windows-json",
        default=None,
        help="Optional JSON file with time windows for event sampling backtest",
    )
    return ap.parse_args()


def _load_env_file(path: str) -> None:
    env_path = Path(path)
    if not env_path.exists():
        raise FileNotFoundError(f"env file not found: {env_path}")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    if not NAUTILUS_AVAILABLE:
        raise RuntimeError(
            f"Nautilus Trader not installed: {_IMPORT_ERR}. Install with pip install nautilus-trader"
        )
    args = parse_args()
    if args.env_file:
        _load_env_file(args.env_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.live_feature_plan:
        os.environ["MLBOT_LIVE_FEATURE_PLAN_YAML"] = str(args.live_feature_plan)
    if args.evidence_quantiles:
        os.environ["MLBOT_EVIDENCE_QUANTILES_JSON"] = str(args.evidence_quantiles)
    data_dir = Path(args.data_dir)
    if not args.symbol and not args.symbols:
        raise ValueError("Provide --symbol or --symbols")
    symbols = (
        [s.strip() for s in str(args.symbols).split(",") if s.strip()]
        if args.symbols
        else [args.symbol]
    )

    spec, resample_rule = parse_timeframe(args.timeframe)
    instrument_ids = [InstrumentId.from_str(f"{s}.BINANCE") for s in symbols]
    data_client_id_str = str(BINANCE_CLIENT_ID) if NAUTILUS_AVAILABLE else "BINANCE"
    adapter_client_name = (
        f"{data_client_id_str}_ADAPTER" if args.use_adapter_data else data_client_id_str
    )

    live_config_path = Path(args.live_config)
    if args.model_path:
        cfg = yaml.safe_load(live_config_path.read_text(encoding="utf-8")) or {}
        nni = cfg.get("nnmultihead_inference") or {}
        nni["enabled"] = True
        nni["model_path"] = str(args.model_path)
        if args.config_dir:
            nni["config_dir"] = str(args.config_dir)
        if args.device:
            nni["device"] = str(args.device)
        cfg["nnmultihead_inference"] = nni
        live_config_path = output_dir / "meta_router_live_config.generated.yaml"
        live_config_path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    catalog_dir = (
        Path(args.catalog_dir)
        if args.catalog_dir
        else Path("results/nautilus_catalog") / f"{'_'.join(symbols)}_{args.timeframe}"
    )
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(path=str(catalog_dir))
    time_windows = []
    if args.time_windows_json:
        if args.reuse_catalog:
            raise ValueError(
                "--time-windows-json requires rebuilding catalog; disable --reuse-catalog"
            )
        time_windows = _normalize_time_windows(
            _load_time_windows(Path(args.time_windows_json))
        )
        if time_windows:
            summary = _summarize_windows(time_windows)
            (output_dir / "time_windows_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            (output_dir / "time_windows.normalized.json").write_text(
                json.dumps(
                    [
                        {
                            "start": w["start"].isoformat(),
                            "end": w["end"].isoformat(),
                            "symbol": w.get("symbol"),
                            "source": w.get("source"),
                        }
                        for w in time_windows
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
    ts_event = pd.Timestamp(args.start_date, tz="UTC").value
    if not args.reuse_catalog:
        for symbol, instrument_id in zip(symbols, instrument_ids):
            df = _load_tick_df(data_dir, symbol, max_files=args.max_files)
            df = _filter_time(df, args.start_date, args.end_date)
            if time_windows:
                df = _filter_time_windows(df, time_windows, symbol=symbol)
            if df.empty:
                raise ValueError(f"No tick data in the given time range for {symbol}")
            instrument = _build_instrument(
                instrument_id=instrument_id,
                symbol=symbol,
                price_prec=args.price_prec,
                size_prec=args.size_prec,
                ts_event=ts_event,
            )
            catalog.write_data([instrument])
            if not args.no_trade_ticks:
                ticks = _build_trade_ticks(
                    df,
                    instrument_id,
                    price_prec=args.price_prec,
                    size_prec=args.size_prec,
                )
                catalog.write_data(ticks)
            bar_type = BarType(instrument_id, spec, AggregationSource.EXTERNAL)
            bars = _build_bars(
                df,
                bar_type,
                price_prec=args.price_prec,
                size_prec=args.size_prec,
                resample_rule=resample_rule,
            )
            catalog.write_data(bars)

    venue_cfg = BacktestVenueConfig(
        name="BINANCE",
        oms_type="NETTING",
        account_type="MARGIN",
        starting_balances=["100000 USDT"],
        base_currency="USDT",
        default_leverage=1.0,
        book_type="L1_MBP",
        bar_execution=True,
        trade_execution=True,
    )
    data_cfgs = []
    for instrument_id in instrument_ids:
        if not args.no_trade_ticks:
            data_cfgs.append(
                BacktestDataConfig(
                    catalog_path=str(catalog_dir),
                    data_cls="nautilus_trader.model.data:TradeTick",
                    instrument_id=instrument_id,
                    start_time=args.start_date,
                    end_time=args.end_date,
                    client_id=adapter_client_name,
                )
            )
        bar_type = BarType(instrument_id, spec, AggregationSource.EXTERNAL)
        data_cfgs.append(
            BacktestDataConfig(
                catalog_path=str(catalog_dir),
                data_cls="nautilus_trader.model.data:Bar",
                bar_types=[str(bar_type)],
                start_time=args.start_date,
                end_time=args.end_date,
                client_id=adapter_client_name,
            )
        )
    data_clients = None
    if args.use_adapter_data:
        key_env = args.adapter_api_key_env
        secret_env = args.adapter_api_secret_env
        if args.adapter_testnet:
            key_env = key_env or "BINANCE_FUTURES_TESTNET_API_KEY"
            secret_env = secret_env or "BINANCE_FUTURES_TESTNET_API_SECRET"
        else:
            key_env = key_env or "BINANCE_API_KEY"
            secret_env = secret_env or "BINANCE_API_SECRET"
        api_key = os.getenv(key_env)
        api_secret = os.getenv(secret_env)
        if not api_key or not api_secret:
            raise ValueError(
                f"Binance API credentials missing. Set {key_env} and {secret_env}."
            )
        account_type = (
            BinanceAccountType.SPOT
            if args.adapter_account_type == "SPOT"
            else BinanceAccountType.USDT_FUTURES
        )
        data_clients = {
            adapter_client_name: BinanceDataClientConfig(
                api_key=api_key,
                api_secret=api_secret,
                account_type=account_type,
                testnet=args.adapter_testnet,
            )
        }
    # config_path now points to a config class; no file-based config needed.
    strategies = []
    for symbol, instrument_id in zip(symbols, instrument_ids):
        bar_type = BarType(instrument_id, spec, AggregationSource.EXTERNAL)
        strategies.append(
            ImportableStrategyConfig(
                strategy_path="src.time_series_model.live.meta_router_strategy:MetaRouterStrategy",
                config_path="src.time_series_model.live.meta_router_strategy:MetaRouterStrategyConfig",
                config={
                    "strategy_name": f"meta_router_{symbol}",
                    "instrument_id": instrument_id,
                    "bar_type": bar_type,
                    "trade_size": float(args.trade_size),
                    "data_client_id": adapter_client_name,
                    "live_config_path": str(live_config_path),
                },
            )
        )
    engine_cfg = BacktestEngineConfig(strategies=strategies)
    run_cfg = BacktestRunConfig(
        venues=[venue_cfg],
        data=data_cfgs,
        engine=engine_cfg,
        start=args.start_date,
        end=args.end_date,
        data_clients=data_clients,
    )
    node = BacktestNode(configs=[run_cfg])
    if args.use_adapter_data:
        node.add_data_client_factory(adapter_client_name, BinanceLiveDataClientFactory)
    node.run()


if __name__ == "__main__":
    main()
