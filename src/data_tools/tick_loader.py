"""Helpers for loading Binance aggTrades (tick-level) data on demand."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


def _parse_timeframe_to_minutes(tf: str) -> Optional[int]:
    """Utility used by callers; kept here for convenience."""
    tf = tf.strip().upper()
    if tf.endswith("T"):
        try:
            return int(tf[:-1])
        except ValueError:
            return None
    if tf.endswith("H"):
        try:
            return int(float(tf[:-1]) * 60)
        except ValueError:
            return None
    if tf.endswith("D"):
        try:
            return int(float(tf[:-1]) * 1440)
        except ValueError:
            return None
    if tf.isdigit():
        return int(tf)
    return None


def _month_range(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> List[Tuple[int, int]]:
    start_period = start_ts.to_period("M")
    end_period = end_ts.to_period("M")
    periods = pd.period_range(start_period, end_period, freq="M")
    return [(p.year, p.month) for p in periods]


def _detect_csv_name(zip_ref: zipfile.ZipFile) -> str:
    for name in zip_ref.namelist():
        if name.endswith(".csv"):
            return name
    return zip_ref.namelist()[0]


def _read_tick_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_name = _detect_csv_name(zf)
        with zf.open(csv_name) as handle:
            first_line = handle.readline().decode("utf-8", errors="ignore")
        read_params = {"low_memory": False}
        if first_line.strip().split(",")[0].replace(".", "").isdigit():
            read_params.update(
                {
                    "header": None,
                    "names": [
                        "agg_trade_id",
                        "price",
                        "quantity",
                        "first_trade_id",
                        "last_trade_id",
                        "transact_time",
                        "is_buyer_maker",
                    ],
                }
            )
        with zf.open(csv_name) as handle:
            df = pd.read_csv(handle, **read_params)
    return df


def _load_month_ticks(
    symbol: str,
    year: int,
    month: int,
    ticks_dir: Path,
    cache_dir: Optional[Path],
) -> pd.DataFrame:
    cache_file: Optional[Path] = None
    if cache_dir is not None:
        cache_file = cache_dir / symbol / f"{symbol}_{year}-{month:02d}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)

    zip_path = ticks_dir / f"{symbol}-aggTrades-{year}-{month:02d}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Missing tick ZIP: {zip_path}. Run `make data-download` first."
        )
    raw_df = _read_tick_csv_from_zip(zip_path)
    if "transact_time" not in raw_df.columns or "price" not in raw_df.columns:
        raise ValueError(f"Unexpected schema inside {zip_path}")
    ticks_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw_df["transact_time"], unit="ms"),
            "price": pd.to_numeric(raw_df["price"], errors="coerce"),
            "volume": pd.to_numeric(
                raw_df.get("quantity", raw_df.get("volume")), errors="coerce"
            ),
        }
    ).dropna()

    if "is_buyer_maker" in raw_df.columns:
        sides = np.where(raw_df["is_buyer_maker"].astype(bool), -1, 1)
    else:
        sides = np.sign(ticks_df["volume"]).replace(0, 1)
    ticks_df["side"] = sides

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        ticks_df.to_parquet(cache_file, index=False)

    return ticks_df


def load_tick_data(
    symbol: str,
    start_ts: str,
    end_ts: str,
    ticks_dir: str = "data/parquet_data",
    lookback_minutes: int = 60,
) -> pd.DataFrame:
    """Load tick parquet range directly (used for ad-hoc analyses)."""
    tick_files = list_tick_files(
        symbol, start_ts, end_ts, ticks_dir=ticks_dir, lookback_minutes=lookback_minutes
    )
    frames = []
    for path in tick_files:
        df = pd.read_parquet(path)
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        raise ValueError("No tick data loaded; please verify tick parquet path.")

    ticks = pd.concat(frames, ignore_index=True)
    start = pd.to_datetime(start_ts) - pd.Timedelta(minutes=lookback_minutes)
    end = pd.to_datetime(end_ts) + pd.Timedelta(minutes=lookback_minutes)
    mask = (ticks["timestamp"] >= start) & (ticks["timestamp"] <= end)
    ticks = ticks.loc[mask]
    if ticks.empty:
        raise ValueError("Tick data is empty after filtering by time range.")

    ticks = ticks.sort_values("timestamp").set_index("timestamp")
    return ticks[["price", "volume", "side"]]


def serialize_tick_loader_params(params: dict) -> str:
    """Utility to store loader params inside feature compute_params."""
    return json.dumps(params)


def deserialize_tick_loader_params(payload: str) -> dict:
    data = json.loads(payload)
    return data


def list_tick_files(
    symbol: str,
    start_ts: str,
    end_ts: str,
    ticks_dir: str,
    lookback_minutes: int = 60,
) -> List[str]:
    """
    Return existing monthly tick parquet files covering [start_ts, end_ts].
    """
    ticks_root = Path(ticks_dir)
    start = pd.to_datetime(start_ts) - pd.Timedelta(minutes=lookback_minutes)
    end = pd.to_datetime(end_ts) + pd.Timedelta(minutes=lookback_minutes)
    months = _month_range(start, end)

    tick_files: List[str] = []
    for year, month in months:
        file_path = ticks_root / f"{symbol}_{year}-{month:02d}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required tick parquet not found: {file_path}. "
                "Please run scripts/data_conversion/convert_zip_to_parquet.py first."
            )
        tick_files.append(str(file_path))
    return sorted(tick_files)


def build_tick_loader_payload(
    symbol: str,
    start_ts: str,
    end_ts: str,
    ticks_dir: str,
    lookback_minutes: int = 60,
) -> str:
    tick_files = list_tick_files(
        symbol,
        start_ts,
        end_ts,
        ticks_dir=ticks_dir,
        lookback_minutes=lookback_minutes,
    )
    payload = {
        "symbol": symbol.upper(),
        "tick_files": tick_files,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "lookback_minutes": lookback_minutes,
    }
    return serialize_tick_loader_params(payload)


def _estimate_bucket_volume_from_cache(
    cache_files: List[str],
    lookback_days: int = 7,
    quantile: float = 0.3,
) -> float:
    hourly_parts = []
    for path_str in cache_files:
        path = Path(path_str)
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["timestamp", "volume"])
        if df.empty:
            continue
        df = df.dropna(subset=["timestamp", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        hourly_parts.append(df["volume"].resample("1H").sum())

    if not hourly_parts:
        return 100.0

    hourly = pd.concat(hourly_parts).sort_index()
    lookback_hours = max(1, lookback_days * 24)
    typical = hourly.rolling(window=lookback_hours, min_periods=1).quantile(quantile)
    bucket_volume = (
        float(typical.iloc[-1]) if not typical.empty else float(hourly.mean())
    )
    return max(bucket_volume, 1e-6)


def _get_monthly_vpin_cache_key(
    file_path: str,
    bucket_volume: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> str:
    """生成按月VPIN缓存的键"""
    import hashlib

    path = Path(file_path)
    month_str = path.stem.split("_")[-1] if "_" in path.stem else path.stem
    key_str = f"vpin_monthly_{month_str}_{bucket_volume:.6f}_{start.isoformat()}_{end.isoformat()}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _load_monthly_vpin_cache(
    cache_dir: Path, cache_key: str
) -> Optional[List[Tuple[pd.Timestamp, float]]]:
    """从缓存加载单月的VPIN buckets"""
    cache_file = cache_dir / f"{cache_key}.pkl"
    if cache_file.exists():
        try:
            import pickle

            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"      ⚠️  Failed to load cache {cache_key}: {e}", flush=True)
    return None


def _save_monthly_vpin_cache(
    cache_dir: Path, cache_key: str, buckets: List[Tuple[pd.Timestamp, float]]
):
    """保存单月的VPIN buckets到缓存"""
    cache_file = cache_dir / f"{cache_key}.pkl"
    try:
        import pickle

        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(buckets, f)
    except Exception as e:
        print(f"      ⚠️  Failed to save cache {cache_key}: {e}", flush=True)


def _compute_vpin_buckets_for_month(
    path: Path,
    bucket_volume: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[Tuple[pd.Timestamp, float]]:
    """计算单月的VPIN buckets"""
    buckets = []
    current_buy = 0.0
    current_sell = 0.0
    filled_volume = 0.0

    df = pd.read_parquet(path, columns=["timestamp", "volume", "side"])
    if df.empty:
        return buckets

    df = df.dropna(subset=["timestamp", "volume", "side"])
    timestamps = pd.to_datetime(df["timestamp"]).to_numpy()
    volumes = df["volume"].astype(float).to_numpy()
    sides = df["side"].astype(int).to_numpy()

    for ts, vol, side in zip(timestamps, volumes, sides):
        if ts < start or ts > end:
            continue
        remaining = vol
        while remaining > 0:
            space_left = bucket_volume - filled_volume
            trade = min(remaining, space_left)
            if side == 1:
                current_buy += trade
            else:
                current_sell += trade
            filled_volume += trade
            remaining -= trade

            if filled_volume >= bucket_volume - 1e-9:
                imbalance = abs(current_buy - current_sell)
                buckets.append((pd.Timestamp(ts), imbalance / bucket_volume))
                current_buy = 0.0
                current_sell = 0.0
                filled_volume = 0.0

    return buckets


def compute_vpin_from_cached_ticks(
    cache_files: List[str],
    start_ts: str,
    end_ts: str,
    bucket_volume: Optional[float],
    n_buckets: int,
    adaptive: bool,
    quantile: float = 0.3,
    lookback_days: int = 7,
    lookback_minutes: int = 60,
    monthly_cache_dir: Optional[str] = "cache/features/monthly",
) -> pd.Series:
    """
    从tick文件计算VPIN，支持按月缓存

    Args:
        cache_files: Tick parquet文件列表
        start_ts: 开始时间
        end_ts: 结束时间
        bucket_volume: Bucket volume（如果为None则自适应计算）
        n_buckets: 滚动窗口大小
        adaptive: 是否自适应bucket volume
        quantile: 自适应计算时的分位数
        lookback_days: 自适应计算时的回看天数
        lookback_minutes: 时间范围扩展的分钟数
        monthly_cache_dir: 按月缓存目录（如果为None则禁用缓存）
    """
    if not cache_files:
        raise ValueError("No cached tick files provided for VPIN computation.")

    start = pd.to_datetime(start_ts) - pd.Timedelta(minutes=lookback_minutes)
    end = pd.to_datetime(end_ts) + pd.Timedelta(minutes=lookback_minutes)

    cache_files = sorted(cache_files)
    if bucket_volume is None:
        if adaptive:
            bucket_volume = _estimate_bucket_volume_from_cache(
                cache_files, lookback_days=lookback_days, quantile=quantile
            )
        else:
            bucket_volume = 100.0

    # 按月缓存目录
    cache_dir = Path(monthly_cache_dir) if monthly_cache_dir else None

    # 收集所有月份的buckets
    all_buckets = []
    total_files = len(cache_files)
    cached_count = 0
    computed_count = 0

    for idx, path_str in enumerate(cache_files, 1):
        path = Path(path_str)
        if not path.exists():
            continue

        # 尝试从缓存加载
        cache_key = None
        month_buckets = None
        if cache_dir:
            cache_key = _get_monthly_vpin_cache_key(path_str, bucket_volume, start, end)
            month_buckets = _load_monthly_vpin_cache(cache_dir, cache_key)

        if month_buckets is not None:
            # 使用缓存
            print(f"      -> [{idx}/{total_files}] {path.name} (cached)", flush=True)
            all_buckets.extend(month_buckets)
            cached_count += 1
        else:
            # 计算并缓存
            print(f"      -> [{idx}/{total_files}] {path.name}", flush=True)
            month_buckets = _compute_vpin_buckets_for_month(
                path, bucket_volume, start, end
            )
            all_buckets.extend(month_buckets)
            computed_count += 1

            # 保存缓存
            if cache_dir and cache_key:
                _save_monthly_vpin_cache(cache_dir, cache_key, month_buckets)

    if not all_buckets:
        raise ValueError(
            "VPIN computation produced no buckets; check tick data source."
        )

    # 合并所有buckets并按时间排序
    buckets_df = (
        pd.DataFrame(all_buckets, columns=["timestamp", "vpin"])
        .set_index("timestamp")
        .sort_index()
    )

    # 计算滚动平均
    vpin_series = buckets_df["vpin"].rolling(window=n_buckets, min_periods=1).mean()

    print(
        f"   ✅ VPIN computed with {len(all_buckets)} buckets (bucket_volume={bucket_volume:.2f})",
        flush=True,
    )
    if cache_dir and cached_count > 0:
        print(
            f"   💾 Used {cached_count} cached months, computed {computed_count} new months",
            flush=True,
        )

    return vpin_series.sort_index()
