"""exp02 数据加载：价格（重采样自 tick）+ funding rate，对齐到统一 K 线网格。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def month_range(start: str, end: str) -> List[str]:
    s = pd.Timestamp(start + "-01" if len(start) == 7 else start)
    e = pd.Timestamp(end + "-01" if len(end) == 7 else end)
    return [m.strftime("%Y-%m") for m in pd.date_range(s, e, freq="MS")]


def load_price_series(
    symbol: str,
    months: List[str],
    data_dir: Path,
    timeframe: str = "1h",
) -> Optional[pd.Series]:
    frames = []
    for ym in months:
        p = data_dir / f"{symbol}_{ym}.parquet"
        if p.exists():
            fr = pd.read_parquet(p, columns=["timestamp", "price"])
            ts = pd.to_datetime(fr["timestamp"])
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_convert(None)
            fr["timestamp"] = ts
            frames.append(fr)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).set_index("timestamp").sort_index()
    close = df["price"].resample(timeframe).last().ffill()
    close.name = symbol
    return close


def load_funding_series(
    symbol: str,
    months: List[str],
    funding_dir: Path,
    timeframe: str = "1h",
) -> Optional[pd.Series]:
    """Binance funding 每 8 小时一次。这里将其 ffill 到目标 K 线频率，数值单位保持不变（小数形式）。"""
    frames = []
    for ym in months:
        p = funding_dir / f"{symbol}_{ym}_funding_rate.parquet"
        if p.exists():
            fr = pd.read_parquet(p)
            if fr.index.tz is not None:
                fr.index = fr.index.tz_convert(None)
            frames.append(fr)
    if not frames:
        return None
    df = pd.concat(frames).sort_index()
    s = df["funding_rate"].astype(float)
    s = s[~s.index.duplicated(keep="last")]
    s = s.resample(timeframe).last().ffill()
    s.name = symbol
    return s


def build_panels(
    symbols: List[str],
    start: str,
    end: str,
    price_dir: Path,
    funding_dir: Path,
    timeframe: str = "1h",
    min_coverage: float = 0.8,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """返回 {'prices': DF, 'returns': DF, 'funding': DF}，列是 symbol，index 是对齐时间。

    min_coverage: 某币种在全时间轴的非 NaN 占比阈值，低于则剔除（避免新币稀释结果）。
    """
    months = month_range(start, end)
    prices: Dict[str, pd.Series] = {}
    fundings: Dict[str, pd.Series] = {}

    for sym in symbols:
        ps = load_price_series(sym, months, price_dir, timeframe)
        if ps is None:
            if verbose:
                print(f"[SKIP] {sym}: 无价格数据")
            continue
        prices[sym] = ps
        fs = load_funding_series(sym, months, funding_dir, timeframe)
        if fs is not None:
            fundings[sym] = fs
        elif verbose:
            print(f"[WARN] {sym}: 无 funding 数据（仍保留价格）")

    if not prices:
        raise SystemExit("没有任何可用价格数据")

    px = pd.concat(prices, axis=1).sort_index()
    # 确定共同时间轴：任一币种的第一/最后一根
    full_idx = pd.date_range(px.index.min(), px.index.max(), freq=timeframe)
    px = px.reindex(full_idx)

    # 按 coverage 过滤
    cov = px.notna().mean()
    keep = cov[cov >= min_coverage].index.tolist()
    drop = cov[cov < min_coverage].index.tolist()
    if verbose:
        for d in drop:
            print(f"[DROP] {d}: coverage={cov[d]:.2%} < {min_coverage:.0%}")
        print(f"[KEEP] {len(keep)} symbols: {keep}")
    px = px[keep]

    # funding 对齐到同一 index/columns
    if fundings:
        fr = pd.concat(fundings, axis=1).reindex(full_idx)
        fr = fr.reindex(columns=keep)
    else:
        fr = pd.DataFrame(index=full_idx, columns=keep, dtype=float)

    # 对数收益（NaN -> 0，在 factor 层处理缺失）
    rets = np.log(px).diff()

    # 去掉完全为 NaN 的前缀行
    valid_start = rets.dropna(how="all").index.min()
    px = px.loc[valid_start:]
    rets = rets.loc[valid_start:]
    fr = fr.loc[valid_start:]

    return {"prices": px, "returns": rets, "funding": fr}
