"""板块定义 + 横截面中性化工具。

板块划分参考行业常用分类（主观但可调整）：
    - L1_MAJOR:  主流 L1（高市值、成熟）
    - L1_ALT:    其他 L1 / L0
    - L2:        Ethereum L2 / Rollup
    - DEFI:      DeFi 蓝筹
    - MEME:      Meme coin
    - AI_DEPIN:  AI / DePIN
    - GAMING_META: 游戏 / 元宇宙
    - EXCHANGE:  交易所平台币
    - PRIVACY:   隐私币
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


SECTOR_MAP: Dict[str, str] = {
    # L1 major
    "BTCUSDT": "L1_MAJOR",
    "ETHUSDT": "L1_MAJOR",
    "SOLUSDT": "L1_MAJOR",
    # L1 alt / L0
    "ADAUSDT": "L1_ALT",
    "XRPUSDT": "L1_ALT",
    "AVAXUSDT": "L1_ALT",
    "DOTUSDT": "L1_ALT",
    "ATOMUSDT": "L1_ALT",
    "NEARUSDT": "L1_ALT",
    "APTUSDT": "L1_ALT",
    "SUIUSDT": "L1_ALT",
    "TONUSDT": "L1_ALT",
    "SEIUSDT": "L1_ALT",
    "TRXUSDT": "L1_ALT",
    "ETCUSDT": "L1_ALT",
    "LTCUSDT": "L1_ALT",
    "BCHUSDT": "L1_ALT",
    "FILUSDT": "L1_ALT",
    "XLMUSDT": "L1_ALT",
    "XTZUSDT": "L1_ALT",
    "ICPUSDT": "L1_ALT",
    "TIAUSDT": "L1_ALT",
    "INJUSDT": "L1_ALT",
    "HYPEUSDT": "L1_ALT",
    "FTMUSDT": "L1_ALT",
    "RUNEUSDT": "L1_ALT",
    "STXUSDT": "L1_ALT",
    # L2
    "ARBUSDT": "L2",
    "OPUSDT": "L2",
    "MATICUSDT": "L2",
    "ZKUSDT": "L2",
    "STRKUSDT": "L2",
    # DeFi
    "UNIUSDT": "DEFI",
    "AAVEUSDT": "DEFI",
    "MKRUSDT": "DEFI",
    "COMPUSDT": "DEFI",
    "CRVUSDT": "DEFI",
    "SNXUSDT": "DEFI",
    "LDOUSDT": "DEFI",
    "LINKUSDT": "DEFI",
    "DYDXUSDT": "DEFI",
    "JUPUSDT": "DEFI",
    "PYTHUSDT": "DEFI",
    "ENAUSDT": "DEFI",
    "JTOUSDT": "DEFI",
    "GMXUSDT": "DEFI",
    # Meme
    "DOGEUSDT": "MEME",
    "WIFUSDT": "MEME",
    "BOMEUSDT": "MEME",
    "BRETTUSDT": "MEME",
    "MEWUSDT": "MEME",
    "NEIROUSDT": "MEME",
    "POPCATUSDT": "MEME",
    "PONKEUSDT": "MEME",
    "TURBOUSDT": "MEME",
    # AI / DePIN
    "RNDRUSDT": "AI_DEPIN",
    # Gaming / meta
    "SANDUSDT": "GAMING_META",
    "MANAUSDT": "GAMING_META",
    "GALAUSDT": "GAMING_META",
    "IMXUSDT": "GAMING_META",
    # Exchange
    "BNBUSDT": "EXCHANGE",
    # Privacy
    "XMRUSDT": "PRIVACY",
}


def get_sectors(symbols: List[str], default: str = "OTHER") -> pd.Series:
    return pd.Series({s: SECTOR_MAP.get(s, default) for s in symbols}, name="sector")


def zscore(row: pd.Series) -> pd.Series:
    """对一行做 z-score（NaN safe）。"""
    x = row.astype(float)
    m = x.mean()
    sd = x.std()
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(0.0, index=x.index)
    return (x - m) / sd


def sector_neutralize(factor: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """每个时点，在每个板块内做 z-score（减板块均值、除板块标准差）。
    单品种板块：归为 0。板块内 <=1 个有效值时同样归 0。
    """
    out = factor.copy() * np.nan
    sec_groups: Dict[str, List[str]] = {}
    for sym, sec in sectors.items():
        sec_groups.setdefault(sec, []).append(sym)

    for sec, syms in sec_groups.items():
        syms_in = [s for s in syms if s in factor.columns]
        if len(syms_in) == 0:
            continue
        sub = factor[syms_in]
        if len(syms_in) == 1:
            out[syms_in] = 0.0
            continue
        mu = sub.mean(axis=1)
        sd = sub.std(axis=1).replace(0.0, np.nan)
        z = sub.sub(mu, axis=0).div(sd, axis=0)
        out[syms_in] = z.fillna(0.0)
    return out


def cross_sectional_zscore(factor: pd.DataFrame) -> pd.DataFrame:
    """不分板块的横截面 z-score。"""
    mu = factor.mean(axis=1)
    sd = factor.std(axis=1).replace(0.0, np.nan)
    return factor.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)
