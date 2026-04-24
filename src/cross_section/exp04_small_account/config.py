"""exp04 小资金配置常量。

目标账户规模：1 万美金
约束：
    - 同时持仓 <= 4 个币（2 多 2 空），每腿约 $2500 notional
    - 持仓周期 >= 7 天（默认 14 天）
    - 候选池聚焦高流动性（24h volume top），排除 meme 与新上币
    - 费率按 Binance perp taker 0.04% + 滑点 0.04% = 8 bps/side
"""

from __future__ import annotations

from typing import List

# 20 个 "高流动性 + 长历史" 候选池（全部在 exp02 sectors 里已有）
LIQUID_POOL: List[str] = [
    # L1 major / alt（市值高、永续成熟）
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "APTUSDT",
    "SUIUSDT",
    "TRXUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "TONUSDT",
    # L2 / DeFi 龙头
    "ARBUSDT",
    "OPUSDT",
    "UNIUSDT",
    "LINKUSDT",
]

# 默认参数
ACCOUNT_SIZE_USD: float = 10_000.0
MAX_LONGS: int = 2
MAX_SHORTS: int = 2
HOLD_BARS_DEFAULT: int = 24 * 14  # 14 天
FEE_BPS_PER_SIDE: float = 8.0  # 0.04% taker + 0.04% 滑点
STOP_LOSS_PER_LEG: float = 0.15  # 单腿亏损 15% 强制平仓
MIN_NOTIONAL_USD: float = 200.0  # Binance perp 最小下单
