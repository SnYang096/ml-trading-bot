import logging
from typing import Dict, Optional
from datetime import datetime, time
from collections import deque
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.core.data import Data
from nautilus_trader.core.message import Event
from nautilus_trader.model.objects import Quantity, Price
from nautilus_trader.core.rust.model import OrderSide, OrderType, TimeInForce
from nautilus_trader.model.data import TradeTick, QuoteTick, Bar
from nautilus_trader.model.events.order import OrderAccepted, OrderFilled
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments.base import Instrument
from nautilus_trader.model.orders.base import Order
from nautilus_trader.model.position import Position
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.indicators import BollingerBands
from nautilus_trader.indicators import MovingAverageType
from nautilus_trader.indicators import VolumeWeightedAveragePrice, MovingAverage, AverageTrueRange
from nautilus_trader.model.data import BarType

from yin_bot.intraday_sniper.indicators.cvd import CVD
from yin_bot.intraday_sniper.indicators.scorer import BreakoutQualityScorer
from yin_bot.intraday_sniper.risk.position_sizer import PositionSizer
from yin_bot.intraday_sniper.filters.time_filter import TimeFilter
from yin_bot.intraday_sniper.filters.event_filter import EventFilter

from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import time
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.identifiers import InstrumentId


class IntradaySniperConfig(StrategyConfig, kw_only=True, frozen=True):
    """
    IntradaySniper策略的配置类
    
    该配置类定义了策略运行所需的所有参数，包括技术指标参数、风险管理参数、
    会话时间设置以及突破质量权重等。
    """
    # 必须字段（没有默认值）
    instrument_id: InstrumentId

    # 可选字段（有默认值）
    order_id_tag: str = "INTRADAY_SNIPE"
    bar_type: str = "ESZ3.TICK_5M_AGG"  # 添加bar_type配置

    # 技术指标参数
    indicators: dict = field(default_factory=lambda: {
        "bollinger_bands": {
            "period": 20,
            "stddev": 2.0
        },
        "adaptive_multi_dim_compression": {
            "indicator_config": {
                "compression_window": 200,
                "volatility_window": 20,
                "volume_window": 20,
                "entropy_window": 20,
                "compression_tdigest": 100.0,
                "minimum_tdigest_warmup": 50,
                "duration_bonus_window": 200,
                "pre_breakout_silence_window": 10,
                "internal_price_density_window": 20
            },
            "weight_config": {
                "compression_atr": 0.15,
                "compression_volume": 0.15,
                "structural_compression": 0.15,
                "momentum_convergence": 0.10,
                "direction_ordered": 0.08,
                "volatility_dense": 0.08,
                "duration_bonus": 0.10,
                "pre_breakout_silence": 0.10,
                "internal_price_density": 0.09
            },
            "threshold_config": {
                "compression_atr_threshold_quantile": 0.3,
                "compression_volume_threshold_quantile": 0.4,
                "structural_compression_body_ratio": 0.5,
                "structural_compression_atr_ratio": 0.8,
                "momentum_convergence_price_threshold": 1e-5,
                "direction_ordered_entropy_threshold": 0.7,
                "volatility_dense_threshold": 0.3,
                "compression_confidence_threshold": 0.6,
                "expansion_confidence_threshold": 0.3,
                "volatility_buffer_base_threshold": 0.1,
                "pre_breakout_silence_atr_threshold_quantile": 0.2,
                "internal_price_density_threshold": 0.6
            }
        },
        "cvd": {
            "window_size": 100
        },
        "order_flow": {
            "delta_ma_window": 5
        }
    })

    # 突破质量评分器
    breakout_quality_scorer: dict = field(default_factory=lambda: {
        "weights": {
            "price_breakout": 0.2,
            "volume_spike": 0.2,
            "delta_strength": 0.2,
            "cvd_momentum": 0.15,
            "order_absorption_bar": 0.15,
            "high_liquidity_time": 0.1
        }
    })

    # 风险管理参数
    risk_management: dict = field(default_factory=lambda: {
        "risk_per_trade": 0.01,
        "target_r_ratio": 3.0,
        "initial_capital": 100000.0
    })

    # 止损参数
    stop_loss: dict = field(default_factory=lambda: {
        "atr_period": 14,
        "trailing_stop_atr_mult": 1.5
    })

    # 会话时间设置
    session: dict = field(default_factory=lambda: {
        "start": "09:30",
        "end": "15:45"
    })

    # 事件过滤器
    event_filter: dict = field(default_factory=lambda: {
        "cooloff_after_event": 300
    })

    # 日志配置
    logging: dict = field(default_factory=lambda: {
        "level": "INFO"
    })

    def __post_init__(self):
        # 验证权重总和是否为1.0
        weights = self.breakout_quality_scorer["weights"]
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.001:  # 允许小的浮点误差
            raise ValueError(
                f"Breakout quality weights must sum to 1.0, but got {total_weight}"
            )

        # 验证时间格式
        try:
            time.fromisoformat(self.session["start"])
            time.fromisoformat(self.session["end"])
        except ValueError:
            raise ValueError(
                f"Invalid time format for session start ({self.session['start']}) or end ({self.session['end']})"
            )

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        从YAML文件加载配置
        """
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 移除YAML中存在但在配置类中未定义的字段
        allowed_fields = {
            "instrument_id", "order_id_tag", "bar_type", "indicators", 
            "breakout_quality_scorer", "risk_management", "stop_loss",
            "session", "event_filter", "logging"
        }
        
        # 只保留配置类中定义的字段
        filtered_config_dict = {
            key: value for key, value in config_dict.items() 
            if key in allowed_fields
        }
        
        return cls(**filtered_config_dict)