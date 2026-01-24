"""
Backtrader PoC Breakout / False-Breakout Bot (增强版)
- 将公共函数抽取到 BaseStrategy
- orderflow_confirm 改为加权评分 + 动态阈值(percentile)
- 所有指标数据统一写入 indicator_data，并在 stop() 时导出 CSV
- trades_log 也在 stop() 时导出 CSV
- 去掉重复的 poc_series 存储
Author: you (enhanced)
"""
from __future__ import annotations

import os
import math
import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt

# 你原来的 helper / indicators / sizer import 路径保留
from backtrader_project.services.common.helper import plot_candles_with_poc_and_trades_bokeh
from backtrader_project.services.common.helper import compute_metrics
from backtrader_project.services.common.data_loader import build_features, load_trade_data
from backtrader_project.services.common.oflow_features import oflow_features
from ..live.volatility_sizer import VolatilitySizer
from ..indicators.swing_levels import SwingLevels
from ..indicators.volume_profile import VolumeProfilePOC
from ..indicators.zigzag import ZigZag

# =============================
# ---------- CONFIG -----------
# =============================


@dataclass
class Config:
    timeframe: str = "1H"
    tz: Optional[str] = None
    profile_bars: int = 240
    profile_alpha: float = 0.2
    rolling_cum_delta_bars: int = 20
    profile_bins: int = 80
    profile_top_n: int = 1
    poc_every_n_bars: int = 5
    commission: float = 0.0004
    slippage: float = 0.0005
    margin: float = 0.01  # 100倍杠杆 => 保证金比例 1%
    mult: float = 1.0  # 币安USDT永续，1张合约 = 1 USDT
    cash: float = 10_000
    plot_datetime_fmt: str = "%Y-%m-%d %H:%M"
    min_trade_size: float = 0.001  # 最小交易单位（币安 BTC 永续）


CFG = Config()


# PandasDataExtra 保留，你数据帧列名应与此匹配
class PandasDataExtra(bt.feeds.PandasData):
    lines = ('vwap', 'taker_buy_ratio', 'delta_vol', 'cum_delta',
             'rolling_cum_delta')
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'price'),
        ('volume', 'qty'),
        ('openinterest', None),
        ('vwap', 'vwap'),
        ('taker_buy_ratio', 'taker_buy_ratio'),
        ('cum_delta', 'cum_delta'),
        ('delta_vol', 'delta_vol'),
        ('rolling_cum_delta', 'rolling_cum_delta'),
    )


# =============================
# --------- BaseStrategy -------
# =============================


class BaseStrategy(bt.Strategy):
    params = dict(
        leverage=100,  # 杠杆倍数
        min_qty=0.001,  # 最小下单单位（币安 BTC
        # orderflow defaults (子类可以覆盖)
        orderflow_weights=dict(
            vol_spike=1.0,
            cumdelta=1.5,
            taker=1.0,
            vwap=0.8,
            atr_delta=1.0,
        ),
        orderflow_window=200,  # 用于历史分布计算的窗口
        orderflow_percentile=0.65,  # 动态阈值分位
        orderflow_min_threshold=0.4,  # 最低允许阈值（归一化后）

        # file outputs (可由外部传参覆盖)
        metrics_path="metrics.csv",  # 统一记录指标数据, 便于后续分析/绘图
        indicator_path='indicator_data.csv',
        trades_path='trades_log.csv',  # 统一记录交易数据, 便于在主图上画线，目前不支持加仓，加仓和普通开仓一样
        log_file="trades_details_debug.log",  # 更详细的调试文件
    )

    def __init__(self):
        # 交易日志没有直接输出到文件，而是先存在这个数组里面，因为比较少
        self.trades_log: List[Dict[str, Any]] = []
        self.indicator_data: List[Dict[str, Any]] = []
        self._equity_curve: List[float] = []

        # orderflow 历史分数（归一化到 0..1）
        self._orderflow_score_hist: List[float] = []

        # cumdelta rolling zscore 的历史（某些策略需要）
        self._cumdelta_hist: List[float] = []
        self._cumdelta_ma_period = 50

        # 每次运行生成一个唯一的日志文件（在原名上附加时间戳和进程id）
        try:
            base = self.p.log_file
            dirname = os.path.dirname(base)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            name, ext = os.path.splitext(os.path.basename(base))
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
            unique_name = f"{name}_{ts}_{os.getpid()}{ext or '.log'}"
            path = os.path.join(dirname,
                                unique_name) if dirname else unique_name
            self.debug_log_fp = open(path, "a", encoding="utf-8")
            self.log_info(f"Opened debug log {path}")
        except Exception as e:
            # 若出错则回退到原始文件名
            try:
                self.debug_log_fp = open(self.p.log_file,
                                         "a",
                                         encoding="utf-8")
                self.log_info(
                    f"Fallback to log file {self.p.log_file} due to error: {e}"
                )
            except Exception:
                self.debug_log_fp = None

    # 常用工具：烛线信息
    @staticmethod
    def _wick_info(o, h, l, c):
        body = abs(c - o)
        upper = h - max(o, c)
        lower = min(o, c) - l
        return body, upper, lower

    # 压缩比（需要子类在 __init__ 中创建 self.bb / self.atr）
    def _compression_ratio(self):
        try:
            bbw = float(self.bb.top[0] - self.bb.bot[0]) if (
                not math.isnan(self.bb.top[0])
                and not math.isnan(self.bb.bot[0])) else 0.0
            atrv = float(self.atr[0])
            if atrv <= 0: return 0.0
            return bbw / atrv
        except Exception:
            return 0.0

    def _structure_stop(self, long: bool, atr_stop: float, last_sup: float,
                        last_res: float, price: float) -> float:
        stop_price = atr_stop  # 直接使用子类计算好的 atr_stop
        try:
            if self.p.use_structure_stop:
                if long and last_sup:
                    # 对于多头，结构止损位是最近的支撑位，取 atr_stop 和 last_sup 中更保守（更接近当前价格）的一个
                    stop_price = min(last_sup, atr_stop)
                elif not long and last_res:
                    # 对于空头，结构止损位是最近的阻力位，取 atr_stop 和 last_res 中更保守（更接近当前价格）的一个
                    stop_price = max(last_res, atr_stop)
        except Exception:
            pass
        # fallback：如果止损价无效， fallback 到最初的 atr_stop
        if stop_price <= 0:
            stop_price = atr_stop

        self.log_info(
            f"[Structure Stop] long={long} atr_stop={atr_stop:.2f} last_sup={last_sup:.2f} last_res={last_res:.2f} price={price:.2f} stop_price={stop_price:.2f}"
        )
        return stop_price

    # 移动止损（包装）
    def _place_or_move_stop(self, new_stop: float):
        if self.position.size == 0:
            return
        # cancel existing if moving favorably
        if getattr(self, 'protective_stop', None):
            try:
                currp = getattr(self.protective_stop, 'price', None)
                if currp is not None:
                    if (self.position.size > 0
                            and new_stop > currp) or (self.position.size < 0
                                                      and new_stop < currp):
                        try:
                            self.log_info(
                                f"[Cancel STOP] old={currp:.2f} new={new_stop:.2f} size={self.position.size}"
                            )
                            self.broker.cancel(self.protective_stop)
                        except Exception as exception:
                            self.log_info(
                                f"[Cancel STOP Exception] old={currp:.2f} new={new_stop:.2f} size={self.position.size} Exception={exception}"
                            )
                            pass
                        self.protective_stop = None
                    else:
                        return
            except Exception:
                pass
        if self.position.size > 0:
            self.log_info(
                f"[Place STOP LONG] price={new_stop:.2f} size={self.position.size}"
            )
            self.protective_stop = self.sell(exectype=bt.Order.Stop,
                                             price=new_stop,
                                             size=self.position.size)
        else:
            self.log_info(
                f"[Place STOP SHORT] price={new_stop:.2f} size={self.position.size}"
            )
            self.protective_stop = self.buy(exectype=bt.Order.Stop,
                                            price=new_stop,
                                            size=abs(self.position.size))

    def _risk_sized(self, risk_per_unit: float) -> float:
        """
        根据账户资金、风险比例和最小下单量来计算仓位大小（BTC 数量）
        """
        risk_pct = getattr(self, '_risk_pct_runtime',
                           None) or self.p.risk_per_trade_pct

        if risk_per_unit <= 0:
            return 0.0

        min_qty = getattr(self.p, 'min_qty', 0.001)  # 默认 0.001 BTC

        # 只用本金风险，不考虑杠杆
        risk_capital = self.broker.getvalue() * risk_pct

        # 仓位 = 本金风险 / 每单位风险
        size = risk_capital / risk_per_unit

        # 向下取整到最小精度
        size = math.floor(size / min_qty) * min_qty

        self.log_info(
            f"[Risk Sizing] account_value={self.broker.getvalue():.2f} risk_capital={risk_capital:.2f} risk_per_unit={risk_per_unit:.2f} raw_size={risk_capital / risk_per_unit:.4f} final_size={size:.4f} min_qty={min_qty}"
        )
        return max(min_qty, size)

    # 根据历史分数和配置计算动态阈值（percentile）
    def _orderflow_compute_threshold(self):
        hist = np.array(
            self._orderflow_score_hist[-int(self.p.orderflow_window):]) if len(
                self._orderflow_score_hist) > 0 else np.array([])
        if hist.size < 5:
            # 数据不足时返回默认最小阈值
            return float(self.p.orderflow_min_threshold)
        perc = float(self.p.orderflow_percentile)
        # percentile expects 0-100
        perc_val = float(np.percentile(hist, perc * 100.0))
        # 保证不低于 orderflow_min_threshold
        return max(float(self.p.orderflow_min_threshold), perc_val)

    # stop() 中导出数据（可被子类调用或重写）
    def _export_on_stop(self):

        # indicator data
        if getattr(self.p, 'indicator_path', None) and len(
                self.indicator_data) > 0:
            try:
                df = pd.DataFrame(self.indicator_data)
                df.to_csv(self.p.indicator_path, index=False)
                self.log_info(
                    f"Saved indicator data to {self.p.indicator_path}")
            except Exception as e:
                self.log_info(f"Failed to write indicator data: {e}")

        # 详细debug, 直接输出到日志
        if hasattr(self, "debug_log_fp") and self.debug_log_fp:
            self.debug_log_fp.close()
            self.debug_log_fp = None

        # 简单交易日志
        if getattr(self.p, 'trades_path', None) and len(self.trades_log) > 0:
            try:
                df = pd.DataFrame(self.trades_log)
                df.to_csv(self.p.trades_path, index=False)
                self.log_info(f"Saved trades log to {self.p.trades_path}")
            except Exception as e:
                self.log_info(f"Failed to write trades log: {e}")
        # metrics（如果有）
        if getattr(self.p, 'metrics_path', None):
            metrics = compute_metrics(self._equity_curve, self.trades_log)
            row = {
                # 'zigzag_atr_period': self.p.zigzag_atr_period,
                # 'zigzag_atr_mult': self.p.zigzag_atr_mult ,
                'swing_lookback': self.p.swing_lookback,
                'swing_window': self.p.swing_window,
                'bb_period': self.p.bb_period,
                'compression_threshold': self.p.compression_threshold,
                'confirm_delta': self.p.confirm_delta,
                'stop_atr': self.p.stop_atr,
                'take_atr': self.p.take_atr,
                'final_value': metrics.get('final_value', np.nan),
                'return_pct': metrics.get('return_pct', np.nan),
                'sharpe': metrics.get('sharpe', np.nan),
                'max_drawdown_pct': metrics.get('max_drawdown_pct', np.nan),
                'win_rate': metrics.get('win_rate', np.nan),
                'profit_factor': metrics.get('profit_factor', np.nan),
                'avg_win': metrics.get('avg_win', np.nan),
                'avg_loss': metrics.get('avg_loss', np.nan),
                'trade_count': metrics.get('trade_count', 0),
            }
            header = list(row.keys())
            exists = os.path.exists(self.p.metrics_path)
            with open(self.p.metrics_path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=header)
                if not exists:
                    w.writeheader()
                w.writerow(row)
            pass

    def log_info(self, msg: str):
        ts = pd.Timestamp(bt.num2date(self.data.datetime[0])) if len(
            self.data) > 0 else pd.Timestamp.now()
        log_line = f"[{ts}] {msg}"
        # 写入文件
        if hasattr(self, "debug_log_fp") and self.debug_log_fp:
            self.debug_log_fp.write(log_line + "\n")
            self.debug_log_fp.flush()

    def record_data(self, current_poc: float):
        # 记录指标数据到 indicator_data（统一）
        current_time = pd.Timestamp(bt.num2date(self.data.datetime[0]))
        indicator_row = {
            'datetime':
            current_time,
            'close':
            float(self.data.close[0]),
            # 'zigzag_pivot': float(self.zz.pivot[0]) if not math.isnan(self.zz.pivot[0]) else None,
            # 'zigzag_sup': float(self.zz.get_recent_lows(1)[0]),
            # 'zigzag_res': float(self.zz.get_recent_highs(1)[0]),
            'poc':
            current_poc,
            'swing_sup':
            float(self.swing.sup[0])
            if not math.isnan(self.swing.sup[0]) else None,
            'swing_res':
            float(self.swing.res[0])
            if not math.isnan(self.swing.res[0]) else None,
            'recent_breakouts':
            [(b['type'], b['level'], b.get('confirm_count', 0))
             for b in self._recent_breakouts],
            # 可按需扩展更多字段
            'volume':
            float(self.data.volume[0]),
            'vwap':
            float(self.data.vwap[0])
            if not math.isnan(self.data.vwap[0]) else None,
            'rolling_cum_delta':
            float(
                getattr(self.data, 'rolling_cum_delta', [np.nan])[0]
                or np.nan),
        }
        # self.log_info(f"Indicator Data: {indicator_row}")
        self.indicator_data.append(indicator_row)

    def stop(self):
        try:
            self._export_on_stop()
        except Exception as e:
            self.log_info(f"Error exporting on stop: {e}")
