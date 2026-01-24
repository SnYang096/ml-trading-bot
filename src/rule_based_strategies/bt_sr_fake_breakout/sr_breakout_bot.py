"""
Backtrader PoC Breakout / False-Breakout Bot (BTC + OPT + MonteCarlo + Candidate Selector)
- Trades-level ingestion → OHLCV + VWAP + Δ + CumDelta
- Rolling Volume Profile → PoC (Point of Control)
- Swing 高低点 + ZigZag 关键位用于突破/假突破判断
- 低波动压缩区 (Boll 带宽 / ATR) 边界“突破”优先按假突破处理（结合 VWAP / CumDelta / 大影线）
- 绘图：mplfinance K线 + PoC + 买卖点；单窗横向 Volume Profile
- BTC 参数网格搜索（Backtrader optstrategy），并将结果汇总到 CSV
- 蒙特卡洛健壮性测试（基于交易盈亏序列的自助重采样 / 随机重排）
- 从优化结果自动筛选 Top-K 候选参数并保存为候选文件

Author: you
Dependencies: backtrader, pandas, numpy, mplfinance, matplotlib
"""
from __future__ import annotations

import os
import math
import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt
from btplotting import BacktraderPlotting

from .base_strategy import CFG, PandasDataExtra, BaseStrategy

from backtrader_project.services.common.helper import plot_candles_with_poc_and_trades_bokeh
from backtrader_project.services.common.helper import compute_metrics
from backtrader_project.services.common.data_loader import build_features, load_trade_data
from backtrader_project.services.common.oflow_features import oflow_features
from ..live.volatility_sizer import VolatilitySizer
from ..indicators.swing_levels import SwingLevels
from ..indicators.volume_profile import VolumeProfilePOC
from ..indicators.zigzag import ZigZag

# CVD = cum_delta（累计的买卖差）

# rolling_cum_delta = 短期累计（更适合突破/反转检测）

# VWAP = 成交量加权均价 → 机构常用执行/判断资金流向

# TWAP = 时间加权均价 → 用来均匀拆分大单执行
# =============================
# ----- SRBreakoutStrategy ----
# =============================


# TODO：
# 1. 关闭仓位判断类型
# 2. fake的也需要3个k线确认
# 3. STRUCTURE STOP SHORT 止损好像非常近， -》去掉了zigzag
class SRBreakoutStrategy(BaseStrategy):
    params = dict(
        # Profile + basic
        profile_bars=CFG.profile_bars,
        profile_alpha=CFG.profile_alpha,
        poc_every_n_bars=CFG.poc_every_n_bars,
        confirm_delta=0.0,
        confirm_k=3,
        vwap_bias=True,
        stop_atr=1.5,
        take_atr=3.0,
        # swing/zigzag params
        swing_lookback=120,
        swing_window=3,
        # zigzag_atr_period=14,
        # zigzag_atr_mult=0.1,
        # compression (Boll bandwidth / ATR)
        bb_period=20,
        bb_dev=2.0,
        compression_threshold=0.5,
        wick_ratio_threshold=2.0,
        fake_lookahead=10,
        # —— 新增：多维确认 ——
        confirm_vol_ma=20,
        vol_spike_mult=1.5,
        atr_delta_period=10,
        atr_delta_min=0.005,
        # order-flow related (覆盖 BaseStrategy 的 defaults)
        use_orderflow=True,
        cumdelta_zscore_thresh=0.8,
        taker_ratio_thresh=0.55,
        # risk & sizing
        risk_per_trade_pct=0.01,
        risk_per_trade_min=0.003,
        risk_per_trade_max=0.02,
        # tranches
        max_tranches=5,
        first_tranche_pct=0.5,
        add_on_retest=True,
        retest_tol_atr=0.2,
        add_on_momentum=False,
        add_trigger_atr=1.0,
        # structure stop and tp
        use_structure_stop=True,
        tp1_rr=1.5,
        tp1_pct=0.5,
        trailing_stop_atr_mult=2.0,
        breakeven_atr_mult=1.0,
        pnl_window=30,
        risk_step=0.002,
        min_pf_for_risk_up=1.5,
        time_stop_bars=5,
        zig_stop_num=2,

        # orderflow tuning (默认值参考建议)
        orderflow_weights=dict(
            vol_spike=1.0,
            cumdelta=1.5,
            taker=1.0,
            vwap=0.8,
            atr_delta=1.0,
        ),
        orderflow_window=200,
        orderflow_percentile=0.65,
        orderflow_min_threshold=0.4,
        atr_range_filter=0.2,  # ATR 范围过滤器（用于假突破检测
    )

    def __init__(self):
        # 初始化 BaseStrategy
        super().__init__()

        d = self.data
        # indicators
        self.atr = bt.ind.ATR(d, period=14)
        self.vol_ma = bt.ind.SMA(d.volume, period=self.p.confirm_vol_ma)
        self.bb = bt.ind.BollingerBands(d.close,
                                        period=self.p.bb_period,
                                        devfactor=self.p.bb_dev)

        self.bar_counter = 0
        self._recent_breakouts: List[Dict[str, Any]] = []
        self._risk_pct_runtime = self.p.risk_per_trade_pct
        self._recent_closed_pnls: List[float] = []

        # position/order state
        self.entry_price = 0.0
        self.entry_bar = None
        self.tranche_filled = 0
        # Note: 这些变量导致一次只能开一个单子，多个单子需要改成 list/dict，每次都会被最后一个覆盖
        self.init_risk_per_unit = 0.0
        self.protective_stop = None
        self.tp1_done = False

        # indicators you already used (assume these indicator classes exist in your project)
        self.swing = SwingLevels(d,
                                 lookback=self.p.swing_lookback,
                                 window=self.p.swing_window)
        # self.zz = ZigZag(d, atr_period=self.p.zigzag_atr_period, atr_mult=self.p.zigzag_atr_mult)
        self.poc = VolumeProfilePOC(d,
                                    bars=self.p.profile_bars,
                                    alpha=self.p.profile_alpha)

    # orderflow/multi confirm helper ------------------------------------------------
    def _orderflow_confirm(self, long: bool) -> bool:
        """综合订单流/成交量/price-vwap 来给出是否允许趋势方向入场（加权 + 动态阈值）"""
        # 1) 计算每个维度是否触发，并把它们转换为分数（0 或 1 或 介于 0..1）
        weights: Dict[str, float] = dict(self.p.orderflow_weights)
        # volume spike (0/1)
        vol_val = float(self.data.volume[0]) if not math.isnan(
            self.data.volume[0]) else 0.0
        vol_ma = float(self.vol_ma[0] or 1.0)
        vol_spike_score = 1.0 if vol_val > self.p.vol_spike_mult * vol_ma else 0.0

        # rolling cum_delta zscore -> 转为 0..1（sigmoid 或线性映射）
        cumdelta = float(
            getattr(self.data, 'rolling_cum_delta', [0])[0] or 0.0)
        self._cumdelta_hist.append(cumdelta)
        hist = np.array(
            self._cumdelta_hist[-self._cumdelta_ma_period:]) if len(
                self._cumdelta_hist) > 0 else np.array([])
        if hist.size >= 5:
            mu = hist.mean()
            sd = hist.std(ddof=0) if hist.std(ddof=0) > 1e-12 else 1e-12
            z = (cumdelta - mu) / sd
        else:
            z = 0.0
        # 使用 zscore 映射到 0..1（这里用 tanh/sigmoid 映射更平滑）
        # 对于多头，z 越大越好；空头则反之
        if long:
            cum_score_raw = max(-5.0, min(5.0, z))  # 截断
        else:
            cum_score_raw = max(-5.0, min(5.0, -z))
        # 用 sigmoid 映射到 0..1
        cumdelta_score = 1.0 / (1.0 + math.exp(-cum_score_raw))

        # taker ratio (0..1)
        taker_ratio = float(
            getattr(self.data, 'taker_buy_ratio', [np.nan])[0] or np.nan)
        if math.isnan(taker_ratio):
            taker_score = 0.5  # 中性（信息缺失）
        else:
            # 如果是多头，越接近 1 越好；空头则越接近 0 越好
            if long:
                taker_score = (taker_ratio -
                               (1.0 - self.p.taker_ratio_thresh)) / (
                                   1.0 - (1.0 - self.p.taker_ratio_thresh))
            else:
                # 空头期望 taker_buy_ratio 小
                inv = 1.0 - taker_ratio
                taker_score = (inv - (1.0 - self.p.taker_ratio_thresh)) / (
                    1.0 - (1.0 - self.p.taker_ratio_thresh))
            taker_score = max(0.0, min(1.0, taker_score))

        # vwap bias (0 or 1)
        price = float(self.data.close[0])
        vwap = float(
            self.data.vwap[0]) if not math.isnan(self.data.vwap[0]) else price
        vwap_score = 1.0 if ((price >= vwap) if long else
                             (price <= vwap)) else 0.0

        # atr delta momentum (0..1)
        atr_now = float(self.atr[0])
        atr_score = 0.5
        try:
            past_len = min(int(self.p.atr_delta_period), len(self.data))
            past_atrs = [
                float(self.atr[-i]) for i in range(1, max(1, past_len))
            ]
            if len(past_atrs) >= max(3, int(self.p.atr_delta_period / 2)):
                past_mean = np.mean(past_atrs)
                if past_mean > 0:
                    atr_delta = (atr_now - past_mean) / past_mean
                    # 对多空分别映射为 0..1
                    if long:
                        atr_score = 1.0 / (1.0 + math.exp(
                            -(atr_delta - self.p.atr_delta_min) * 10.0))
                    else:
                        atr_score = 1.0 / (1.0 + math.exp(
                            -(-atr_delta - self.p.atr_delta_min) * 10.0))
                    atr_score = max(0.0, min(1.0, atr_score))
            else:
                atr_score = 0.5
        except Exception:
            atr_score = 0.5

        # 2) 根据权重计算归一化分数（0..1）
        # 每项分数乘以权重后，除以权重总和
        score_components = {
            'vol_spike': float(vol_spike_score),
            'cumdelta': float(cumdelta_score),
            'taker': float(taker_score),
            'vwap': float(vwap_score),
            'atr_delta': float(atr_score),
        }
        total_weight = sum(
            weights.get(k, 1.0) for k in score_components.keys())
        weighted = sum(score_components[k] * weights.get(k, 1.0)
                       for k in score_components.keys())
        normalized_score = weighted / (total_weight
                                       if total_weight > 0 else 1.0)

        # 记录历史分数（用于动态阈值）
        self._orderflow_score_hist.append(normalized_score)

        # 3) 计算动态阈值并比较
        threshold = self._orderflow_compute_threshold()
        # 如果历史数据不足时，也可对单次 score 设置绝对最小阈值（避免过于宽松）
        threshold = max(threshold, float(self.p.orderflow_min_threshold))

        self.log_info(
            f"[OFLOW] bar={len(self.data)} long={long} score={normalized_score:.3f} "
            f"thr={threshold:.3f} comps={score_components}")

        return normalized_score >= threshold

    # ==== 通知/下单/交易记录 ====
    def notify_order(self, order):
        # order.status 的枚举值对应 order.Submitted=1, Accepted=2, Partial=3, Completed=4, Canceled=5, Expired=6, Margin=7, Rejected=8。
        # 7 = Margin，意思是 保证金不足，不能开仓。
        if order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log_info(
                f"Order failed: {order.info.get('tag', '')} {order.size} @ {order.price}, status: {order.status}"
            )
            # 可记录或打印
            used_margin = self.broker.get_value() - self.broker.get_cash()
            context = {
                "time":
                pd.Timestamp(bt.num2date(self.data.datetime[0])),
                "direction":
                "buy" if order.isbuy() else "sell",
                "price":
                order.price,
                "size": (order.size),
                "stop_price":
                0,
                "risk_per_unit":
                0,
                "atr":
                self.atr[0],
                "vwap":
                getattr(self.data, "vwap", [np.nan])[0],
                "rolling_cum_delta":
                getattr(self.data, "rolling_cum_delta", [np.nan])[0],
                "taker_ratio":
                getattr(self.data, "taker_buy_ratio", [np.nan])[0],
                "volume":
                float(self.data.volume[0]),
                "orderflow_score":
                self._orderflow_score_hist[-1]
                if self._orderflow_score_hist else None,
                "orderflow_threshold":
                self._orderflow_compute_threshold(),
                "tag":
                f"Order {order.ref} failed: {order.status}",
            }
            self.log_info(
                f"[Margin不足] 已用保证金: {used_margin:.2f}, 可用资金: {self.broker.get_cash():.2f}, context: {context}"
            )

            self.cancel(order)
            pass

    def notify_trade(self, trade):
        if trade.isopen and trade.justopened:
            c = {
                'time': pd.Timestamp(trade.open_datetime()),
                'type': "buy" if trade.long else "sell",
                'price': trade.price,
                'size': trade.size,
                'pnl': None,
                'pnlcomm': None,
                'trade_ref': trade.ref,
            }
            self.trades_log.append(c)
            self.log_info(
                f"Trade opened: {trade.size} @ {trade.price}, ref: {trade.ref}, context: {c}"
            )

        if trade.isclosed:
            c = {
                'time': pd.Timestamp(bt.num2date(self.data.datetime[0])),
                'type': 'closed',
                'price': float(self.data.close[0]),
                'size': trade.size,
                'pnl': float(trade.pnl),
                'pnlcomm': float(trade.pnlcomm),
                'trade_ref': trade.ref,
            }
            self.trades_log.append(c)
            self.log_info(
                f"Trade closed: {trade.size} @ {trade.price}, ref: {trade.ref}, context: {c}"
            )

            self._recent_closed_pnls.append(float(trade.pnl))
            self._recent_closed_pnls = self._recent_closed_pnls[-self.p.
                                                                pnl_window:]
            # 风险调整（保留原实现）
            self._pnl_risk_adjust()
            # reset
            self.entry_price = 0.0
            self.entry_bar = None
            self.tranche_filled = 0
            self.init_risk_per_unit = 0.0
            self.tp1_done = False
            self.protective_stop = None

    # 简单的 pnl 风险调整（保留原逻辑）
    def _pnl_risk_adjust(self):
        # 基于最近 pnl 来调整 self._risk_pct_runtime（示例算法，可扩展）
        if len(self._recent_closed_pnls) < 5:
            return
        pf = 1.0
        wins = [p for p in self._recent_closed_pnls if p > 0]
        losses = [p for p in self._recent_closed_pnls if p < 0]
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = -np.mean(losses) if losses else 0.0
        if avg_loss > 0:
            pf = (avg_win / avg_loss) if avg_loss > 0 else 1.0
        # 调整规则：pf 高时升风险（最多 p.risk_per_trade_max），低时降风险
        if pf >= self.p.min_pf_for_risk_up:
            self._risk_pct_runtime = min(
                self.p.risk_per_trade_max,
                self._risk_pct_runtime + self.p.risk_step)
        elif pf <= 1.0:
            self._risk_pct_runtime = max(
                self.p.risk_per_trade_min,
                self._risk_pct_runtime - self.p.risk_step)

    # ==== next 主循环 ====
    def next(self):
        self._equity_curve.append(self.broker.getvalue())
        self.bar_counter += 1
        i = len(self.data) - 1

        # POC 每隔 n bars 更新（直接读取 indicator）
        current_poc = float(self.poc.poc[0])
        # 获取支撑阻力（swing/zigzag fallback 使用最近 high/low）
        last_res, last_sup, last_zz = self.find_sr()

        price = float(self.data.close[0])
        vwap = float(
            self.data.vwap[0]) if not math.isnan(self.data.vwap[0]) else price
        atr = float(self.atr[0])

        # candidate detection: 当 price 超过任一关键位（res/sup/zz/poc）视作 "突破候选"
        break_level, candidate_long_break, candidate_short_break = self.find_candidate_break(
            current_poc, last_res, last_sup, last_zz, price)

        self.calc_break(atr, break_level, candidate_long_break,
                        candidate_short_break, i, last_res, last_sup, price)

        self.execute_position_management_strategies(atr, i, last_res, last_sup,
                                                    price, vwap)

        self.record_data(current_poc)
        # 将 candidate 推入待确认列表（避免重复）

    def execute_position_management_strategies(self, atr: float, i: int,
                                               last_res: float | None,
                                               last_sup: float | None,
                                               price: float, vwap: float):
        # "解释一下下面代码：def execute..."Click to see Yuanbao's response https://yuanbao.tencent.com/bot/app/share/chat/TeG6YZcsk8Mc
        # 仓内操作（回踩补仓 / 时间止损 / 保本/分级止盈/移动止损）
        if self.position.size == 0:
            pass
        else:
            # 回踩加仓
            if self.tranche_filled < self.p.max_tranches and self.p.add_on_retest and self.entry_price > 0:
                tol = self.p.retest_tol_atr * atr
                if self.position.size > 0:
                    if abs(price -
                           max(last_res if last_res is not None else vwap,
                               vwap)) <= tol and self._orderflow_confirm(True):
                        self.log_info(
                            f"[RETEST ADD] bar={i} price={price:.2f} last_res={last_res} vwap={vwap:.2f} tol={tol:.2f}"
                        )
                        self._add_tranche(long=True,
                                          atr=atr,
                                          last_sup=last_sup,
                                          last_res=last_res)
                else:
                    if abs(price -
                           min(last_sup if last_sup is not None else vwap, vwap
                               )) <= tol and self._orderflow_confirm(False):
                        self.log_info(
                            f"[RETEST ADD] bar={i} price={price:.2f} last_sup={last_sup} vwap={vwap:.2f} tol={tol:.2f}"
                        )
                        self._add_tranche(long=False,
                                          atr=atr,
                                          last_sup=last_sup,
                                          last_res=last_res)

            # 时间止损
            if self.p.time_stop_bars and self.entry_bar is not None:
                holding_bars = (i - self.entry_bar)
                if holding_bars >= self.p.time_stop_bars:
                    # 计算当前未实现盈亏（多头为 price - entry_price，空头为 entry_price - price）
                    if self.entry_price and self.position.size != 0:
                        unrealized = (
                            price -
                            self.entry_price) if self.position.size > 0 else (
                                self.entry_price - price)
                    else:
                        unrealized = 0.0
                    # 如果没有盈利（未实现盈亏 <= 0），则关闭仓位
                    if unrealized <= 0.0:
                        self.log_info(
                            f"[TIME STOP / NO PROFIT CLOSE] bar={i} holding_bars={holding_bars} max={self.p.time_stop_bars} unrealized={unrealized:.4f}"
                        )
                        self.close()
                    else:
                        self.log_info(
                            f"[TIME STOP] bar={i} holding_bars={holding_bars} unrealized={unrealized:.4f} - keep position"
                        )

            # 多单失败止损逻辑
            # if self.p.use_structure_stop and self.position.size > 0:
            #     recent_highs = self.zz.get_recent_highs(self.p.zig_stop_num)
            #     if recent_highs and all(h < self.entry_price for h in recent_highs):
            #         self.log_info(f"[STRUCTURE STOP LONG] bar={i} recent_highs={recent_highs} entry={self.entry_price}")
            #         self.close()  # 平多单

            # 空单失败止损逻辑
            # if self.p.use_structure_stop and self.position.size < 0:
            #     recent_lows = self.zz.get_recent_lows(self.p.zig_stop_num)
            #     if recent_lows and all(l > self.entry_price for l in recent_lows):
            #         self.log_info(f"[STRUCTURE STOP SHORT] bar={i} recent_lows={recent_lows} entry={self.entry_price}")
            #         self.close()  # 平空单

        # 保本 / 分级止盈 / trailing stop
        if self.position.size != 0 and self.entry_price > 0:
            atr_val = atr
            direction = 1 if self.position.size > 0 else -1
            profit = direction * (price - self.entry_price)
            if profit >= self.p.breakeven_atr_mult * atr_val:
                self.log_info(
                    f"[BREAKEVEN STOP] bar={i} price={price:.2f} entry={self.entry_price:.2f} profit={profit:.2f}"
                )
                self._place_or_move_stop(self.entry_price)
            r_multiple = profit / (self.init_risk_per_unit
                                   if self.init_risk_per_unit > 0 else 1e-9)
            if (not self.tp1_done) and (r_multiple >= self.p.tp1_rr):
                tp1_size = int(abs(self.position.size) * self.p.tp1_pct)
                if tp1_size >= CFG.min_trade_size:
                    if self.position.size > 0:
                        self.log_info(
                            f"[TP1 HIT Sell] bar={i} price={price:.2f} entry={self.entry_price:.2f} profit={profit:.2f} r={r_multiple:.2f} sell {tp1_size}"
                        )
                        self.sell(size=tp1_size)
                    else:
                        self.log_info(
                            f"[TP1 HIT Buy] bar={i} price={price:.2f} entry={self.entry_price:.2f} profit={profit:.2f} r={r_multiple:.2f} buy {tp1_size}"
                        )
                        self.buy(size=tp1_size)
                    self.tp1_done = True

            if self.tp1_done and self.position.size != 0:
                if self.position.size > 0:
                    new_sl = price - self.p.trailing_stop_atr_mult * atr_val
                else:
                    new_sl = price + self.p.trailing_stop_atr_mult * atr_val
                self.log_info(
                    f"[TRAILING STOP MOVE] bar={i} price={price:.2f} entry={self.entry_price:.2f} profit={profit:.2f} r={r_multiple:.2f} new_sl={new_sl:.2f}"
                )
                self._place_or_move_stop(new_sl)

    def calc_break(self, atr: float, break_level: bool,
                   candidate_long_break: float, candidate_short_break: bool,
                   i: int, last_res: float | None, last_sup: float | None,
                   price: float):

        # 如果已有持仓，则不再寻找新的候选（但仍可以让已有候选完成判定）
        has_position = self.position.size != 0

        # 只保留单一候选（队列长度最多为1）
        # 只有在当前无候选且无持仓时去记录新的候选
        if not self._recent_breakouts and not has_position:
            if candidate_long_break and break_level is not None:
                self._push_candidate('up', break_level, i, price)
            elif candidate_short_break and break_level is not None:
                self._push_candidate('down', break_level, i, price)

        # 标记可能的 wick fake 线索（仅对最后一个候选标记）
        if self._recent_breakouts:
            self.wick_fake_break_check_for_last_candidate(break_level, i)

        # 处理（仅）当前候选：确认/标记为 fake，并在判定后移除（无论是否开仓）
        if not self._recent_breakouts:
            return

        b = self._recent_breakouts[-1]
        is_long = (b['type'] == 'up')
        level = b['level']

        # 初始化计数器
        b['confirm_count'] = b.get('confirm_count', 0)
        b['fail_count'] = b.get('fail_count', 0)
        # 如果存在直接被其它函数（如 wick 检测）设置的 fake 标记，将其转为“假突破提示（stage）”，
        # 并通过计数等待 confirm_k 根 K 线后真正判定为 fake。
        if b.get('fake') and not b.get('fake_stage'):
            b['fake_stage'] = True
            b['fake_stage_count'] = 1
            b['fake_initial_bar'] = b.get('fake_bar', i)
            # 清除即时 fake 标记，等待阶段性确认
            b.pop('fake', None)

        # 检查是否继续满足突破条件 → 累计确认计数；否则增加失败计数（等待 confirm_k 根 K 确认 fake）
        if is_long:
            if price > level:
                b['confirm_count'] += 1
                # 若有 fake_stage，则确认价位被维护，取消 fake_stage
                if b.get('fake_stage'):
                    b.pop('fake_stage', None)
                    b.pop('fake_stage_count', None)
                    b.pop('fake_initial_bar', None)
                b['fail_count'] = 0
                self.log_info(
                    f"[BREAK CHECK CONFIRM] bar={i} long={is_long} level={level:.2f} price={price:.2f} cnt={b['confirm_count']}"
                )
            else:
                b['fail_count'] += 1
                # 若处于 fake_stage，则推进其计数；否则如果 wick 已提示则启动 fake_stage
                if b.get('fake_stage'):
                    b['fake_stage_count'] = b.get('fake_stage_count', 0) + 1
                    self.log_info(
                        f"[BREAK CHECK FAIL (stage)] bar={i} long={is_long} level={level:.2f} price={price:.2f} fail_cnt={b['fail_count']} fake_stage_cnt={b['fake_stage_count']}"
                    )
                else:
                    self.log_info(
                        f"[BREAK CHECK FAIL] bar={i} long={is_long} level={level:.2f} price={price:.2f} fail_cnt={b['fail_count']}"
                    )
        else:
            if price < level:
                b['confirm_count'] += 1
                if b.get('fake_stage'):
                    b.pop('fake_stage', None)
                    b.pop('fake_stage_count', None)
                    b.pop('fake_initial_bar', None)
                b['fail_count'] = 0
                self.log_info(
                    f"[BREAK CHECK CONFIRM] bar={i} long={is_long} level={level:.2f} price={price:.2f} cnt={b['confirm_count']}"
                )
            else:
                b['fail_count'] += 1
                if b.get('fake_stage'):
                    b['fake_stage_count'] = b.get('fake_stage_count', 0) + 1
                    self.log_info(
                        f"[BREAK CHECK FAIL (stage)] bar={i} long={is_long} level={level:.2f} price={price:.2f} fail_cnt={b['fail_count']} fake_stage_cnt={b['fake_stage_count']}"
                    )
                else:
                    self.log_info(
                        f"[BREAK CHECK FAIL] bar={i} long={is_long} level={level:.2f} price={price:.2f} fail_cnt={b['fail_count']}"
                    )

        resolved = False

        # 如果达到确认阈值 -> 判定为真实突破（confirmed）
        if b.get('confirm_count', 0) >= max(1, self.p.confirm_k):
            allow = True
            if self.p.use_orderflow:
                allow = self._orderflow_confirm(long=is_long)
            # 只有在当前无持仓时才开仓；无论是否开仓，候选都被视为已决断并移除
            if allow and not has_position:
                tag = 'confirmed_break_long' if is_long else 'confirmed_break_short'
                if is_long:
                    self._enter_long(atr, last_sup, last_res, tag=tag)
                else:
                    self._enter_short(atr, last_sup, last_res, tag=tag)
                self.log_info(
                    f"[BREAK CONFIRMED & ENTER] bar={i} long={is_long} level={level:.2f} tag={tag}"
                )
            else:
                self.log_info(
                    f"[BREAK CONFIRMED BUT SKIPPED ENTER] bar={i} long={is_long} level={level:.2f} allow={allow} has_pos={has_position}"
                )
            resolved = True

        # fake 判定：将立即 fake 标记改为阶段性确认，只有在 fail_count 或 fake_stage_count 达到 confirm_k 时才视为真正的 fake
        fake_confirmed = False
        if b.get('fake'):  # 兼容旧逻辑（极少数情况下）
            fake_confirmed = True
        else:
            # 如果存在 fake_stage 且其计数达阈值 -> 确认 fake
            if b.get('fake_stage') and b.get('fake_stage_count', 0) >= max(
                    1, self.p.confirm_k):
                fake_confirmed = True
            # 或者连续 fail 达到阈值，也视为 fake
            if b.get('fail_count', 0) >= max(1, self.p.confirm_k):
                fake_confirmed = True

        if fake_confirmed:
            # 标记为真正的 fake（供后续记录/清理），并使用当前 bar 作为 fake_bar
            b['fake'] = True
            b['fake_bar'] = i
            # 仅在没有持仓的情况下允许反向开仓；若有持仓则不作新的寻找/开平
            allow_rev = True
            if self.p.use_orderflow:
                allow_rev = self._orderflow_confirm(long=not is_long)
            if not has_position and allow_rev and (
                    i - b['bar']) <= self.p.fake_lookahead:
                tag = 'fake_up_break_reverse_short' if is_long else 'fake_down_break_reverse_long'
                if is_long:
                    self.log_info(
                        f"[FAKE -> REVERSE ENTER SHORT] bar={i} level={level:.2f} tag={tag}"
                    )
                    self._enter_short(atr, last_sup, last_res, tag=tag)
                else:
                    self.log_info(
                        f"[FAKE -> REVERSE ENTER LONG] bar={i} level={level:.2f} tag={tag}"
                    )
                    self._enter_long(atr, last_sup, last_res, tag=tag)
            else:
                self.log_info(
                    f"[FAKE -> NO REVERSE] bar={i} level={level:.2f} has_pos={has_position} allow_rev={allow_rev}"
                )
            resolved = True

        # 超出 lookahead 窗口仍未确认 -> 清理候选，准备寻找下一个
        if not resolved and (i - b['bar']) > self.p.fake_lookahead:
            self.log_info(
                f"[CANDIDATE EXPIRED] bar={i} cand_bar={b['bar']} level={level:.2f}"
            )
            resolved = True

        # 一旦判定（confirmed/fake/expired），移除候选以便下一个候选可以被记录
        if resolved:
            self._recent_breakouts.clear()

    def wick_fake_break_check_for_last_candidate(self,
                                                 break_level: bool | None,
                                                 i: int):
        # wick_fake_break_check_for_last_candidate() 里影线（wick）的判断，是不是多余的，能不能删掉让 calc_break() 更简洁？
        # 我帮你从逻辑上拆开看一下：
        # 1. 现有逻辑是什么？
        # 核心突破判定在 calc_break() →
        # 用 price/level 比较 + confirm_count / fail_count 来判断是真突破还是假突破。
        # wick 假突破检测在 wick_fake_break_check_for_last_candidate() →
        # 它提前用大影线 + 低成交量来给最后一个候选打个 fake 提示（不是立即 fake，而是 fake_stage → 等 confirm_k 根 K 线再确认）。
        # 所以 wick 检测的作用是：
        # 给候选的“假突破可能性”提供额外的先验提示，加快 fake_stage 的触发。

        # 2. 如果去掉 wick 检测会怎样？
        # calc_break() 里仍然有 fail_count 达到 confirm_k 的假突破逻辑，所以即使删掉 wick 逻辑，程序还是会最终判定 fake，只是速度变慢（需要价格连续“失败”才判定）。
        # wick_fake_break_check_for_last_candidate() 的作用是：
        # 在 volume 低 & 大影线条件下，更早介入 fake 判断。
        # 在弱量能的突破下“加强过滤”，减少误判确认为真突破。
        # 去掉 wick 检测，不会影响代码主流程的正确性，但是会牺牲一部分“假突破提前识别”的能力。
        try:
            wick_up_big, wick_lo_big = self.find_pre_wick()
        except Exception:
            wick_up_big, wick_lo_big = False, False

        # 将大影线线索标记到最近加入的 candidate（如果匹配）上，后续 loop 会处理 fake 行为（反向入场）
        if break_level is not None and self._recent_breakouts:
            # 可能刚刚 push 的 candidate 在列表末尾，检查并标记
            last = self._recent_breakouts[-1]
            # 确保与当前 level/direction 匹配，避免误标记旧的 candidate
            atr_now = float(self.atr[0]) if len(self.atr) > 0 else 1e-9
            if abs(last.get('level', 0) -
                   break_level) <= self.p.atr_range_filter * atr_now:
                # 例如： 若量能不足且前一根又有大影线，倾向于标记假突破
                try:
                    vol = float(self.data.volume[0] or 0.0)
                    vol_ma = float(self.vol_ma[0] or 1.0)
                    low_vol = vol < 0.8 * vol_ma  # 简单阈值，可调
                    if break_level is not None and self._recent_breakouts:
                        last = self._recent_breakouts[-1]
                    if last.get('fake') is not True and low_vol:
                        # 若量能不足且前一根又有大影线，倾向于标记假突破
                        if (last['type'] == 'up'
                                and wick_up_big) or (last['type'] == 'down'
                                                     and wick_lo_big):
                            last['fake'] = True
                            last['fake_bar'] = i
                            self.log_info(
                                f"[LOW VOL + WICK -> MARK FAKE] bar={i} dir={last['type']} level={break_level:.2f} vol={vol:.0f} ma={vol_ma:.0f}"
                            )
                except Exception:
                    pass

    def _push_candidate(self, direction, level, index, price):
        for b in self._recent_breakouts:
            # 这段代码中的条件判断语句主要用于​​避免将重复的价格突破点加入待确认列表​​，其核心逻辑是通过比较当前突破点与已有突破点的方向和水平值，过滤掉高度相似的记录
            atr_now = float(self.atr[0]) if len(self.atr) > 0 else 1e-9
            if b['type'] == direction and abs(
                    b.get('level', level) -
                    level) <= self.p.atr_range_filter * atr_now:
                self.log_info(
                    f"[DUPLICATE CANDIDATE] bar={index} dir={direction} level={level:.2f} price={price:.2f}"
                )
                return
        self._recent_breakouts.append({
            'bar': index,
            'type': direction,
            'level': level,
            'confirm_count': 0,
            'origin_price': price,
        })
        self.log_info(
            f"[CANDIDATE] bar={index} dir={direction} level={level:.2f} price={price:.2f}"
        )

    def find_candidate_break(self, current_poc: float, last_res: float | None,
                             last_sup: float | None, last_zz: float | None,
                             price: float) -> tuple[bool, float, bool]:
        # 使用穿越(cross)判断：只有当当前价从上一根 K 穿越关键位时才视为 candidate
        price_curr = price
        try:
            price_prev = float(self.data.close[-1])
        except Exception:
            price_prev = price_curr

        def crossed_up(level: float) -> bool:
            return (price_prev <= level) and (price_curr > level)

        def crossed_down(level: float) -> bool:
            return (price_prev >= level) and (price_curr < level)

        break_level = None
        candidate_long_break = False
        candidate_short_break = False

        # 优先级：swing -> zigzag -> poc （long）
        if last_res is not None and break_level is None:
            if crossed_up(last_res):
                candidate_long_break = True
                break_level = last_res
                self.log_info(
                    f"[CANDIDATE LONG CROSS] prev={price_prev:.2f} curr={price_curr:.2f} crossed_up last_res={last_res:.2f}"
                )
        if last_zz is not None and break_level is None:
            if crossed_up(last_zz):
                candidate_long_break = True
                break_level = last_zz
                self.log_info(
                    f"[CANDIDATE LONG CROSS] prev={price_prev:.2f} curr={price_curr:.2f} crossed_up last_zz={last_zz:.2f}"
                )
        if current_poc is not None and break_level is None:
            if crossed_up(current_poc):
                candidate_long_break = True
                break_level = current_poc
                self.log_info(
                    f"[CANDIDATE LONG CROSS] prev={price_prev:.2f} curr={price_curr:.2f} crossed_up poc={current_poc:.2f}"
                )

        # 优先级：swing -> zigzag -> poc （short）
        if last_sup is not None and break_level is None:
            if crossed_down(last_sup):
                candidate_short_break = True
                break_level = last_sup
                self.log_info(
                    f"[CANDIDATE SHORT CROSS] prev={price_prev:.2f} curr={price_curr:.2f} crossed_down last_sup={last_sup:.2f}"
                )
        if last_zz is not None and break_level is None:
            if crossed_down(last_zz):
                candidate_short_break = True
                break_level = last_zz
                self.log_info(
                    f"[CANDIDATE SHORT CROSS] prev={price_prev:.2f} curr={price_curr:.2f} crossed_down last_zz={last_zz:.2f}"
                )
        if current_poc is not None and break_level is None:
            if crossed_down(current_poc):
                candidate_short_break = True
                break_level = current_poc
                self.log_info(
                    f"[CANDIDATE SHORT CROSS] prev={price_prev:.2f} curr={price_curr:.2f} crossed_down poc={current_poc:.2f}"
                )

        return break_level, candidate_long_break, candidate_short_break

    def find_sr(self) -> tuple[float | None, float | None, float | None]:
        try:
            last_res = float(self.swing.res[0]) if not math.isnan(
                self.swing.res[0]) else None
        except Exception:
            last_res = None
        try:
            last_sup = float(self.swing.sup[0]) if not math.isnan(
                self.swing.sup[0]) else None
        except Exception:
            last_sup = None
        try:
            last_zz = float(
                self.zz.zz[0]) if not math.isnan(self.zz.zz[0]) else None
        except Exception:
            last_zz = None
        return last_res, last_sup, last_zz

    def find_pre_wick(self):
        o, h, l, c = float(self.data.open[-1]), float(
            self.data.high[-1]), float(self.data.low[-1]), float(
                self.data.close[-1])
        body, up_wick, lo_wick = self._wick_info(o, h, l, c)
        wick_up_big = (body <= 1e-9
                       and up_wick > 0) or (up_wick / (body + 1e-9)
                                            >= self.p.wick_ratio_threshold)
        wick_lo_big = (body <= 1e-9
                       and lo_wick > 0) or (lo_wick / (body + 1e-9)
                                            >= self.p.wick_ratio_threshold)
        return wick_up_big, wick_lo_big

    # ==== 下单助手：保留原实现，仅略微保证市价下单后记录 tag ====
    def _enter_long(self,
                    atr: float,
                    last_sup: float,
                    last_res: float,
                    tag: str = ""):
        price = float(self.data.close[0])
        atr_stop = price - self.p.stop_atr * atr
        stop_price = self._structure_stop(True, atr_stop, last_sup, last_res,
                                          price)
        risk_per_unit = price - stop_price
        size_total = self._risk_sized(risk_per_unit)
        if size_total <= 0:
            self.log_info(
                f"[ENTER LONG] bar={len(self.data)} price={price:.2f} stop={stop_price:.2f} risk_per_unit={risk_per_unit:.2f} size={size_total:.4f} tag={tag} (no size)"
            )
            return
        if self.p.max_tranches >= 2:
            # 计算第一批次的数量，并向下取整到 min_qty
            size1 = math.floor(size_total * self.p.first_tranche_pct /
                               self.p.min_qty) * self.p.min_qty
        else:
            # 不分批次时使用总仓位
            size1 = size_total

        # 确保下单数量不小于最小下单单位
        size1 = max(self.p.min_qty, size1)
        self.buy(size=size1)
        self.entry_price = price
        self.entry_bar = len(self.data) - 1
        self.tranche_filled = 1
        self.init_risk_per_unit = risk_per_unit
        self._place_or_move_stop(stop_price)
        c = {
            "reason":
            tag,  # confirmed_break_long / fake_down_break_reverse_long ...
            "atr":
            atr,
            "vwap":
            getattr(self.data, "vwap", [np.nan])[0],
            "poc":
            float(self.poc.poc[0]),
            "last_res":
            last_res,
            "last_sup":
            last_sup,
            "bb_width": (self.bb.top[0] - self.bb.bot[0]) / self.bb.mid[0],
            "vol":
            float(self.data.volume[0]),
            "orderflow_score":
            self._orderflow_score_hist[-1]
            if self._orderflow_score_hist else None,
            "orderflow_threshold":
            self._orderflow_compute_threshold(),
        }
        self.log_info(
            f"[ENTER LONG] bar={len(self.data)} price={price:.2f} stop={stop_price:.2f} risk_per_unit={risk_per_unit:.2f} size={size1:.4f} size_total={size_total:.4f} tag={tag} context: {c}"
        )

    def _enter_short(self,
                     atr: float,
                     last_sup: float,
                     last_res: float,
                     tag: str = ""):
        price = float(self.data.close[0])
        atr_stop = price + self.p.stop_atr * atr
        stop_price = self._structure_stop(False, atr_stop, last_sup, last_res,
                                          price)
        risk_per_unit = stop_price - price
        size_total = self._risk_sized(risk_per_unit)
        if size_total <= 0:
            self.log_info(
                f"[ENTER SHORT] bar={len(self.data)} price={price:.2f} stop={stop_price:.2f} risk_per_unit={risk_per_unit:.2f} size={size_total:.4f} tag={tag} (no size)"
            )
            return
        if self.p.max_tranches >= 2:
            # 计算第一批次的数量，并向下取整到 min_qty
            size1 = math.floor(size_total * self.p.first_tranche_pct /
                               self.p.min_qty) * self.p.min_qty
        else:
            # 不分批次时使用总仓位
            size1 = size_total

        # 确保下单数量不小于最小下单单位
        size1 = max(self.p.min_qty, size1)
        self.sell(size=size1)
        self.entry_price = price
        self.entry_bar = len(self.data) - 1
        self.tranche_filled = 1
        self.init_risk_per_unit = risk_per_unit
        self._place_or_move_stop(stop_price)
        c = {
            "reason":
            tag,  # confirmed_break_long / fake_down_break_reverse_long ...
            "atr":
            atr,
            "vwap":
            getattr(self.data, "vwap", [np.nan])[0],
            "poc":
            float(self.poc.poc[0]),
            "last_res":
            last_res,
            "last_sup":
            last_sup,
            "bb_width": (self.bb.top[0] - self.bb.bot[0]) / self.bb.mid[0],
            "vol":
            float(self.data.volume[0]),
            "orderflow_score":
            self._orderflow_score_hist[-1]
            if self._orderflow_score_hist else None,
            "orderflow_threshold":
            self._orderflow_compute_threshold(),
        }
        self.log_info(
            f"[ENTER SHORT] bar={len(self.data)} price={price:.2f} stop={stop_price:.2f} "
            f"risk_per_unit={risk_per_unit:.2f} size={size1:.4f} size_total={size_total:.4f} tag={tag} context: {c}"
        )

    def _add_tranche(self, long: bool, atr: float, last_sup: float,
                     last_res: float):
        price = float(self.data.close[0])
        current_size = abs(self.position.size)
        if long:
            atr_stop = price - self.p.stop_atr * atr
            stop_price = self._structure_stop(True, atr_stop, last_sup,
                                              last_res, price)
            risk_per_unit = price - stop_price
        else:
            atr_stop = price + self.p.stop_atr * atr
            stop_price = self._structure_stop(False, atr_stop, last_sup,
                                              last_res, price)
            risk_per_unit = stop_price - price
        size_total = self._risk_sized(risk_per_unit)
        # 重新计算 current_size 为 min_qty 的倍数，防止精度问题
        current_size = math.floor(
            abs(self.position.size) / self.p.min_qty) * self.p.min_qty
        add_size = max(0, size_total - current_size)
        # 确保增加的仓位也是 min_qty 的倍数
        add_size = math.floor(add_size / self.p.min_qty) * self.p.min_qty
        if add_size <= 0:
            self.log_info(
                f"[ADD TRANCHE {'LONG' if long else 'SHORT'}] bar={len(self.data)} price={price:.2f} stop={stop_price:.2f} risk_per_unit={risk_per_unit:.2f} current_size={current_size:.4f} add_size={add_size:.4f} total_size={size_total:.4f} (no size)"
            )
            return
        if long:
            self.buy(size=add_size)
            # 在下单后立即记录加仓日志
            self.trades_log.append({
                'time':
                pd.Timestamp(bt.num2date(self.data.datetime[0])),
                'type':
                'add_long',  # <-- 关键修改：明确标记为加仓
                'price':
                self.data.close[0],  # 使用当前收盘价或执行价
                'size':
                add_size,
                'pnl':
                None,
                'pnlcomm':
                None,
                'trade_ref':
                'ADD'  # 或其他唯一标记
            })
        else:
            self.sell(size=add_size)
            self.trades_log.append({
                'time':
                pd.Timestamp(bt.num2date(self.data.datetime[0])),
                'type':
                'add_short',  # <-- 关键修改：明确标记为加仓
                'price':
                self.data.close[0],
                'size':
                add_size,
                'pnl':
                None,
                'pnlcomm':
                None,
                'trade_ref':
                'ADD'
            })
        self.tranche_filled += 1
        self._place_or_move_stop(stop_price)
        c = {
            "time":
            pd.Timestamp(bt.num2date(self.data.datetime[0])),
            "direction":
            "long",
            "risk_per_unit":
            risk_per_unit,
            "atr":
            atr,
            "vwap":
            getattr(self.data, "vwap", [np.nan])[0],
            "rolling_cum_delta":
            getattr(self.data, "rolling_cum_delta", [np.nan])[0],
            "taker_ratio":
            getattr(self.data, "taker_buy_ratio", [np.nan])[0],
            "volume":
            float(self.data.volume[0]),
            "orderflow_score":
            self._orderflow_score_hist[-1]
            if self._orderflow_score_hist else None,
            "orderflow_threshold":
            self._orderflow_compute_threshold(),
            "tag":
            "_add_tranche",
        }
        self.log_info(
            f"[ADD TRANCHE {'LONG' if long else 'SHORT'}] bar={len(self.data)} price={price:.2f} stop={stop_price:.2f} risk_per_unit={risk_per_unit:.2f} add_size={add_size:.4f} total_size={size_total:.4f} context: {c}"
        )


# =============================
# --------- RUN / PLOT --------
# =============================
data_name = "BTC/USDT"


def run_backtest(trades_csv: str,
                 timeframe: str = CFG.timeframe,
                 tz: Optional[str] = CFG.tz,
                 strat_params: Optional[dict] = None):
    CFG.timeframe = timeframe
    CFG.tz = tz

    bars, trade_data = build_features(trades_csv, timeframe)
    print("build_features check:", len(bars))
    print(bars.head(30))
    print(bars.tail(30))
    print("Final features shape:", bars.shape)

    data = PandasDataExtra(dataname=bars)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(CFG.cash)
    cerebro.broker.setcommission(commission=CFG.commission,
                                 mult=1,
                                 leverage=100,
                                 name=data_name)
    cerebro.addsizer(VolatilitySizer,
                     risk_per_trade=0.02,
                     lot_size=0.001,
                     use_volatility=True,
                     atr_period=14)
    cerebro.broker.set_slippage_perc(CFG.slippage)

    if strat_params is None:
        strat_params = {}

    cerebro.adddata(data, name=data_name)
    cerebro.addstrategy(SRBreakoutStrategy, **strat_params)
    results = cerebro.run()
    p = BacktraderPlotting(style='bar')
    cerebro.plot(p)
    strat: SRBreakoutStrategy = results[0]
    return cerebro, strat, bars, trade_data


# =============================
# ------ OPTIMIZATION / MC ----
# =============================


def run_optimization(trades_csv: str, timeframe: str, tz: Optional[str],
                     opt_out: str):
    CFG.timeframe = timeframe
    CFG.tz = tz
    bars = build_features(trades_csv, timeframe)
    data = PandasDataExtra(dataname=bars)

    cerebro = bt.Cerebro(maxcpus=None)
    cerebro.broker.setcash(CFG.cash)
    cerebro.broker.setcommission(CFG.commission)
    cerebro.addsizer(bt.sizers.FixedSize, stake=1)
    cerebro.broker.set_slippage_perc(CFG.slippage)

    grid = dict(
        zigzag_pct=[0.01, 0.02, 0.03],
        swing_lookback=[50, 100, 200],
        swing_window=[2, 3, 5],
        bb_period=[20, 30],
        compression_threshold=[0.4, 0.6],
        confirm_delta=[0.0, 100.0, 300.0],
        stop_atr=[1.0, 1.5],
        take_atr=[2.0, 3.0],
        vol_spike_mult=[1.2, 1.5, 2.0],
        atr_delta_period=[5, 10],
        atr_delta_min=[-0.5, 0, 0.5, 1.0],
        first_tranche_pct=[0.5, 0.6],
        tp1_rr=[1.0, 1.5, 2.0],
        tp1_pct=[0.4, 0.5, 0.6],
        trailing_stop_atr_mult=[1.5, 2.0, 2.5],
    )

    if os.path.exists(opt_out):
        os.remove(opt_out)

    cerebro.adddata(data)
    cerebro.optstrategy(SRBreakoutStrategy, metrics_path=opt_out, **grid)
    cerebro.run(maxcpus=None)
    print(f"Optimization finished. Metrics written to {opt_out}")


def select_top_candidates(opt_csv: str,
                          out_csv: str = 'btc_candidates.csv',
                          top_k: int = 10,
                          max_drawdown_limit: float = None,
                          min_profit_factor: float = None) -> pd.DataFrame:
    if not os.path.exists(opt_csv):
        raise FileNotFoundError(opt_csv)
    df = pd.read_csv(opt_csv)
    for col in [
            'return_pct', 'max_drawdown_pct', 'profit_factor', 'final_value'
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if max_drawdown_limit is not None and 'max_drawdown_pct' in df.columns:
        df = df[df['max_drawdown_pct'] >= max_drawdown_limit]
    if min_profit_factor is not None and 'profit_factor' in df.columns:
        df = df[df['profit_factor'] >= min_profit_factor]
    sort_cols = [
        c for c in ['return_pct', 'profit_factor', 'max_drawdown_pct']
        if c in df.columns
    ]
    ascending = [False, False, True][:len(sort_cols)]
    df_sorted = df.sort_values(by=sort_cols, ascending=ascending).head(top_k)
    df_sorted.to_csv(out_csv, index=False)
    print(f'Selected top {len(df_sorted)} candidates -> {out_csv}')
    return df_sorted


def monte_carlo_from_trades(trades_log: List[Dict],
                            n_iter: int = 2000) -> Dict[str, float]:
    """基于交易日志的 Monte Carlo 模拟（重采样 pnl 求和）
    "def monte_carlo_from..."Click to see Yuanbao's response
https://yuanbao.tencent.com/bot/app/share/chat/GjzxCxO4bCCU
    """
    # 🚩 仅取 closed 交易的 pnl
    pnls: List[float] = []
    for t in trades_log:
        if str(t.get("type",
                     "")).lower() == "closed" and t.get("pnl") is not None:
            try:
                pnls.append(float(t["pnl"]))
            except (TypeError, ValueError):
                continue

    if not pnls:
        return {'mean': np.nan, 'p5': np.nan, 'p95': np.nan}

    results = []
    n = len(pnls)
    for _ in range(n_iter):
        sampled = [random.choice(pnls) for _ in range(n)]
        results.append(sum(sampled))

    arr = np.asarray(results, dtype=float)
    return {
        'mean': float(np.mean(arr)),
        'p5': float(np.percentile(arr, 5)),
        'p95': float(np.percentile(arr, 95)),
    }


# =============================
# ----------- MAIN -----------
# =============================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trades',
        type=str,
        required=True,
        help=
        'Path to trades CSV (timestamp,price,quantity,side/is_buyer_maker optional)'
    )
    parser.add_argument('--timeframe', type=str, default=CFG.timeframe)
    parser.add_argument('--tz', type=str, default=None)
    parser.add_argument('--out', type=str, default='poc_plot.png')
    parser.add_argument('--opt',
                        action='store_true',
                        help='Run parameter grid optimization for BTC')
    parser.add_argument('--opt_out',
                        type=str,
                        default='btc_opt_metrics.csv',
                        help='CSV to store optimization metrics')
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Top K candidate sets to select after optimization')
    parser.add_argument(
        '--max_dd',
        type=float,
        default=None,
        help='Minimum max_drawdown_pct to allow (e.g. -0.2 means -20%%)')
    parser.add_argument('--min_pf',
                        type=float,
                        default=None,
                        help='Minimum profit factor to allow')
    parser.add_argument(
        '--montecarlo',
        action='store_true',
        help='Run Monte Carlo robustness test after single backtest')
    parser.add_argument('--mc_iter', type=int, default=2000)
    args = parser.parse_args()

    data_dir = os.getenv('DATA_DIR', '.')
    trades_full_path = os.path.join(data_dir, args.trades)

    if args.opt:
        run_optimization(trades_full_path, args.timeframe, args.tz,
                         args.opt_out)
        candidates_file = 'btc_candidates.csv'
        select_top_candidates(args.opt_out,
                              out_csv=candidates_file,
                              top_k=args.top_k,
                              max_drawdown_limit=args.max_dd,
                              min_profit_factor=args.min_pf)
    else:
        cerebro, strat, bars, trade_data = run_backtest(
            trades_full_path, args.timeframe, args.tz)
        print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
        print("----- 交易数据时间戳验证 -----")
        print(f"主数据框 (df) 时间范围: {bars.index.min()} 到 {bars.index.max()}")
        plot_candles_with_poc_and_trades_bokeh(trade_data,
                                               bars,
                                               strat,
                                               savepath=args.out)
        print(f'Saved plot to {args.out}')

        metrics = compute_metrics(strat._equity_curve, strat.trades_log)
        print('Metrics:', metrics)

        if args.montecarlo:
            mc = monte_carlo_from_trades(strat.trades_log, n_iter=args.mc_iter)
            print('Monte Carlo (PnL sum) stats:', mc)

# python bt_poc_breakout_bot.py --trades ./btc_trades.csv --timeframe 5T --out poc_plot.png --montecarlo --mc_iter 5000
# python bt_poc_breakout_bot.py --trades ./btc_trades.csv --timeframe 5T --opt --opt_out btc_opt_metrics.csv
# python bt_poc_breakout_bot.py --trades btc_trades.csv --timeframe 5T --opt --opt_out btc_opt_metrics.csv --top_k 10 --max_dd -0.25 --min_pf 1.2
# python bt_poc_breakout_bot.py \
#   --trades btc_trades.csv \
#   --timeframe 5T \
#   --opt \
#   --opt_out btc_opt_metrics.csv \
#   --top_k 15 \
#   --max_dd -0.25 \
#   --min_pf 1.3
