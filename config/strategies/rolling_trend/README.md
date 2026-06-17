# Rolling Trend（趋势滚仓）—— 指标驱动杠杆复利
#
# 入场: 周线EMA200下方 + EMA1200金叉VWAP1200
# 杠杆: 2x起步 → 3x（价格跌20%且当前盈利时升级）
# 卖出: 达5倍后spot式阶梯越长越卖，永不全部清仓
#
# 与 B 系统的区别:
#   - B 系统: per-trade risk-based sizing (1-2% risk/trade)
#   - 滚仓:   portfolio-level leverage maintenance (2-3x target)
#
# 与旧版（TPC信号）的区别:
#   - 旧版: 依赖TPC E13_structural模型信号入场
#   - 新版: 纯指标驱动（EMA200周线 + EMA1200/VWAP1200金叉）
#
# 模拟器: scripts/trend_rolling_simulate.py

strategy:
  name: rolling_trend
  description: |
    趋势滚仓：周线EMA200深熊 + EMA1200金叉VWAP1200入场，
    2x杠杆起步，价格跌20%且盈利→3x，
    5倍后spot式阶梯越长越卖，永不全部清仓。
  owner: quant_team
  archetype: RollingTrendLeverage

  # 信号: 指标驱动（非TPC模型）
  signal_source: indicator
  timeframe: "120T"
  symbol_include: []
  symbol_exclude: []

# 滚仓参数
rolling:
  initial_leverage: 2.0       # 初始杠杆
  max_leverage: 3.0           # 最大杠杆上限
  leverage_step: 1.0          # 每次滚仓加多少倍

  # 入场条件 (AND)
  entry:
    weekly_ema_200_position_lt: 0.0    # 价格在周线EMA200下方
    ema1200_cross_above_vwap1200: true # EMA1200金叉VWAP1200

  # 滚仓触发 (AND条件)
  roll_trigger:
    price_drawdown_from_entry: 0.20  # 从入场价回撤 ≥20%
    require_profitable: true          # 且当前有浮盈

  # 止盈 (spot式阶梯)
  take_profit:
    type: profit_ladder                # 阶梯卖出
    trigger_multiple: 5.0             # entry价格 ×5 触发
    base_daily_sell_fraction: 0.05    # 每次最多卖剩余仓位的5%
    acceleration:
      type: power
      exponent: 0.75
      max_speed_multiplier: 4.0
    never_full_exit: true             # 永不全部清仓

  # 风控
  risk:
    equity_hard_stop: 0.50    # 总equity回撤50% → 全平
    max_symbols: 3            # 最多同时持仓币种数

# 实验记录
experiments:
  phase_1_initial:
    date: 2026-06-17
    description: 指标驱动版初版回测
    results: pending
