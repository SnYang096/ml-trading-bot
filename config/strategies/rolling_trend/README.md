# Rolling Trend（滚仓趋势）—— Portfolio-level Leverage Compounding
#
# 信号源: TPC (Trend Pullback Continuation) — E13_structural 配置
# 执行层: 组合级别杠杆滚仓
#
# 与 B 系统的区别:
#   - B 系统: per-trade risk-based sizing (1-2% risk/trade)
#   - 滚仓:   portfolio-level leverage maintenance (2-5x target)
#
# 账户: 独立 Binance U 本位合约账户

strategy:
  name: rolling_trend
  description: |
    Portfolio-level leveraged trend following using TPC entry signals.
    Starts at 2x leverage, rolls to higher leverage on EMA1200 dips,
    takes profit at 5x/10x entry multiples.
  owner: quant_team
  archetype: RollingTrendLeverage

  # 信号复用 TPC
  signal_source: tpc
  signal_config: E13_structural  # ema1200 structural exit, no trailing
  
  timeframe: "120T"
  symbol_include: []
  symbol_exclude: []

# 滚仓参数
rolling:
  initial_leverage: 2.0       # 初始杠杆 (1-3x)
  max_leverage: 5.0           # 最大杠杆上限
  leverage_step: 1.0          # 每次滚仓加多少倍
  
  # 滚仓触发 (AND条件)
  roll_trigger:
    drawdown_threshold: 0.20  # 从peak回撤 ≥20%
    ema_recovery: ema1200     # 价格回到EMA1200上方
  
  # 止盈
  take_profit:
    target_1: 5.0             # entry价格 ×5 → 减半仓
    target_2: 10.0            # entry价格 ×10 → 全平
  
  # 风控
  risk:
    equity_hard_stop: 0.50    # 总equity回撤50% → 全平
    max_symbols: 3            # 最多同时持仓币种数

# 实验记录
experiments:
  phase_1_bull:
    date: 2026-06-10
    segment: bull_2023_2024
    config: E13_structural
    results:
      spot_1x: "$60k→$221k (3.7x), CAGR 137%, busts 0/6"
      lever_2_3x: "$60k→$358k (6.0x), CAGR 214%, busts 0/6"
      lever_2_5x: "$60k→$599k (10x), CAGR 367%, busts 2/6 near"
    conclusion: "2-3x safest across all symbols"
  
  phase_1_bear:
    date: 2026-06-10
    segment: bear_2022
    results:
      spot_1x: "$60k→$68k (1.1x) — survived bear!"
      lever_2_3x: "$60k→$57k (1.0x)"
    conclusion: "Bear market: spot only, no leverage"
