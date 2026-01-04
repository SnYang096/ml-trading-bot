# 自适应 R/R 方案分析：基于未来波动率的动态止盈止损

## 问题背景

**当前固定 R/R 的问题：**
- BTC 波动较大，固定 2R 止盈、1R 止损不够灵活
- 低波动期：2R 可能太小，容易被噪音触发
- 高波动期：2R 可能太大，需要等待很久
- 不同市场 regime 下，最优 R/R 应该不同

## 方案设计

### 方案 A：基于未来波动率的自适应 R/R（标签生成阶段）

**核心思想：**
- 在标签生成时，使用**未来窗口内的实际波动率**来动态调整止盈止损
- 止盈 = entry ± (未来波动率 × 倍数)
- 止损 = entry ± (未来波动率 × 倍数)

**优点：**
1. ✅ **自适应**：能根据市场实际波动调整
2. ✅ **标签阶段可用**：标签本身就是基于未来结果，使用未来波动率是合理的
3. ✅ **更真实的标签**：反映"在真实波动环境下，策略能否盈利"

**缺点：**
1. ⚠️ **实盘无法使用**：实盘时无法知道未来波动率
2. ⚠️ **需要预测模型**：实盘时需要预测未来波动率

**实现方式：**
```python
def compute_adaptive_rr_label(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    atr_col: str = "atr",
    max_holding_bars: int = 50,
    stop_loss_multiplier: float = 1.0,  # 止损倍数（相对于未来波动率）
    take_profit_multiplier: float = 2.0,  # 止盈倍数（相对于未来波动率）
    volatility_window: int = 10,  # 计算未来波动率的窗口
    use_breakeven_stop: bool = True,  # 是否使用保本止损
) -> pd.Series:
    """
    基于未来波动率的自适应 R/R 标签
    
    逻辑：
    1. 计算未来 volatility_window 内的实际波动率
    2. 使用该波动率动态调整止盈止损
    3. 支持保本止损（当价格达到 stop_loss_multiplier × 未来波动率时，止损上移到保本）
    """
    # 1. 计算未来波动率（标签阶段可以使用未来信息）
    future_volatility = df[price_col].pct_change().rolling(
        window=volatility_window, min_periods=1
    ).std().shift(-volatility_window)  # 使用未来窗口的波动率
    
    # 2. 对于每个信号，使用该时刻的"未来波动率"来设定止盈止损
    # 3. 扫描价格路径，判断是否先触达 TP 或 SL
    # ...
```

### 方案 B：基于历史波动率的自适应 R/R（实盘可用）

**核心思想：**
- 使用**历史波动率**（而非未来波动率）来动态调整止盈止损
- 止盈 = entry ± (历史波动率 × 倍数 × 调整因子)
- 止损 = entry ± (历史波动率 × 倍数 × 调整因子)

**优点：**
1. ✅ **实盘可用**：只使用历史信息，无未来泄露
2. ✅ **自适应**：能根据市场regime调整

**缺点：**
1. ⚠️ **可能不够准确**：历史波动率 ≠ 未来波动率
2. ⚠️ **需要回测验证**：需要验证历史波动率是否能预测未来波动率

**实现方式：**
```python
def compute_adaptive_rr_label_historical(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    atr_col: str = "atr",
    max_holding_bars: int = 50,
    stop_loss_multiplier: float = 1.0,
    take_profit_multiplier: float = 2.0,
    volatility_lookback: int = 20,  # 历史波动率窗口
    volatility_forecast_window: int = 10,  # 预测未来多少期的波动率
    use_breakeven_stop: bool = True,
) -> pd.Series:
    """
    基于历史波动率的自适应 R/R 标签（实盘可用）
    
    逻辑：
    1. 使用历史 volatility_lookback 窗口计算波动率
    2. 使用 GARCH 或其他模型预测未来 volatility_forecast_window 的波动率
    3. 使用预测的波动率来设定止盈止损
    """
    # 1. 计算历史波动率
    historical_vol = df[price_col].pct_change().rolling(
        window=volatility_lookback, min_periods=1
    ).std()
    
    # 2. 预测未来波动率（使用 GARCH 或其他模型）
    # predicted_vol = forecast_volatility(historical_vol, window=volatility_forecast_window)
    
    # 3. 使用预测的波动率来设定止盈止损
    # ...
```

### 方案 C：混合方案（推荐）

**核心思想：**
- **标签生成阶段**：使用方案 A（未来波动率），生成更真实的标签
- **实盘交易阶段**：使用方案 B（历史波动率预测），或使用固定 R/R + 波动率调整因子

**优点：**
1. ✅ **标签更真实**：反映在真实波动环境下的表现
2. ✅ **实盘可用**：使用历史波动率预测
3. ✅ **灵活性高**：可以根据不同市场regime调整

## 具体实现建议

### 1. 标签生成阶段（使用未来波动率）

```python
def compute_adaptive_rr_label_with_future_vol(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    atr_col: str = "atr",
    max_holding_bars: int = 50,
    stop_loss_multiplier: float = 1.0,
    take_profit_multiplier: float = 2.0,
    volatility_window: int = 10,
    use_breakeven_stop: bool = True,
) -> pd.Series:
    """
    基于未来波动率的自适应 R/R 标签
    
    对于每个信号：
    1. 计算未来 volatility_window 内的实际波动率
    2. 使用该波动率设定止盈止损：
       - TP = entry ± (未来波动率 × take_profit_multiplier)
       - SL = entry ± (未来波动率 × stop_loss_multiplier)
    3. 如果 use_breakeven_stop=True，当价格达到 stop_loss_multiplier × 未来波动率时，止损上移到保本
    4. 扫描价格路径，判断是否先触达 TP 或 SL
    """
    # 实现细节...
```

### 2. 实盘交易阶段（使用历史波动率预测）

```python
def compute_adaptive_rr_params_from_historical_vol(
    df: pd.DataFrame,
    price_col: str = "close",
    volatility_lookback: int = 20,
    volatility_forecast_window: int = 10,
    base_stop_loss_multiplier: float = 1.0,
    base_take_profit_multiplier: float = 2.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    基于历史波动率预测，计算自适应的止盈止损倍数
    
    返回：
    - stop_loss_multiplier_series: 每个时刻的止损倍数
    - take_profit_multiplier_series: 每个时刻的止盈倍数
    """
    # 1. 计算历史波动率
    historical_vol = df[price_col].pct_change().rolling(
        window=volatility_lookback, min_periods=1
    ).std()
    
    # 2. 预测未来波动率（简化版：使用历史波动率的移动平均）
    predicted_vol = historical_vol.rolling(
        window=volatility_forecast_window, min_periods=1
    ).mean()
    
    # 3. 根据波动率调整倍数
    # 高波动期：增加倍数（因为波动大，需要更大的空间）
    # 低波动期：减少倍数（因为波动小，可以更紧密）
    vol_percentile = predicted_vol.rolling(window=100, min_periods=1).rank(pct=True)
    
    # 调整因子：波动率越高，倍数越大
    adjustment_factor = 0.5 + vol_percentile * 1.0  # 0.5x ~ 1.5x
    
    stop_loss_multiplier_series = base_stop_loss_multiplier * adjustment_factor
    take_profit_multiplier_series = base_take_profit_multiplier * adjustment_factor
    
    return stop_loss_multiplier_series, take_profit_multiplier_series
```

## 方案可行性分析

### ✅ 方案 A（未来波动率）的可行性

**标签生成阶段：完全可行**
- ✅ 标签本身就是基于未来结果，使用未来波动率是合理的
- ✅ 能生成更真实的标签，反映"在真实波动环境下的表现"
- ✅ 可以用于训练模型，让模型学习"在不同波动环境下，策略的表现"

**实盘交易阶段：不可行**
- ❌ 无法知道未来波动率
- ⚠️ 需要预测模型来估计未来波动率

### ✅ 方案 B（历史波动率预测）的可行性

**标签生成阶段：可行但不够准确**
- ⚠️ 历史波动率 ≠ 未来波动率，可能导致标签不够准确
- ✅ 但可以用于验证"如果使用历史波动率预测，策略表现如何"

**实盘交易阶段：完全可行**
- ✅ 只使用历史信息，无未来泄露
- ✅ 可以实时计算和调整

### ✅ 方案 C（混合方案）的可行性

**标签生成阶段：使用方案 A**
- ✅ 生成更真实的标签
- ✅ 让模型学习"在不同波动环境下的表现"

**实盘交易阶段：使用方案 B**
- ✅ 使用历史波动率预测
- ✅ 或使用固定 R/R + 波动率调整因子

## 推荐实现方案

### 阶段 1：标签生成（使用未来波动率）

```python
def compute_adaptive_rr_label_future_vol(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    atr_col: str = "atr",
    max_holding_bars: int = 50,
    stop_loss_multiplier: float = 1.0,
    take_profit_multiplier: float = 2.0,
    volatility_window: int = 10,
    use_breakeven_stop: bool = True,
) -> pd.Series:
    """
    基于未来波动率的自适应 R/R 标签
    
    对于每个信号点 i：
    1. 计算未来 [i+1, i+volatility_window] 窗口内的实际波动率
    2. 使用该波动率设定止盈止损
    3. 扫描价格路径，判断是否先触达 TP 或 SL
    """
    # 实现...
```

### 阶段 2：实盘交易（使用历史波动率预测）

```python
def get_adaptive_rr_params(
    df: pd.DataFrame,
    price_col: str = "close",
    volatility_lookback: int = 20,
    base_stop_loss_multiplier: float = 1.0,
    base_take_profit_multiplier: float = 2.0,
) -> Dict[str, pd.Series]:
    """
    基于历史波动率，计算自适应的止盈止损倍数（实盘可用）
    
    返回：
    - stop_loss_multiplier: 每个时刻的止损倍数
    - take_profit_multiplier: 每个时刻的止盈倍数
    """
    # 实现...
```

## 预期效果

### 标签质量提升
- ✅ 更真实的标签：反映在真实波动环境下的表现
- ✅ 减少标签噪音：避免在低波动期被噪音触发，高波动期等待过久

### 策略表现提升
- ✅ 自适应：能根据市场regime调整
- ✅ 更合理的止盈止损：基于实际波动率，而非固定倍数

## 注意事项

1. **未来信息泄露**：
   - 标签生成阶段可以使用未来波动率（因为标签本身就是基于未来结果）
   - 实盘交易阶段必须使用历史波动率预测

2. **波动率预测准确性**：
   - 历史波动率 ≠ 未来波动率
   - 需要验证预测模型的准确性

3. **参数调优**：
   - `volatility_window`：计算未来波动率的窗口大小
   - `stop_loss_multiplier` / `take_profit_multiplier`：相对于波动率的倍数
   - 需要回测验证最优参数

## 总结

**方案可行性：✅ 完全可行**

**推荐实现路径：**
1. 先实现方案 A（未来波动率）用于标签生成
2. 实现方案 B（历史波动率预测）用于实盘交易
3. 对比固定 R/R 和自适应 R/R 的效果
4. 根据回测结果优化参数

**关键优势：**
- 自适应：能根据市场波动调整
- 更真实：标签反映真实波动环境下的表现
- 实盘可用：使用历史波动率预测，无未来泄露

