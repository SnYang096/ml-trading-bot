# VectorBT优化与预聚合方案

## 🎯 目标

1. **预聚合K线** - 将tick预先聚合，避免实时计算
2. **Warmup数据** - 为大周期提供足够历史数据
3. **进度显示** - 实时显示回测进度
4. **VectorBT对比** - 快速验证策略逻辑
5. **保证一致性** - 回测与实盘结果一致

## 📊 当前性能问题

### Nautilus回测慢的原因
```
1. 实时聚合tick → bars: 
   - 每个tick都要处理
   - 14M ticks 非常慢
   
2. 大周期数据不足:
   - 单日只有6个4H bar
   - 战略层无法工作
   
3. 没有进度显示:
   - 只能盲等
   - 不知道还要多久
```

## ✅ 解决方案

### 方案1: 预聚合K线 ⭐⭐⭐

```python
# bar_aggregator.py (已实现)
def pre_aggregate_for_backtest(tick_file, timeframes):
    # 1. 加载tick数据
    df_ticks = pd.read_csv(tick_file)
    
    # 2. 计算买卖成交量
    df_ticks["buy_vol"] = np.where(~df_ticks["is_buyer_maker"], df_ticks["qty"], 0)
    df_ticks["sell_vol"] = np.where(df_ticks["is_buyer_maker"], df_ticks["qty"], 0)
    
    # 3. 按频率重采样
    for layer, freq in timeframes.items():
        bars = df_ticks.resample(freq).agg({
            'price': ['first', 'max', 'min', 'last'],  # OHLC
            'qty': 'sum',                               # volume
            'buy_vol': 'sum',
            'sell_vol': 'sum'
        })
        
        # 4. 计算CVD
        bars['cvd'] = (bars['buy_vol'] - bars['sell_vol']).cumsum()
        
    return bars_dict
```

**效果**: 
- 只计算一次，缓存复用
- 避免实时聚合开销
- 提速10倍+

### 方案2: Warmup数据支持 ⭐⭐⭐

```python
def prepare_warmup_and_backtest(bars_dict, warmup_bars_per_layer):
    """
    分离warmup和回测数据
    
    warmup_bars_per_layer = {
        'execution': 100,   # 5m需要100个 ≈ 8小时
        'tactical': 100,    # 30m需要100个 ≈ 2天
        'strategic': 50     # 4h需要50个 ≈ 8天
    }
    """
    for layer, df in bars_dict.items():
        warmup_count = warmup_bars_per_layer[layer]
        warmup_df = df.iloc[:warmup_count]    # 前N个用于warmup
        backtest_df = df.iloc[warmup_count:]  # 剩余用于回测
```

**效果**:
- 战略层有足够历史数据
- 指标计算更准确
- 模拟真实交易环境

### 方案3: 进度显示 ⭐⭐

```python
# nautilus_backtest.py
class ProgressCallback:
    def __init__(self, total_bars):
        self.total = total_bars
        self.current = 0
        self.last_update = 0
        
    def update(self, timestamp):
        self.current += 1
        if self.current % 100 == 0:  # 每100个bar更新一次
            pct = self.current / self.total * 100
            print(f"⏳ 进度: {self.current}/{self.total} ({pct:.1f}%) - {timestamp}")

# 在on_bar中调用
def on_bar(self, bar):
    if hasattr(self, 'progress'):
        self.progress.update(bar.ts_event)
```

### 方案4: VectorBT快速验证 ⭐⭐⭐

```python
import vectorbt as vbt

def vectorbt_backtest(bars_dict, strategy_signals):
    """
    使用VectorBT进行快速回测验证
    
    优势:
    - 向量化计算，极快（秒级）
    - 快速验证策略逻辑
    - 对比Nautilus结果
    """
    # 使用execution层数据
    df = bars_dict['execution']
    
    # 生成信号（复用三层决策）
    entries = []
    exits = []
    
    for i in range(len(df)):
        # 三层决策
        signal = evaluate_three_tiers(i, bars_dict)
        if signal.should_trade:
            entries.append(True)
            exits.append(False)
        else:
            entries.append(False)
            exits.append(False)
    
    # VectorBT回测
    portfolio = vbt.Portfolio.from_signals(
        df['close'],
        entries,
        exits,
        init_cash=100000,
        fees=0.001
    )
    
    print(f"VectorBT结果: 总盈亏={portfolio.total_return():.2%}")
    print(f"夏普比率: {portfolio.sharpe_ratio():.2f}")
    
    return portfolio
```

**工作流程**:
```
1. 预聚合K线（10秒）
2. VectorBT快速回测（5秒）← 验证策略逻辑
3. 如果效果好，再用Nautilus详细回测（5分钟）
```

## 🔧 实施步骤

### Step 1: 修改nautilus_backtest.py

```python
# 添加预聚合支持
from yin_bot.dynamic_sr.bar_aggregator import pre_aggregate_for_backtest, prepare_warmup_and_backtest

def run_backtest_with_config(...):
    # 1. 预聚合K线
    timeframes = {
        'execution': '5min',
        'tactical': '30min',
        'strategic': '4h'
    }
    
    bars_dict = pre_aggregate_for_backtest(
        tick_file=data_file,
        timeframes=timeframes,
        cache_dir="./cache",
        use_cache=True  # 第二次运行直接用缓存
    )
    
    # 2. 分离warmup和backtest
    warmup_dict, backtest_dict = prepare_warmup_and_backtest(
        bars_dict,
        warmup_bars_per_layer={
            'execution': 100,
            'tactical': 100,
            'strategic': 50
        }
    )
    
    # 3. Warmup策略（喂历史数据）
    for layer in ['strategic', 'tactical', 'execution']:
        for bar in warmup_dict[layer]:
            strategy.warmup_bar(layer, bar)  # 新增方法
    
    # 4. 正式回测（用backtest_dict）
    total_bars = len(backtest_dict['execution'])
    progress = ProgressCallback(total_bars)
    
    for bar in backtest_dict['execution']:
        progress.update(bar['timestamp'])
        engine.process_bar(bar)
```

### Step 2: 创建VectorBT验证脚本

```python
# vectorbt_validator.py
from yin_bot.dynamic_sr.bar_aggregator import pre_aggregate_for_backtest
from yin_bot.dynamic_sr.three_tier_layer import ThreeTierLayer
import vectorbt as vbt

def quick_validate_with_vectorbt(data_file, config):
    """快速验证策略（1-2分钟）"""
    
    # 1. 预聚合
    bars_dict = pre_aggregate_for_backtest(data_file, ...)
    
    # 2. 初始化三层决策
    three_tier = ThreeTierLayer(config, models)
    
    # 3. 生成信号
    signals = []
    for i in range(len(bars_dict['execution'])):
        strategic = three_tier.make_strategic_decision(bars_dict['strategic'][:i])
        tactical = three_tier.make_tactical_decision(...)
        execution = three_tier.make_execution_decision(...)
        decision = three_tier.fuse_three_tiers(strategic, tactical, execution)
        
        signals.append(decision.should_trade and decision.final_direction == 'long')
    
    # 4. VectorBT回测
    portfolio = vbt.Portfolio.from_signals(
        bars_dict['execution']['close'],
        signals,
        short_signals=...,
        init_cash=100000
    )
    
    return portfolio.stats()
```

### Step 3: 进度显示

```python
# strategy.py
def on_bar(self, bar):
    # 进度显示（每100个bar）
    if not hasattr(self, '_bar_count'):
        self._bar_count = 0
        self._start_time = time.time()
    
    self._bar_count += 1
    
    if self._bar_count % 100 == 0:
        elapsed = time.time() - self._start_time
        bars_per_sec = self._bar_count / elapsed
        eta = (self._total_bars - self._bar_count) / bars_per_sec if bars_per_sec > 0 else 0
        
        print(f"⏳ 进度: {self._bar_count}/{self._total_bars} ({self._bar_count/self._total_bars*100:.1f}%) "
              f"- 已用{elapsed/60:.1f}分钟, 预计剩余{eta/60:.1f}分钟")
```

## 📈 预期效果对比

| 方案 | 时间 | 用途 |
|------|------|------|
| **当前Nautilus** | 5-10分钟/周 | 详细回测 |
| **预聚合+Nautilus** | 1-2分钟/周 | 详细回测（优化后） |
| **VectorBT** | 5-10秒/周 | 快速验证 |

**工作流程**:
```
1. VectorBT快速验证（10秒）
   ↓ 如果效果不好，调整参数，重新验证
   ↓ 如果效果好
2. 预聚合+Nautilus详细回测（2分钟）
   ↓ 验证细节
3. 参数确定，实盘部署
```

## 🔨 立即实施

### 文件1: `vectorbt_quick_test.py`
```python
"""VectorBT快速测试脚本"""
import vectorbt as vbt
import pandas as pd
from yin_bot.dynamic_sr.bar_aggregator import pre_aggregate_for_backtest

# 预聚合
bars_dict = pre_aggregate_for_backtest(
    tick_file="data/BTCUSDT-aggTrades-2025-05.csv",
    timeframes={'execution': '5min', 'tactical': '30min', 'strategic': '4h'},
    use_cache=True
)

# 简单策略：只用战术层SR
df = bars_dict['execution']

# 生成信号（示例）
entries = df['close'] > df['close'].shift(1)  # 简单上涨做多

# VectorBT回测
pf = vbt.Portfolio.from_signals(
    df['close'],
    entries,
    init_cash=100000,
    fees=0.001
)

print(pf.stats())
```

### 文件2: 修改`nautilus_backtest.py`添加进度

```python
# 在BacktestEngine运行前
total_bars = len(bars_dict['execution'])
bar_count = [0]  # 使用list以便闭包修改

def progress_callback(bar):
    bar_count[0] += 1
    if bar_count[0] % 100 == 0:
        pct = bar_count[0] / total_bars * 100
        print(f"\r⏳ 进度: {bar_count[0]}/{total_bars} ({pct:.1f}%)", end='', flush=True)

# 在策略中添加回调
```

---

**准备就绪！需要我立即实施吗？**

预计改动：
1. `bar_aggregator.py` - 已创建 ✅
2. `vectorbt_quick_test.py` - 新建
3. `nautilus_backtest.py` - 添加预聚合+进度
4. `strategy.py` - 添加warmup_bar方法

是否继续？

