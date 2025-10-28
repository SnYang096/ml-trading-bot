# 性能问题分析与优化

## 🐌 性能瓶颈

### 1. 数据量巨大
```
两周数据: 13个ZIP文件
总ticks: ~14,000,000
预估内存: ~2.5 GB
```

### 2. 每分钟都处理 ⚠️
```
1m级别 = 1440分钟/天
两周 = 20,160分钟
每次处理:
  - SR检测 (execution + tactical + strategic)
  - State检测
  - Confluence融合
  - 三层决策
→ 总计算量 = 20,160 × (SR检测 + State + Confluence + 三层决策)
```

### 3. SR检测最耗时
```python
# 每次_process_signals都调用
for tf in self.timeframes:
    srs = self.models[tf].detect_sr_levels(bars)  # ⚠️ 很慢！
    for sr in srs:
        conf, state, trigger, feats = self.models[tf].score_sr(...)
```

## ✅ 已实施的优化

### 1. 只在执行层处理 ✓
```python
if tf == self.execution_tf:  # 只在1m bar时处理
    self._process_signals()
else:
    self.log.debug(f"{tf}层bar已更新，等待{self.execution_tf}层触发")
```

**效果**: 减少处理次数（之前每个层级都处理）

### 2. 复用SR结果 ✓
```python
# 从local_scores中获取已计算的SR（避免重复计算）
for tf, conf, state, trigger, sr, feats in local_scores:
    if tf == self.tactical_tf:
        sr_zones.append({...})
```

**效果**: 不重复调用`detect_sr_levels()`

### 3. 早期退出 ✓
```python
if decision.signal is None:
    return  # 无信号直接返回，不做三层决策
```

## 🚀 进一步优化建议

### 优化1: SR缓存（最重要！）⭐⭐⭐
```python
class DynamicSRStrategy:
    def __init__(self):
        self.sr_cache = {}  # {tf: (timestamp, sr_list)}
        self.sr_cache_ttl = 600  # 10分钟有效期
    
    def _process_signals(self):
        for tf in self.timeframes:
            # 检查缓存
            if tf in self.sr_cache:
                cache_time, cached_srs = self.sr_cache[tf]
                if current_time - cache_time < self.sr_cache_ttl:
                    srs = cached_srs  # 使用缓存
                    continue
            
            # 缓存未命中，重新计算
            srs = self.models[tf].detect_sr_levels(bars)
            self.sr_cache[tf] = (current_time, srs)
```

**预期效果**: 减少90%+ SR检测计算

### 优化2: 降低INFO日志级别
```yaml
# config.yaml
logging_level: "WARNING"  # 从INFO改为WARNING
```

**效果**: 减少日志IO开销

### 优化3: 减少tail()操作
```python
# 当前
self.bars_data[tf] = pd.concat([self.bars_data[tf], bar_df]).tail(1000)

# 优化
if len(self.bars_data[tf]) >= 1000:
    self.bars_data[tf] = self.bars_data[tf].iloc[-999:]  # 手动切片更快
self.bars_data[tf] = pd.concat([self.bars_data[tf], bar_df])
```

### 优化4: 战略层降级模式
```python
# 如果战略层数据不足，允许降级
if strategic.confidence == 0 and self.config.get("allow_degraded_mode"):
    # 只用战术层+执行层决策
    should_trade = tactical_pass and execution_pass
```

### 优化5: 使用更大的执行层周期
```yaml
# 改成5m执行层（处理频率降低5倍）
bar_types:
  "execution": "BTCUSDT.BINANCE-5-MINUTE-LAST-INTERNAL"
  "tactical": "BTCUSDT.BINANCE-30-MINUTE-LAST-INTERNAL"
  "strategic": "BTCUSDT.BINANCE-4-HOUR-LAST-INTERNAL"
```

**效果**: 1天处理次数从1440降到288

## 📊 性能对比

### 当前（1m/10m/2h）
```
执行层频率: 1440次/天
战术层频率: 144次/天
战略层频率: 12次/天
总处理: ~1440次/天（只在execution触发）
```

### 优化后（5m/30m/4h）
```
执行层频率: 288次/天
战术层频率: 48次/天
战略层频率: 6次/天
总处理: ~288次/天
性能提升: 5倍！
```

## 🎯 立即行动建议

### 方案A: 快速修复（推荐）
1. 改成5m/30m/4h（提升5倍）
2. 添加SR缓存（提升10倍）
3. 降低日志级别到WARNING

**预期效果**: 50倍性能提升

### 方案B: 激进优化
1. 改成15m/1h/4h（提升15倍）
2. 添加SR缓存
3. 实现降级模式

**预期效果**: 100倍+性能提升

## 🔧 代码示例

### SR缓存实现
```python
def _get_sr_with_cache(self, tf: str, bars: pd.DataFrame):
    """获取SR（带缓存）"""
    current_time = bars.iloc[-1]['timestamp'] if not bars.empty else 0
    
    # 检查缓存
    if tf in self.sr_cache:
        cache_time, cached_srs = self.sr_cache[tf]
        # 如果在同一个大周期内（如30分钟），使用缓存
        cache_interval = 600_000_000_000  # 10分钟(纳秒)
        if (current_time - cache_time) < cache_interval:
            return cached_srs
    
    # 重新计算
    srs = self.models[tf].detect_sr_levels(bars)
    self.sr_cache[tf] = (current_time, srs)
    return srs
```

---

**结论**: 
1. 当前1m级别太频繁，导致性能差
2. 每次SR检测耗时长
3. 建议改成5m/30m/4h + SR缓存

