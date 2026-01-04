# Nautilus Trader 集成指南

本指南说明如何将 YAML 特征加载系统与 [Nautilus Trader](https://github.com/nautechsystems/nautilus_trader) 集成，实现实盘交易。

## 目录

- [概述](#概述)
- [架构设计](#架构设计)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [代码示例](#代码示例)
- [性能优化](#性能优化)
- [故障排查](#故障排查)

## 概述

### 集成优势

✅ **事件驱动架构**：Nautilus Trader 的事件驱动架构天然适合实时流特征计算

✅ **无缝集成**：特征加载器可以直接在 Strategy 类中使用，无需修改核心逻辑

✅ **数据一致性**：通过 Nautilus 的数据流保证数据一致性

✅ **状态管理**：利用 Nautilus 的缓存和状态管理机制

✅ **多交易所支持**：支持 Binance、Hyperliquid 等多个交易所

### 核心组件

1. **NautilusStrategyWithFeatures**：集成特征加载的 Strategy 类
2. **RealtimeFeatureManager**：实时流特征管理器
3. **StrategyFeatureLoader**：YAML 配置的特征加载器

## 架构设计

### 数据流

```
Nautilus Bar → DataFrame → 特征计算 → 特征 DataFrame → 信号生成 → 订单执行
```

### 事件处理流程

```
on_start()
  ├─ 加载策略配置
  ├─ 初始化特征管理器
  ├─ 加载模型
  └─ 订阅市场数据

on_bar(bar)
  ├─ Bar → DataFrame
  ├─ 计算特征
  ├─ 生成信号
  └─ 执行交易

on_stop()
  └─ 清理资源
```

## 快速开始

### 1. 安装依赖

```bash
# 安装 Nautilus Trader
pip install nautilus-trader

# 或使用项目依赖
pip install -r requirements.txt
```

### 2. 配置 API 凭证

设置环境变量：

```bash
# Binance 测试网
export BINANCE_FUTURES_TESTNET_API_KEY="your_api_key"
export BINANCE_FUTURES_TESTNET_API_SECRET="your_api_secret"

# Binance 实盘（谨慎使用）
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

### 3. 运行策略

```bash
python -m time_series_model.live.run_nautilus_strategy \
    --strategy sr_reversal \
    --symbol BTCUSDT-PERP \
    --timeframe 15T \
    --testnet \
    --trade-size 0.001
```

## 配置说明

### 策略配置

策略配置位于 `config/strategies/{strategy_name}/` 目录下：

```
config/strategies/sr_reversal/
├── features.yaml      # 特征配置
├── labels.yaml    # 标签配置
├── model.yaml     # 模型配置
├── evaluation.yaml # 评估配置
└── backtest.yaml  # 回测配置
```

### TradingNode 配置

主要配置项：

- **trader_id**：交易者唯一标识
- **data_clients**：数据客户端配置（Binance 等）
- **exec_clients**：执行客户端配置
- **cache**：缓存配置（可选，使用 Redis）
- **exec_engine**：执行引擎配置（对账、重试等）

### 特征管理器配置

```python
feature_manager = RealtimeFeatureManager(
    strategy_name="sr_reversal",
    history_window=1000,  # 历史窗口大小
    config_base_path="config/strategies",
)
```

## 代码示例

### 基本使用

```python
from nautilus_trader.model import InstrumentId, BarType
from nautilus_trader.model import BarSpecification, BarAggregation, PriceType
from nautilus_trader.model import AggregationSource

from src.time_series_model.live.nautilus_strategy_with_features import (
    NautilusStrategyWithFeatures
)

# 创建策略实例
instrument_id = InstrumentId.from_str("BTCUSDT-PERP.BINANCE")
bar_type = BarType(
    instrument_id=instrument_id,
    bar_spec=BarSpecification(15, BarAggregation.MINUTE, PriceType.LAST),
    aggregation_source=AggregationSource.EXTERNAL,
)

strategy = NautilusStrategyWithFeatures(
    strategy_name="sr_reversal",
    instrument_id=instrument_id,
    bar_type=bar_type,
    trade_size=0.001,
    history_window=1000,
)
```

### 自定义信号生成

在 `NautilusStrategyWithFeatures` 中重写 `_generate_signal()` 方法：

```python
def _generate_signal(self, features_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """自定义信号生成逻辑"""
    # 你的自定义逻辑
    if features_df["rsi"].iloc[0] < 30:
        return {"side": OrderSide.BUY, "prediction": 1}
    elif features_df["rsi"].iloc[0] > 70:
        return {"side": OrderSide.SELL, "prediction": -1}
    return None
```

### 使用模型预测

```python
# 在 on_start() 中加载模型
self.model = self._load_model("models/sr_reversal/model.pkl")

# 在 _generate_signal() 中使用模型
prediction = self.model.predict(X)[0]
```

## 性能优化

### 1. 特征计算优化

- ✅ 使用 `max_workers=1` 禁用并行计算（实时流中串行更快）
- ✅ 预编译依赖图，避免重复解析
- ✅ 增量计算：只计算新数据，复用历史结果

### 2. 内存管理

- ✅ 限制历史窗口大小（如 1000 条）
- ✅ 定期清理过期缓存
- ✅ 使用 Nautilus Trader 的内存管理配置

### 3. 延迟优化

- ✅ 特征计算放在异步任务中（如果允许）
- ✅ 使用缓存避免重复计算
- ✅ 监控特征计算耗时

## 故障排查

### 常见问题

#### 1. 特征计算失败

**症状**：`on_bar()` 中特征计算报错

**解决方案**：
- 检查历史数据是否足够（某些特征需要较长历史窗口）
- 检查特征依赖配置是否正确
- 查看日志中的详细错误信息

#### 2. 模型加载失败

**症状**：`on_start()` 中模型加载失败

**解决方案**：
- 检查模型文件路径是否正确
- 确保模型文件格式正确（pickle）
- 检查模型版本是否与特征版本匹配

#### 3. API 连接失败

**症状**：无法连接到 Binance

**解决方案**：
- 检查 API 凭证是否正确
- 检查网络连接
- 确认是否使用测试网（`--testnet`）

#### 4. 订单执行失败

**症状**：信号生成但订单未执行

**解决方案**：
- 检查账户余额
- 检查风险限制
- 查看 Nautilus Trader 日志

### 调试技巧

1. **启用详细日志**：
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **监控特征计算耗时**：
   ```python
   import time
   start = time.time()
   features_df = self.feature_manager.compute_features(new_bar_df)
   elapsed = time.time() - start
   self.log.info(f"Feature computation took {elapsed:.3f}s")
   ```

3. **检查特征值**：
   ```python
   self.log.debug(f"Features: {latest_features.to_dict()}")
   ```

## 参考文档

- [Nautilus Trader 官方文档](https://docs.nautilustrader.io/)
- [实盘特征加载流程分析（legacy）](../legacy/实盘特征加载流程分析.md)
- [特征加载器 README](../src/features/loader/README.md)

## 相关文件

- `src/time_series_model/live/nautilus_strategy_with_features.py`：Strategy 集成类
- `src/time_series_model/live/realtime_feature_integration_example.py`：特征管理器示例
- `src/time_series_model/live/run_nautilus_strategy.py`：运行脚本入口

## 贡献

欢迎提交 Issue 和 Pull Request 来改进集成方案！

