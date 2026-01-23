# 订单流数据存储和恢复系统

## 概述

本系统实现了完整的订单流数据监听、聚合、特征计算和存储功能，支持从断线中恢复。

## 存储结构

数据按以下结构存储：

```
data/live_storage/
├── features_4h/
│   └── {symbol}/
│       └── {YYYY-MM-DD}.parquet      # 4小时特征（每4小时保存）
├── features_15min/
│   └── {symbol}/
│       └── {YYYY-MM-DD}.parquet       # 15分钟特征（每15分钟保存）
└── ticks/
    └── {symbol}/
        └── {YYYY-MM-DD}.parquet       # 1分钟聚合tick数据（实时保存，包括未完成的bar）
```

## 关键设计

### 1. 4小时特征存储 (`features_4h/`)

- **用途**: Warmup启动时加载历史特征
- **保存频率**: 每4小时保存一次
- **重要性**: 如果不定期保存，warmup时无法加载历史特征

### 2. 15分钟特征存储 (`features_15min/`)

- **用途**: 
  - Warmup启动时恢复特征计算状态
  - 从断线中恢复时知道最后计算到哪个时间点
- **保存频率**: 每15分钟保存一次
- **重要性**: 如果不定期保存，无法恢复特征计算状态

### 3. 1分钟聚合tick数据存储 (`ticks/`)

- **用途**: 
  - 补数据时知道从哪里开始
  - 保存未完成的bar，用于恢复时知道当前bar的状态
- **保存频率**: 实时保存（包括未完成的bar）
- **重要性**: 如果不保存未完成的bar，补数据时不知道从哪里开始

## 使用示例

### 基本使用

```python
from src.live_data_stream import StorageManager, OrderFlowListener
from src.time_series_model.live.incremental_feature_computer import IncrementalFeatureComputer

# 1. 创建存储管理器
storage_manager = StorageManager(base_path="data/live_storage")

# 2. 创建特征计算器
feature_computer = IncrementalFeatureComputer(
    tick_window_minutes=240,  # 4小时
    bar_window_size=240,      # 4小时（假设1分钟bar）
)

# 3. 创建订单流监听器
listener = OrderFlowListener(
    symbol="BTCUSDT",
    storage_manager=storage_manager,
    feature_computer=feature_computer,
    memory_window_hours=4,
    feature_compute_interval_minutes=15,
    feature_4h_interval_hours=4,
)

# 4. Warmup（加载历史数据）
warmup_data = listener.warmup(days=30)
print(f"加载了 {len(warmup_data['features_4h'])} 条4小时特征")
print(f"加载了 {len(warmup_data['features_15min'])} 条15分钟特征")
print(f"加载了 {len(warmup_data['ticks_1min'])} 条1分钟bar")

# 5. 启动监听器
await listener.start()

# 6. 处理 TradeTick 事件（需要集成 Nautilus Trader）
# 在 Nautilus Trader 的回调中调用：
# listener.on_trade_tick(tick)

# 7. 停止监听器
await listener.stop()
```

### 从断线中恢复

```python
# 1. 获取恢复状态
recovery_state = listener.get_recovery_state()
print(f"最新15分钟特征时间: {recovery_state['latest_15min_timestamp']}")
print(f"最新1分钟bar时间: {recovery_state['latest_1min_timestamp']}")
print(f"未完成的bar: {recovery_state['incomplete_bar']}")

# 2. 如果发现数据缺失，可以补数据
if recovery_state['latest_1min_timestamp']:
    # 从币安API补数据（从 latest_1min_timestamp 到现在）
    # ... 补数据逻辑 ...
    pass

# 3. Warmup恢复状态
listener.warmup(days=30)
```

### 补数据逻辑

```python
from src.live_data_stream import StorageManager

storage_manager = StorageManager()

# 获取恢复状态
recovery_state = storage_manager.get_recovery_state("BTCUSDT")

if recovery_state['latest_1min_timestamp']:
    # 计算需要补数据的时间范围
    latest_ts = recovery_state['latest_1min_timestamp']
    now = pd.Timestamp.now(tz="UTC")
    
    # 如果缺失超过1天，从币安API获取
    if (now - latest_ts).total_seconds() > 86400:
        # 从币安API获取数据
        # ... 实现补数据逻辑 ...
        pass
    else:
        # 如果缺失在1天内，从本地Parquet文件加载（如果有）
        # ... 实现warmup逻辑 ...
        pass
```

## 存储类说明

### `StorageManager`

统一管理三种存储，提供便捷的保存和加载接口。

**主要方法**:
- `save_4h_features()`: 保存4小时特征
- `save_15min_features()`: 保存15分钟特征
- `save_1min_ticks()`: 保存1分钟聚合tick数据
- `warmup_load()`: 加载warmup数据
- `get_recovery_state()`: 获取恢复状态

### `OrderFlowListener`

订单流监听器，集成Nautilus Trader、特征计算和存储。

**主要功能**:
- 监听 TradeTick 事件
- 按1分钟聚合tick数据
- 维护内存滑动窗口（默认4小时）
- 每15分钟计算特征并保存
- 每4小时聚合特征并保存
- 支持从断线中恢复

**主要方法**:
- `on_trade_tick()`: 处理 TradeTick 事件
- `warmup()`: 加载warmup数据
- `start()`: 启动监听器
- `stop()`: 停止监听器
- `get_recovery_state()`: 获取恢复状态

## 注意事项

1. **定期保存**: 确保4小时特征和15分钟特征定期保存，否则warmup时无法加载
2. **未完成的bar**: 1分钟tick数据需要保存未完成的bar，用于补数据时知道从哪里开始
3. **内存管理**: 内存滑动窗口默认4小时，可以根据需要调整
4. **恢复逻辑**: 从断线中恢复时，需要检查数据缺失情况，决定是从本地文件warmup还是从币安API补数据

## 文件说明

- `feature_storage.py`: 存储类实现（Feature4HStorage, Feature15MinStorage, Tick1MinStorage, StorageManager）
- `order_flow_listener.py`: 订单流监听器实现（OrderFlowListener）
- `README_STORAGE.md`: 本文档
