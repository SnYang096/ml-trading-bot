# 订单流数据监听与补数据系统使用示例

## 概述

本系统实现了完整的订单流数据监听、聚合、特征计算和存储功能，支持从断线中恢复。

## 快速开始

### 1. 基本使用

```python
import asyncio
from src.live_data_stream import (
    StorageManager,
    OrderFlowListener,
    GapFiller,
    OrderFlowListenerConfig,
)
from src.time_series_model.live.incremental_feature_computer import IncrementalFeatureComputer

# 1. 创建存储管理器
storage_manager = StorageManager(base_path="data/live_storage")

# 2. 创建配置
config = OrderFlowListenerConfig(
    symbol="BTCUSDT",
    memory_window_hours=4.0,
    feature_compute_interval_minutes=15,
    feature_4h_interval_hours=4,
)

# 3. 创建特征计算器
feature_computer = IncrementalFeatureComputer(
    tick_window_minutes=config.tick_window_minutes,
    bar_window_size=config.bar_window_size,
)

# 4. 创建数据补全器（可选，需要ccxt exchange）
try:
    import ccxt
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    gap_filler = GapFiller(
        storage_manager=storage_manager,
        exchange=exchange,
        feature_store_dir="feature_store",
        feature_store_layer="features_xxx",
    )
except ImportError:
    gap_filler = None

# 5. 创建订单流监听器
listener = OrderFlowListener(
    symbol=config.symbol,
    storage_manager=storage_manager,
    feature_computer=feature_computer,
    gap_filler=gap_filler,
    memory_window_hours=config.memory_window_hours,
    feature_compute_interval_minutes=config.feature_compute_interval_minutes,
    feature_4h_interval_hours=config.feature_4h_interval_hours,
    # 可选：实盘下单，需配置环境变量后自动注入 OrderManager
    # order_manager=init_order_manager_from_env(),
)

# 6. Warmup（加载历史数据）
warmup_data = listener.warmup(days=30, use_gap_filler=True)
print(f"加载了 {len(warmup_data.get('features_4h', pd.DataFrame()))} 条4小时特征")
print(f"加载了 {len(warmup_data.get('features_15min', pd.DataFrame()))} 条15分钟特征")
print(f"加载了 {len(warmup_data.get('ticks_1min', pd.DataFrame()))} 条1分钟bar")

# 7. 启动监听器
await listener.start()

# 8. 处理 TradeTick 事件（需要集成 Nautilus Trader）
# 在 Nautilus Trader 的回调中调用：
# listener.on_trade_tick(tick)

# 9. 停止监听器
await listener.stop()
```

### 2. 从断线中恢复

```python
# 1. 创建监听器（同上）
listener = OrderFlowListener(...)

# 2. 从断线中恢复
recovery_state = listener.recover_from_interruption()
print(f"恢复状态: {recovery_state}")

# 3. Warmup恢复数据
warmup_data = listener.warmup(days=30, use_gap_filler=True)

# 4. 继续运行
await listener.start()
```

### 3. 使用配置类

```python
from src.live_data_stream import OrderFlowListenerConfig

# 从字典创建配置
config_dict = {
    "symbol": "BTCUSDT",
    "memory_window_hours": 4.0,
    "feature_compute_interval_minutes": 15,
    "feature_4h_interval_hours": 4,
    "feature_store_dir": "feature_store",
    "feature_store_layer": "features_xxx",
}

config = OrderFlowListenerConfig.from_dict(config_dict)

# 使用配置创建监听器
listener = OrderFlowListener(
    symbol=config.symbol,
    storage_manager=storage_manager,
    memory_window_hours=config.memory_window_hours,
    feature_compute_interval_minutes=config.feature_compute_interval_minutes,
    feature_4h_interval_hours=config.feature_4h_interval_hours,
)
```

### 4. 手动补数据

```python
from src.live_data_stream import GapFiller
import pandas as pd

# 创建数据补全器
gap_filler = GapFiller(
    storage_manager=storage_manager,
    exchange=exchange,
    feature_store_dir="feature_store",
    feature_store_layer="features_xxx",
)

# 补数据（自动选择数据源）
start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
end_time = pd.Timestamp("2024-01-02 00:00:00", tz="UTC")

fill_data = gap_filler.fill_gap(
    symbol="BTCUSDT",
    start_time=start_time,
    end_time=end_time,
    source="auto",  # 或 "feature_store", "parquet", "binance"
)

print(f"补全了 {len(fill_data)} 条数据")
```

### 5. 使用内存窗口

```python
from src.live_data_stream import MemoryWindow

# 创建内存窗口（4小时）
memory_window = MemoryWindow(window_hours=4.0)

# 添加数据
memory_window.add({
    "timestamp": pd.Timestamp.now(tz="UTC"),
    "open": 50000.0,
    "high": 50100.0,
    "low": 49900.0,
    "close": 50050.0,
    "volume": 100.0,
})

# 获取时间范围内的数据
start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
end_time = pd.Timestamp("2024-01-01 04:00:00", tz="UTC")
data = memory_window.get_range(start_time, end_time)

# 转换为DataFrame
df = memory_window.to_dataframe()

# 获取最新数据
latest = memory_window.get_latest(n=10)
```

## 完整示例：集成Nautilus Trader

```python
import asyncio
from nautilus_trader.model import TradeTick
from nautilus_trader.model.enums import AggressorSide
from src.live_data_stream import (
    StorageManager,
    OrderFlowListener,
    GapFiller,
    OrderFlowListenerConfig,
)

# 配置
config = OrderFlowListenerConfig(
    symbol="BTCUSDT",
    memory_window_hours=4.0,
    feature_compute_interval_minutes=15,
    feature_4h_interval_hours=4,
)

# 创建组件
storage_manager = StorageManager()
gap_filler = GapFiller(storage_manager, exchange=exchange)
listener = OrderFlowListener(
    symbol=config.symbol,
    storage_manager=storage_manager,
    gap_filler=gap_filler,
    memory_window_hours=config.memory_window_hours,
)

# Warmup
listener.warmup(days=30, use_gap_filler=True)

# 启动
await listener.start()

# 在Nautilus Trader的回调中处理TradeTick
def on_trade_tick(tick: TradeTick):
    listener.on_trade_tick(tick)

# 停止
await listener.stop()
```

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
        └── {YYYY-MM-DD}.parquet       # 1分钟聚合tick数据（实时保存）
```

## 注意事项

1. **定期保存**: 确保4小时特征和15分钟特征定期保存，否则warmup时无法加载
2. **未完成的bar**: 1分钟tick数据需要保存未完成的bar，用于补数据时知道从哪里开始
3. **内存管理**: 内存滑动窗口默认4小时，可以根据需要调整
4. **恢复逻辑**: 从断线中恢复时，需要检查数据缺失情况，决定是从本地文件warmup还是从币安API补数据
5. **Feature Store**: 如果配置了Feature Store，Warmup时会优先从Feature Store加载特征

## 相关文档

- `README_STORAGE.md`: 存储系统详细说明
- `feature_storage.py`: 存储类实现
- `order_flow_listener.py`: 订单流监听器实现
- `gap_filler.py`: 数据补全器实现
- `memory_window.py`: 内存窗口实现
