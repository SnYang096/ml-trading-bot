# Nautilus Trader 集成说明

## 概述

`OrderFlowListener` 设计为与 Nautilus Trader 集成，通过 Nautilus Trader 的数据客户端订阅实时 tick 数据流。

## 两种使用方式

### 方式1：使用 Nautilus Trader 数据客户端（推荐，生产环境）

**特点**：
- 使用 Nautilus Trader 的 `BinanceDataClient` 订阅实时数据
- 数据流由 Nautilus Trader 管理，自动处理重连、错误恢复等
- 适合生产环境

**实现**：
- 创建 `OrderFlowStrategy`（继承自 Nautilus Trader 的 `Strategy`）
- 在 `on_trade_tick` 回调中调用 `OrderFlowListener.on_trade_tick`
- 使用 `TradingNode` 运行策略

**示例**：
```python
from src.live_data_stream import create_order_flow_node
import asyncio

# 创建并运行订单流监听器（使用 Nautilus Trader）
async def main():
    node = create_order_flow_node(
        symbol="BTCUSDT",
        storage_path="data/live_storage",
        testnet=True,
    )
    
    node.start()
    # Nautilus Trader 会自动订阅 tick 数据并调用 OrderFlowStrategy.on_trade_tick
    await asyncio.Event().wait()

asyncio.run(main())
```

### 方式2：手动加载数据（测试环境）

**特点**：
- 从 parquet 文件或其他数据源加载数据
- 手动转换为 `TradeTick` 对象
- 适合测试和回测

**实现**：
- 使用 `TickDataSimulator` 从 parquet 文件加载数据
- 转换为 `TradeTick` 对象
- 手动调用 `OrderFlowListener.on_trade_tick`

**示例**：
```python
from src.live_data_stream import OrderFlowListener, StorageManager
from src.live_data_stream.tests.test_data_simulator import TickDataSimulator

# 创建监听器
storage_manager = StorageManager()
listener = OrderFlowListener(
    symbol="BTCUSDT",
    storage_manager=storage_manager,
)

# 从 parquet 文件加载数据
simulator = TickDataSimulator(
    symbol="BTCUSDT",
    data_dir="data/parquet_data_1s",
)

# 处理 tick 数据
for tick in simulator.stream_ticks():
    listener.on_trade_tick(tick)
```

## 当前实现状态

### 测试环境

- ✅ 使用 `TickDataSimulator` 从 `parquet_data_1s` 加载数据
- ✅ 转换为 Nautilus Trader 的真实 `TradeTick` 对象（非 Mock）
- ✅ 测试通过，功能正常

### 生产环境（待集成）

- 📝 已创建 `nautilus_integration.py` 提供集成示例
- 📝 需要在实际 Nautilus Trader 策略中集成 `OrderFlowListener`

## 集成到现有 Nautilus Trader 策略

如果你已经有 Nautilus Trader 策略，可以这样集成：

```python
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import TradeTick
from src.live_data_stream import OrderFlowListener, StorageManager

class YourStrategy(Strategy):
    def __init__(self, ...):
        super().__init__()
        # 创建 OrderFlowListener
        storage_manager = StorageManager()
        self.order_flow_listener = OrderFlowListener(
            symbol="BTCUSDT",
            storage_manager=storage_manager,
        )
    
    def on_start(self):
        # 订阅 trade ticks
        self.subscribe_trade_ticks(self.instrument_id)
        
        # Warmup
        self.order_flow_listener.warmup(days=30)
        
        # 启动监听器
        asyncio.create_task(self.order_flow_listener.start())
    
    def on_trade_tick(self, tick: TradeTick):
        # 传递给 OrderFlowListener 处理
        self.order_flow_listener.on_trade_tick(tick)
        
        # 你的其他逻辑...
```

## 数据流架构

```
Nautilus Trader BinanceDataClient
    ↓ (订阅实时数据)
OrderFlowStrategy.on_trade_tick()
    ↓ (调用)
OrderFlowListener.on_trade_tick()
    ↓ (处理)
    ├─ 1分钟聚合
    ├─ 内存滑动窗口
    ├─ 特征计算（每15分钟）
    └─ 数据保存（Parquet）
```

## 优势

使用 Nautilus Trader 数据客户端的优势：

1. **自动重连**：Nautilus Trader 自动处理 WebSocket 断线重连
2. **错误恢复**：内置错误处理和恢复机制
3. **统一接口**：与其他 Nautilus Trader 组件统一
4. **生产就绪**：经过生产环境验证的数据客户端

## API Key配置

### 环境变量方式（推荐）

```bash
# 测试网
export BINANCE_FUTURES_TESTNET_API_KEY="your_testnet_api_key"
export BINANCE_FUTURES_TESTNET_API_SECRET="your_testnet_api_secret"

# 主网
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

### 配置文件方式

创建 `config/local/binance_testnet.env`:

```bash
export BINANCE_FUTURES_TESTNET_API_KEY="your_testnet_api_key"
export BINANCE_FUTURES_TESTNET_API_SECRET="your_testnet_api_secret"
```

然后加载：

```bash
source config/local/binance_testnet.env
```

### 获取API Key

1. **测试网**：https://testnet.binancefuture.com/
2. **主网**：https://www.binance.com/

⚠️ **注意**：
- 不要将API key提交到Git
- 主网API key有实际资金权限，请妥善保管
- 建议先用测试网测试

## 相关文件

- `src/live_data_stream/nautilus_integration.py` - Nautilus Trader 集成示例
- `src/live_data_stream/order_flow_listener.py` - 订单流监听器
- `src/live_data_stream/websocket_client.py` - BinanceWebSocketClient（不需要API key）
- `tests/live_data_stream/test_data_simulator.py` - 测试数据模拟器
- `tests/live_data_stream/test_live_binance_ws.py` - 币安WebSocket实盘测试
- `src/live_data_stream/LIVE_TESTING.md` - 实盘测试详细指南
