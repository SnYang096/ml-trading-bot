# 币安WebSocket实盘测试指南

## 概述

本指南说明如何使用币安WebSocket连接实盘数据流，并集成到订单流监听系统。

## 重要说明

### API Key需求

- **WebSocket公共数据流（trade stream）不需要API key**：币安的trade stream是公开的，任何人都可以订阅
- **Nautilus Trader数据客户端需要API key**：用于订阅某些需要认证的数据流（如用户数据流、订单更新等）

### 测试网 vs 主网

- **测试网**：建议先用测试网测试，避免影响实盘交易
- **主网**：确认无误后再使用主网

## 方式1：使用BinanceWebSocketClient（推荐用于测试）

### 特点

- ✅ 不需要API key
- ✅ 简单直接
- ✅ 自动重连
- ✅ 支持多symbol订阅

### 使用示例

```python
from src.live_data_stream.websocket_client import BinanceWebSocketClient
import asyncio

async def main():
    # 创建WebSocket客户端
    client = BinanceWebSocketClient(
        symbols=["BTCUSDT", "ETHUSDT"],
        use_futures=True,  # 使用期货市场
    )
    
    stop_event = asyncio.Event()
    
    # 接收tick数据
    async for tick in client.stream_ticks(stop_event):
        print(f"收到tick: {tick.symbol} @ {tick.price} (vol: {tick.volume})")
        
        # 处理tick数据...
        
        # 停止条件
        if some_condition:
            stop_event.set()
            break

asyncio.run(main())
```

### 集成到OrderFlowListener

```python
from src.live_data_stream import OrderFlowListener, StorageManager
from src.live_data_stream.websocket_client import BinanceWebSocketClient
import asyncio

async def main():
    # 创建OrderFlowListener
    storage_manager = StorageManager()
    listener = OrderFlowListener(
        symbol="BTCUSDT",
        storage_manager=storage_manager,
    )
    
    # 创建WebSocket客户端
    client = BinanceWebSocketClient(
        symbols=["BTCUSDT"],
        use_futures=True,
    )
    
    stop_event = asyncio.Event()
    
    async for tick in client.stream_ticks(stop_event):
        # 将BinanceTick转换为TradeTick格式
        # 注意：需要适配层，因为OrderFlowListener期望Nautilus Trader的TradeTick对象
        # 或者使用方式2（Nautilus Trader）
        pass

asyncio.run(main())
```

## 方式2：使用Nautilus Trader（推荐用于生产）

### 特点

- ✅ 完整的交易框架
- ✅ 统一的数据格式（TradeTick）
- ✅ 自动重连和错误处理
- ✅ 支持订单执行
- ⚠️ 需要API key（用于数据客户端）

### API Key配置

#### 方式1：环境变量（推荐）

```bash
# 测试网
export BINANCE_FUTURES_TESTNET_API_KEY="your_testnet_api_key"
export BINANCE_FUTURES_TESTNET_API_SECRET="your_testnet_api_secret"

# 主网
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

#### 方式2：配置文件

创建 `config/local/binance_testnet.env`:

```bash
export BINANCE_FUTURES_TESTNET_API_KEY="your_testnet_api_key"
export BINANCE_FUTURES_TESTNET_API_SECRET="your_testnet_api_secret"
```

然后加载：

```bash
source config/local/binance_testnet.env
```

### 使用示例

```python
from src.live_data_stream.nautilus_integration import run_order_flow_listener
import asyncio

# 单symbol
asyncio.run(run_order_flow_listener(
    symbol="BTCUSDT",
    storage_path="data/live_storage",
    testnet=True,  # 使用测试网
))

# 多symbol
from src.live_data_stream.nautilus_integration import run_multi_symbol_order_flow_listener

asyncio.run(run_multi_symbol_order_flow_listener(
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    storage_path="data/live_storage",
    testnet=True,
))
```

### 获取API Key

1. **测试网**：
   - 访问：https://testnet.binancefuture.com/
   - 注册账号
   - 在API管理页面创建API key

2. **主网**：
   - 访问：https://www.binance.com/
   - 登录账号
   - 在API管理页面创建API key
   - ⚠️ 注意：主网API key有实际资金权限，请妥善保管

## 运行测试

### 测试WebSocket连接

```bash
# 测试基本连接（不处理数据）
pytest tests/live_data_stream/test_live_binance_ws.py::test_binance_websocket_connection -v -s

# 测试多symbol连接
pytest tests/live_data_stream/test_live_binance_ws.py::test_binance_websocket_multi_symbol -v -s
```

### 测试完整集成

```bash
# 测试WebSocket + OrderFlowListener集成
pytest tests/live_data_stream/test_live_binance_ws.py::test_binance_websocket_with_order_flow_listener -v -s
```

## 注意事项

### 1. 网络连接

- 确保网络可以访问币安WebSocket服务器
- 如果在中国大陆，可能需要代理

### 2. 数据量

- 实盘数据流非常快，注意控制处理速度
- 建议使用异步处理，避免阻塞

### 3. 错误处理

- WebSocket连接可能中断，需要自动重连
- `BinanceWebSocketClient` 已实现自动重连

### 4. 测试环境

- 建议先在测试网测试
- 确认无误后再使用主网

### 5. API Key安全

- ⚠️ **不要将API key提交到Git**
- 使用环境变量或配置文件（已加入.gitignore）
- 定期轮换API key

## 故障排除

### 问题1：连接失败

**可能原因**：
- 网络问题
- 防火墙阻止
- 币安服务器维护

**解决方案**：
- 检查网络连接
- 使用代理（如需要）
- 查看币安公告

### 问题2：接收不到数据

**可能原因**：
- symbol名称错误
- 市场类型错误（spot vs futures）

**解决方案**：
- 确认symbol名称正确（如 "BTCUSDT"）
- 确认 `use_futures` 参数正确

### 问题3：API key认证失败

**可能原因**：
- API key无效或过期
- API key权限不足
- 环境变量未正确设置

**解决方案**：
- 检查API key是否正确
- 确认API key有数据读取权限
- 检查环境变量是否正确加载

## 相关文件

- `src/live_data_stream/websocket_client.py` - BinanceWebSocketClient实现
- `src/live_data_stream/nautilus_integration.py` - Nautilus Trader集成
- `tests/live_data_stream/test_live_binance_ws.py` - 实盘测试脚本
- `config/local/binance_testnet.env` - 测试网API key配置（需要创建）
- `config/local/binance_mainnet.env` - 主网API key配置（需要创建）

## 下一步

1. 配置API key（如使用Nautilus Trader）
2. 运行测试验证连接
3. 集成到实际交易系统
4. 监控和日志记录
