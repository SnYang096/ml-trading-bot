# WebSocket重连机制使用说明

## 概述

本项目实现了健壮的WebSocket断开重连机制，包括：

- **指数退避策略**: 重连延迟逐渐增加（5s → 10s → 20s → 40s，最大60s）
- **最大重连次数**: 可配置的最大重连尝试次数（默认：无限）
- **连接状态监控**: 实时跟踪连接状态（connected/disconnected/reconnecting）
- **重连后重新订阅**: 确保重连后自动重新订阅所有流
- **重连统计**: 记录重连次数、最后重连时间等
- **重连回调**: 支持重连成功/失败的回调通知
- **健康检查**: 定期检查连接健康状态（心跳检测）

## 核心组件

### 1. ReconnectionManager (重连管理器)

位置: `src/live_data_stream/reconnection_manager.py`

功能：
- 指数退避策略
- 重连次数限制
- 连接状态管理
- 重连统计

### 2. ConnectionMonitor (连接监控器)

位置: `src/live_data_stream/connection_monitor.py`

功能：
- 心跳检测
- 超时检测
- 健康状态评估

### 3. 改进的WebSocket客户端

- `BinanceWebSocketClient`: 已集成新的重连机制
- `AlltickWebsocketClient`: 已集成新的重连机制
- `HyperliquidDataCollector`: 已修复重连后重新订阅问题

## 使用示例

### BinanceWebSocketClient

```python
from src.live_data_stream.websocket_client import BinanceWebSocketClient
from src.live_data_stream.reconnection_manager import ReconnectionConfig
import asyncio

async def main():
    # 创建客户端（使用默认重连配置）
    client = BinanceWebSocketClient(
        symbols=["BTCUSDT", "ETHUSDT"],
        use_futures=True,
    )
    
    # 或者使用自定义重连配置
    client = BinanceWebSocketClient(
        symbols=["BTCUSDT"],
        reconnect_config=ReconnectionConfig(
            initial_delay=5.0,      # 初始延迟5秒
            max_delay=60.0,         # 最大延迟60秒
            backoff_multiplier=2.0, # 退避倍数
            max_retries=10,         # 最多重试10次
        ),
        heartbeat_timeout=60.0,     # 心跳超时60秒
        health_check_interval=30.0, # 健康检查间隔30秒
    )
    
    # 添加数据回调
    def on_tick(tick):
        print(f"收到tick: {tick.symbol} @ {tick.price}")
    
    client.add_callback(on_tick)
    
    # 添加重连成功回调
    def on_reconnect():
        print("重连成功！")
    
    client.add_reconnect_callback(on_reconnect)
    
    # 运行客户端
    stop_event = asyncio.Event()
    await client.run(stop_event)

asyncio.run(main())
```

### 获取统计信息

```python
# 获取重连统计
reconnect_stats = client.get_reconnect_stats()
print(f"总重连次数: {reconnect_stats['total_reconnects']}")
print(f"成功重连: {reconnect_stats['successful_reconnects']}")
print(f"失败重连: {reconnect_stats['failed_reconnects']}")
print(f"当前延迟: {reconnect_stats['current_delay']:.2f}s")

# 获取健康状态
health = client.get_health_status()
print(f"健康状态: {health['status']}")
print(f"消息数: {health['message_count']}")
print(f"心跳丢失次数: {health['heartbeat_missed_count']}")
```

## 测试

运行测试脚本模拟各种断开场景：

```bash
# 运行所有测试场景
python scripts/test_websocket_reconnection_scenarios.py --symbols BTCUSDT --duration 60

# 指定多个交易对
python scripts/test_websocket_reconnection_scenarios.py --symbols BTCUSDT,ETHUSDT --duration 120
```

测试场景包括：
1. **正常断开重连**: 模拟网络中断 → 自动重连 → 验证数据恢复
2. **频繁断开重连**: 连续多次断开 → 验证指数退避生效
3. **最大重连次数**: 断开超过最大重连次数 → 验证停止重连逻辑
4. **数据完整性**: 断开前记录数据 → 重连后验证数据连续性

## 配置参数

### ReconnectionConfig

```python
ReconnectionConfig(
    initial_delay=5.0,           # 初始重连延迟（秒）
    max_delay=60.0,              # 最大重连延迟（秒）
    backoff_multiplier=2.0,      # 退避倍数
    max_retries=None,            # 最大重连次数（None=无限）
    reset_after_success=True,   # 成功后重置延迟
    jitter=True,                 # 是否添加随机抖动
)
```

### ConnectionMonitor

```python
ConnectionMonitor(
    heartbeat_timeout=60.0,      # 心跳超时（秒）
    health_check_interval=30.0,  # 健康检查间隔（秒）
)
```

## 状态说明

### ConnectionState (连接状态)

- `DISCONNECTED`: 已断开
- `CONNECTING`: 正在连接
- `CONNECTED`: 已连接
- `RECONNECTING`: 正在重连
- `FAILED`: 连接失败（达到最大重连次数）

### HealthStatus (健康状态)

- `HEALTHY`: 健康
- `DEGRADED`: 降级（延迟较高但仍在工作）
- `UNHEALTHY`: 不健康（心跳超时或连接异常）
- `DEAD`: 已断开

## 注意事项

1. **重连延迟**: 使用指数退避策略，避免频繁重连对服务器造成压力
2. **最大重连次数**: 建议设置合理的最大重连次数，避免无限重连
3. **心跳检测**: 确保心跳超时时间设置合理，既能及时发现问题，又不会误判
4. **重连后重新订阅**: HyperliquidDataCollector已自动处理，其他客户端需要手动处理

## 相关文件

- `src/live_data_stream/reconnection_manager.py` - 重连管理器
- `src/live_data_stream/connection_monitor.py` - 连接监控器
- `src/live_data_stream/websocket_client.py` - Binance WebSocket客户端
- `src/live_data_stream/alltick_ws.py` - Alltick WebSocket客户端
- `src/data_tools/hyperliquid_data.py` - Hyperliquid数据收集器
- `scripts/test_websocket_reconnection_scenarios.py` - 测试脚本
