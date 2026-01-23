# 订阅问题排查指南

## 问题现象

Nautilus Trader订阅trade ticks后，没有收到任何tick数据，订阅任务可能被取消。

## 可能的原因

### 1. WebSocket连接未建立
- **症状**: 数据客户端状态显示未连接
- **检查方法**: 
  - 查看日志中的"数据客户端状态"和"数据客户端已连接"信息
  - 检查WebSocket URL是否正确（testnet vs mainnet）
- **解决方法**: 
  - 确保API key和secret正确
  - 检查网络连接
  - 等待更长时间让连接建立

### 2. Instrument未正确加载
- **症状**: Instrument不在cache中
- **检查方法**: 
  - 查看日志中的"Instrument在cache中"信息
  - 检查instruments加载日志
- **解决方法**: 
  - 确保instrument_provider配置正确（load_ids或load_all=True）
  - 等待instruments加载完成后再订阅
  - 检查instrument ID格式是否正确（例如：BTCUSDT-PERP.BINANCE）

### 3. 订阅命令未正确传递
- **症状**: 订阅命令已发送，但DataClient未收到
- **检查方法**: 
  - 查看日志中的"订阅命令已发送"信息
  - 检查订阅状态（10秒后）
- **解决方法**: 
  - 确保在on_start中调用subscribe_trade_ticks
  - 检查client_id参数（可以为None，会自动推断）

### 4. 订阅任务被取消
- **症状**: 订阅任务创建后立即被取消
- **检查方法**: 
  - 查看Nautilus Trader内部日志
  - 检查订阅状态（10秒后）
- **解决方法**: 
  - 确保WebSocket连接已建立
  - 确保Instrument在cache中
  - 检查是否有其他错误导致任务取消

## 调试步骤

### 步骤1: 检查数据客户端连接
```python
# 在on_start中检查
data_client = trader.data_engine.get_client(BINANCE)
print(f"数据客户端状态: {data_client.state}")
print(f"数据客户端已连接: {data_client.is_connected}")
```

### 步骤2: 检查Instrument加载
```python
# 在on_start中检查
cached_instrument = cache.instrument(instrument_id)
if cached_instrument:
    print(f"Instrument已加载: {cached_instrument.id}")
else:
    print(f"Instrument未加载: {instrument_id}")
```

### 步骤3: 检查订阅状态
```python
# 在订阅后10秒检查
subscribed = data_client.subscribed_trade_ticks()
print(f"已订阅的trade ticks: {subscribed}")
```

### 步骤4: 查看详细日志
- 启用DEBUG级别日志
- 查看订阅命令的创建和传递过程
- 查看WebSocket连接状态

## 替代方案

如果订阅仍然失败，可以考虑：

1. **使用独立WebSocket客户端**
   - 使用`websockets`库直接连接Binance WebSocket
   - 手动转换为Nautilus Trader的`TradeTick`对象
   - 通过`on_trade_tick`方法手动注入数据

2. **检查Nautilus Trader版本**
   - 确保使用最新版本
   - 检查是否有已知的订阅bug

3. **联系Nautilus Trader社区**
   - 在GitHub上提交issue
   - 提供详细的日志和配置信息

## 相关文件

- `src/live_data_stream/live_test_strategy.py`: 策略实现
- `scripts/run_live_test.py`: 测试脚本
- `scripts/test_binance_testnet_websocket.py`: 独立WebSocket客户端测试
