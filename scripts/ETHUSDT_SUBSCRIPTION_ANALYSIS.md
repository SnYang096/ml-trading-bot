# ETHUSDT订阅问题分析

## 问题描述

主网订阅测试中，ETHUSDT订阅成功，但在2分钟和5分钟的测试中都没有收到任何tick数据。

## 分析结果

### 1. 测试网独立WebSocket ✅

**结论**: 测试网独立WebSocket客户端**能够正常收到ticks**

- 测试结果: 37.1秒内收到127条tick
  - BTCUSDT: 65条 (1.75 ticks/秒)
  - ETHUSDT: 62条 (1.67 ticks/秒)
- 测试脚本: `scripts/test_binance_testnet_websocket.py`
- 结论: 测试网数据流正常，问题在于Nautilus Trader的使用方式

### 2. 主网ETHUSDT订阅问题分析

#### 订阅状态
- ✅ 订阅命令已发送: `SubscribeTradeTicks(instrument_id=ETHUSDT-PERP.BINANCE)`
- ✅ 订阅成功: `Subscribed ETHUSDT-PERP.BINANCE trades`
- ❌ **未收到tick数据**: 2分钟和5分钟测试中都是0条tick

#### 可能的原因

1. **代码逻辑问题** (已修复)
   - 原代码使用 `self.symbol_map.get(instrument_id)` 直接查找
   - 如果InstrumentId对象比较有问题，可能导致查找失败
   - **修复**: 添加了备用查找逻辑，包括字符串匹配和直接提取symbol

2. **数据流问题**
   - 在测试时间段内，ETHUSDT可能确实没有交易（不太可能，ETHUSDT交易很活跃）
   - WebSocket连接可能有问题，虽然订阅成功但数据流中断

3. **InstrumentId对象比较问题**
   - Python中，自定义对象的`==`比较可能有问题
   - 即使两个InstrumentId对象表示同一个instrument，`==`可能返回False
   - **解决方案**: 使用字符串比较作为备用

### 3. 代码修复

已修复 `on_trade_tick` 方法，添加了多层查找逻辑：

1. 首先尝试直接查找: `self.symbol_map.get(instrument_id)`
2. 如果失败，尝试字符串匹配查找
3. 如果还是失败，从instrument_id直接提取symbol并验证

这样可以确保即使InstrumentId对象比较有问题，也能正确找到对应的symbol。

### 4. 测试结果

#### 第一次测试（2分钟）
- BTCUSDT: 1条tick ✅
- ETHUSDT: 0条tick ❌

#### 第二次测试（5分钟）
- BTCUSDT: 0条tick ❌
- ETHUSDT: 0条tick ❌

**注意**: 第二次测试中BTCUSDT也没有收到tick，这可能表明：
- WebSocket连接可能有问题
- 或者在这个时间段内确实没有交易

## 建议

1. **重新测试**
   - 使用修复后的代码重新运行测试
   - 建议测试时长至少5-10分钟，确保有足够的交易数据

2. **检查WebSocket连接**
   - 查看日志中是否有WebSocket连接成功的消息
   - 检查是否有连接中断或重连的情况

3. **监控订阅状态**
   - 使用 `_check_subscription_status` 方法检查订阅状态
   - 确认订阅是否真的建立成功

4. **如果仍然失败**
   - 考虑使用独立WebSocket客户端作为替代方案
   - 手动转换为Nautilus Trader的TradeTick对象
   - 通过`on_trade_tick`方法手动注入数据

## 相关文件

- 修复后的代码: `src/live_data_stream/live_test_strategy.py`
- 测试脚本: `scripts/run_live_test.py`
- 测试网WebSocket测试: `scripts/test_binance_testnet_websocket.py`
