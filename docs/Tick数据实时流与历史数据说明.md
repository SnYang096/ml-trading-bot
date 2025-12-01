# Tick 数据：实时流与历史数据说明

## 一、数据获取方式对比

### 1.1 实时数据流（WebSocket）

**币安提供 WebSocket 实时 tick 数据流**

```python
# 币安 WebSocket aggTrades stream 示例
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    # data 包含：
    # - e: "aggTrade"
    # - s: symbol (如 "BTCUSDT")
    # - p: price
    # - q: quantity
    # - m: is buyer maker
    # - T: trade time
    print(f"Tick: {data['p']} @ {data['q']}")

# 订阅 aggTrades stream
ws = websocket.WebSocketApp(
    "wss://fstream.binance.com/ws/btcusdt@aggTrade",
    on_message=on_message
)
ws.run_forever()
```

**特点**：
- ✅ 实时接收，延迟低（毫秒级）
- ✅ 适合实时交易和特征计算
- ⚠️ 连接中断会丢失数据
- ⚠️ 无法获取历史数据

### 1.2 历史数据下载

**币安官方数据下载页面：https://data.binance.vision/**

- **Spot 数据**：`data/spot/daily/aggTrades/{SYMBOL}/`
- **Futures 数据**：`data/futures/{um|cm}/daily/aggTrades/{SYMBOL}/`

**特点**：
- ✅ 可以获取任意历史时间段的数据
- ✅ 用于补全实时流中断后的缺失数据
- ⚠️ 只能获取历史数据，无法实时获取
- ⚠️ 数据按天/月打包，需要下载和解压

## 二、使用场景

### 2.1 实时流（WebSocket）

**适用场景**：
1. **实时交易**：需要实时 tick 数据做决策
2. **特征计算**：实时计算订单流特征（VPIN、CVD 等）
3. **实时监控**：监控市场实时变化

**实现方式**：
```python
# 在实时交易引擎中订阅 WebSocket
async def subscribe_tick_stream(self, symbol: str):
    """订阅 tick 数据流"""
    stream = f"{symbol.lower()}@aggTrade"
    ws_url = f"wss://fstream.binance.com/ws/{stream}"
    
    # 使用 WebSocket 客户端订阅
    # 收到数据后调用 on_tick 处理
    await self.ws_client.subscribe(ws_url, self.on_tick)
```

### 2.2 历史数据下载（本工具）

**适用场景**：
1. **数据补全**：实时流中断后，补全缺失的数据
2. **系统启动**：系统重启后，补全停机期间的数据
3. **历史回测**：需要历史 tick 数据进行回测
4. **数据备份**：定期下载历史数据作为备份

**实现方式**：
```python
from src.data_tools.tick_data_downloader import TickDataGapFiller
from datetime import datetime, timedelta

# 创建补全器
tick_filler = TickDataGapFiller(
    symbol="BTCUSDT",
    market_type="futures",
    contract_type="um",
)

# 补全缺失的数据
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
files = tick_filler.fill_missing_data(start_date, end_date)
```

## 三、数据补全策略

### 3.1 实时流中断后的补全

**流程**：
1. **检测中断**：检测到 WebSocket 连接断开或数据缺失
2. **确定缺失时间段**：通过时间戳连续性检测
3. **下载历史数据**：从币安官方页面下载对应时间段的数据
4. **提取缺失部分**：从下载的数据中提取缺失的时间段
5. **写入存储**：将补全的数据写入 QuestDB 和 Parquet

**示例**：
```python
def on_websocket_disconnect(self):
    """WebSocket 断开时的处理"""
    # 1. 记录断开时间
    disconnect_time = datetime.now()
    
    # 2. 检测缺失的时间段
    last_tick_time = self.get_last_tick_time()
    missing_start = last_tick_time
    missing_end = disconnect_time
    
    # 3. 下载并补全
    tick_filler = TickDataGapFiller(...)
    files = tick_filler.fill_missing_data(missing_start, missing_end)
    
    # 4. 从 Parquet 加载并写入 QuestDB
    for parquet_file in files:
        df = pd.read_parquet(parquet_file)
        # 过滤出缺失的时间段
        mask = (df["timestamp"] >= missing_start) & (df["timestamp"] <= missing_end)
        missing_df = df[mask]
        
        # 写入 QuestDB
        for _, row in missing_df.iterrows():
            self.questdb_writer.write_tick({
                "timestamp": int(row["timestamp"].timestamp() * 1000),
                "price": row["price"],
                "size": row["volume"],
                "side": "buy" if row["side"] == 1 else "sell",
            })
```

### 3.2 系统启动时的补全

**流程**：
1. **检查最近数据**：检查最近 N 天的数据完整性
2. **检测缺失月份**：找出缺失的月份
3. **下载缺失数据**：从币安官方页面下载
4. **转换为 Parquet**：解压 ZIP 并转换为 Parquet
5. **加载到内存**：加载数据供特征计算使用

**示例**：
```python
def on_start(self):
    """系统启动时初始化"""
    # 1. 检查最近 30 天的数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # 2. 检测并补全缺失数据
    tick_filler = TickDataGapFiller(...)
    missing_periods = tick_filler.detect_missing_periods(start_date, end_date)
    
    if missing_periods:
        print(f"⚠️ 发现 {len(missing_periods)} 个月份的数据缺失")
        files = tick_filler.fill_missing_data(start_date, end_date)
        print(f"✅ 已补全 {len(files)} 个文件")
```

## 四、最佳实践

### 4.1 实时流 + 历史数据补全

**推荐架构**：
1. **主流程**：使用 WebSocket 实时接收 tick 数据
2. **补全流程**：检测到缺失后，自动下载历史数据补全
3. **定期检查**：定期检查数据完整性，及时补全

```python
class RealtimeTickManager:
    """实时 Tick 数据管理器"""
    
    def __init__(self):
        # WebSocket 客户端（实时流）
        self.ws_client = WebSocketClient()
        
        # 数据补全器（历史数据）
        self.tick_filler = TickDataGapFiller(...)
        
        # QuestDB 写入器
        self.questdb_writer = QuestDBWriter()
    
    async def on_tick(self, tick_data):
        """处理实时 tick 数据"""
        # 1. 写入 QuestDB
        self.questdb_writer.write_tick(tick_data)
        
        # 2. 检测缺失（与前一条数据对比）
        if self.detect_gap(tick_data):
            # 3. 自动补全
            await self.fill_gap()
    
    async def fill_gap(self):
        """补全缺失数据"""
        missing_start, missing_end = self.get_missing_period()
        
        # 下载历史数据
        files = self.tick_filler.fill_missing_data(missing_start, missing_end)
        
        # 写入 QuestDB
        for file in files:
            df = pd.read_parquet(file)
            for _, row in df.iterrows():
                self.questdb_writer.write_tick(row.to_dict())
```

### 4.2 数据存储策略

**三层存储**：
1. **内存**：最近 N 条 tick（用于实时计算）
2. **QuestDB**：实时和历史 tick 数据（用于查询和分析）
3. **Parquet**：历史 tick 数据（用于回测和备份）

```python
# 内存：滑动窗口
self.memory_buffer = deque(maxlen=10000)  # 最近 10000 条

# QuestDB：实时和历史数据
self.questdb_writer.write_tick(tick_data)

# Parquet：定期备份
if len(self.memory_buffer) >= 10000:
    df = pd.DataFrame(list(self.memory_buffer))
    df.to_parquet(f"data/backup/ticks_{datetime.now().strftime('%Y%m%d')}.parquet")
```

## 五、总结

### 5.1 数据获取方式

| 方式 | 实时性 | 历史数据 | 适用场景 |
|------|--------|----------|----------|
| **WebSocket 实时流** | ✅ 实时 | ❌ 无 | 实时交易、特征计算 |
| **历史数据下载** | ❌ 非实时 | ✅ 有 | 数据补全、回测、备份 |

### 5.2 推荐方案

1. **实时流为主**：使用 WebSocket 接收实时 tick 数据
2. **历史数据补全**：检测到缺失后，自动下载历史数据补全
3. **定期检查**：定期检查数据完整性，及时补全缺失数据

### 5.3 关键优势

✅ **数据完整性**：实时流 + 历史数据补全，保证数据连续性  
✅ **自动化**：自动检测和补全，无需手动干预  
✅ **可靠性**：多层存储，数据不丢失  
✅ **灵活性**：支持实时和历史两种数据源  

