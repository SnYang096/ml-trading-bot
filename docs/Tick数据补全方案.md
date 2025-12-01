# Tick 数据补全方案

## 一、问题分析

### 1.0 币安 Tick 数据获取方式

币安提供两种方式获取 tick 数据：

1. **实时数据流（WebSocket）**
   - ✅ 通过 WebSocket 订阅 `aggTrades` stream
   - ✅ 实时接收 tick 数据
   - ✅ 延迟低，适合实时交易
   - ⚠️ 连接中断会丢失数据

2. **历史数据下载**
   - ✅ 从币安官方数据下载页面 (https://data.binance.vision/) 下载
   - ✅ 按天/月打包成 ZIP 文件
   - ✅ 用于补全实时流中断后的缺失数据
   - ⚠️ 只能获取历史数据，无法实时获取

**本工具用于补全历史数据或实时流中断后的数据缺失。**

### 1.1 Tick 数据的特殊性

与 K线数据不同，tick 数据（订单流数据）有以下特点：

1. **数据量大**：每秒可能有数百甚至数千条 tick
2. **实时流可用**：币安提供 WebSocket 实时 tick 数据流（aggTrades stream）
3. **历史数据需下载**：历史 tick 数据只能从币安官方数据下载页面获取
4. **按天/月存储**：历史数据按天或月打包成 ZIP 文件

### 1.2 数据缺失场景

1. **实时流中断**：WebSocket 连接断开，丢失部分 tick
2. **系统重启**：程序重启，丢失停机期间的 tick
3. **历史数据缺失**：某些月份的数据文件未下载

### 1.3 币安数据下载页面

币安官方数据下载页面：https://data.binance.vision/

- **Spot 数据**：`data/spot/daily/aggTrades/{SYMBOL}/`
- **Futures 数据**：`data/futures/{um|cm}/daily/aggTrades/{SYMBOL}/`

文件命名格式：`{SYMBOL}-aggTrades-{YYYY}-{MM}-{DD}.zip`

## 二、解决方案

### 2.1 自动下载缺失数据

从币安官方数据下载页面自动下载缺失的 tick 数据：

1. **检测缺失**：检查本地 Parquet 文件，找出缺失的月份
2. **自动下载**：从币安官方页面下载 ZIP 文件
3. **自动转换**：解压 ZIP 并转换为 Parquet 格式
4. **集成到补全流程**：与实时数据管理器集成

### 2.2 实时流数据补全

对于实时流中断导致的缺失：

1. **检测缺失时间段**：通过时间戳连续性检测（WebSocket 流中断）
2. **下载对应日期**：从币安官方数据下载页面下载对应日期的历史数据
3. **提取缺失部分**：从下载的数据中提取缺失的时间段
4. **写入 QuestDB**：将补全的数据写入 QuestDB

**注意**：实时流通过 WebSocket 接收，历史数据通过本工具下载补全。

## 三、实现方案

### 3.1 数据下载器

```python
from src.data_tools.tick_data_downloader import BinanceTickDataDownloader

# 创建下载器
downloader = BinanceTickDataDownloader(
    symbol="BTCUSDT",
    market_type="futures",  # "spot" 或 "futures"
    contract_type="um",  # "um" (USDT-M) 或 "cm" (COIN-M)
    download_dir="data/downloads",
)

# 下载指定月份的数据
zip_path = downloader.download_file(year=2024, month=1)

# 转换为 Parquet
parquet_path = downloader.extract_and_convert_to_parquet(
    zip_path,
    output_dir=Path("data/parquet_data"),
)
```

### 3.2 Tick 数据补全器

```python
from src.data_tools.tick_data_downloader import TickDataGapFiller
from datetime import datetime, timedelta

# 创建补全器
tick_filler = TickDataGapFiller(
    symbol="BTCUSDT",
    market_type="futures",
    contract_type="um",
    download_dir="data/downloads",
    parquet_dir="data/parquet_data",
)

# 补全缺失的数据
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

files = tick_filler.fill_missing_data(start_date, end_date)
```

### 3.3 集成到实时数据管理器

```python
from src.data_tools.realtime_data_manager import RealtimeDataManager
from src.data_tools.tick_data_downloader import TickDataGapFiller

class RealtimeTickDataManager(RealtimeDataManager):
    """支持 Tick 数据补全的实时数据管理器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Tick 数据补全器
        self.tick_filler = TickDataGapFiller(
            symbol=self.symbol,
            market_type="futures",
            contract_type="um",
        )
    
    def fill_missing_ticks(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> int:
        """
        补全缺失的 tick 数据
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
        
        Returns:
            补全的数据条数
        """
        # 转换为 datetime
        start_date = start_time.to_pydatetime()
        end_date = end_time.to_pydatetime()
        
        # 下载并转换缺失的数据
        files = self.tick_filler.fill_missing_data(start_date, end_date)
        
        # 从 Parquet 加载数据并写入 QuestDB
        # （这里需要实现从 Parquet 读取并写入 QuestDB 的逻辑）
        
        return len(files)
```

## 四、使用流程

### 4.1 系统启动时补全

```python
def on_start(self):
    """策略启动时初始化"""
    # 1. 检查最近 30 天的数据完整性
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # 2. 下载缺失的数据
    tick_filler = TickDataGapFiller(
        symbol="BTCUSDT",
        market_type="futures",
        contract_type="um",
    )
    
    missing_periods = tick_filler.detect_missing_periods(start_date, end_date)
    if missing_periods:
        print(f"⚠️ 发现 {len(missing_periods)} 个月份的数据缺失")
        files = tick_filler.fill_missing_data(start_date, end_date)
        print(f"✅ 已下载 {len(files)} 个文件")
```

### 4.2 实时流中断后补全

```python
def on_tick_data_gap_detected(self, missing_start: pd.Timestamp, missing_end: pd.Timestamp):
    """检测到 tick 数据缺失时"""
    # 1. 下载缺失时间段的数据
    tick_filler = TickDataGapFiller(...)
    
    start_date = missing_start.to_pydatetime()
    end_date = missing_end.to_pydatetime()
    
    # 2. 下载并转换
    files = tick_filler.fill_missing_data(start_date, end_date)
    
    # 3. 从 Parquet 加载并写入 QuestDB
    for parquet_file in files:
        if parquet_file.suffix == ".parquet":
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
                    "symbol": self.symbol,
                })
```

### 4.3 定期检查

```python
def periodic_tick_data_check(self):
    """定期检查 tick 数据完整性"""
    # 检查最近 7 天的数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    missing_periods = self.tick_filler.detect_missing_periods(start_date, end_date)
    
    if missing_periods:
        print(f"⚠️ 定期检查发现 {len(missing_periods)} 个月份的数据缺失")
        self.tick_filler.fill_missing_data(start_date, end_date)
```

## 五、注意事项

### 5.1 下载限制

- ⚠️ 币安数据下载页面可能有频率限制
- ✅ 系统会自动控制下载频率（每次下载后等待 1 秒）
- ✅ 支持重试机制（默认 3 次）

### 5.2 数据格式

- ✅ ZIP 文件包含 CSV 格式的 tick 数据
- ✅ 自动转换为 Parquet 格式（更高效）
- ✅ 保留原始字段：timestamp, price, volume, side

### 5.3 存储策略

- ✅ 下载的 ZIP 文件保存在 `data/downloads/`
- ✅ 转换的 Parquet 文件保存在 `data/parquet_data/`
- ✅ 建议定期清理旧的 ZIP 文件（保留 Parquet）

### 5.4 性能考虑

- ⚠️ 下载大文件可能较慢（每月数据可能几百 MB）
- ✅ 支持断点续传（检查文件是否已存在）
- ✅ 异步下载（不阻塞主流程）

## 六、最佳实践

### 6.1 推荐配置

```python
# 1. 系统启动时检查并补全
tick_filler = TickDataGapFiller(
    symbol="BTCUSDT",
    market_type="futures",
    contract_type="um",
)

# 检查最近 30 天
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
tick_filler.fill_missing_data(start_date, end_date)

# 2. 定期检查（每天一次）
# 在后台任务中执行
```

### 6.2 数据备份

```python
# 定期备份 Parquet 文件
def backup_tick_data():
    import shutil
    from datetime import datetime
    
    source_dir = Path("data/parquet_data")
    backup_dir = Path(f"data/backup/tick_{datetime.now().strftime('%Y%m%d')}")
    
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    for parquet_file in source_dir.glob("BTCUSDT_*.parquet"):
        shutil.copy2(parquet_file, backup_dir / parquet_file.name)
```

### 6.3 监控和告警

```python
# 监控数据完整性
def check_tick_data_integrity():
    missing_periods = tick_filler.detect_missing_periods(
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now(),
    )
    
    if len(missing_periods) > 0:
        send_alert(f"Tick 数据缺失告警: {len(missing_periods)} 个月份")
```

## 七、总结

### 7.1 核心功能

✅ **自动检测缺失**：检查本地 Parquet 文件，找出缺失的月份  
✅ **自动下载**：从币安官方页面下载 ZIP 文件  
✅ **自动转换**：解压 ZIP 并转换为 Parquet 格式  
✅ **集成补全**：与实时数据管理器集成，支持自动补全  

### 7.2 关键优势

1. **自动化**：无需手动下载和转换
2. **可靠性**：支持重试和断点续传
3. **高效**：转换为 Parquet 格式，查询更快
4. **灵活**：支持 Spot 和 Futures 数据

### 7.3 使用建议

1. **启动时检查**：系统启动时检查并补全最近 30 天的数据
2. **定期检查**：每天检查一次数据完整性
3. **及时补全**：发现缺失后立即补全
4. **定期备份**：定期备份 Parquet 文件

