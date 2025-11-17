# 实盘因子计算：factor-compute vs 直接调用代码

## 你的观点是对的！

如果你已经有自己的 Python 代码，确实**可以直接调用特征工程代码**，`factor-compute` 只是一个封装工具。

---

## 两种方式对比

### 方式 1：直接调用代码（推荐，如果你有代码）

```python
# 你自己的实盘代码
import json
from src.data_tools.comprehensive_feature_engineering import ComprehensiveFeatureEngineer
from src.data_tools.data_loader import MarketDataLoader

# 1. 加载 top factors
with open("results/dim_compare/.../top_factors.json") as f:
    top_factors_data = json.load(f)
    top_factors = [item['name'] if isinstance(item, dict) else item 
                   for item in top_factors_data.get('top_factors', top_factors_data)]

# 2. 加载数据
loader = MarketDataLoader("data/parquet_data")
df = loader.load_data(symbol="BTCUSDT", start_date="2024-01-01", end_date="2024-12-31")

# 3. 计算因子（只计算需要的）
engineer = ComprehensiveFeatureEngineer(feature_types="comprehensive")
df_features = engineer.engineer_all_features(
    df,
    fit=False,  # 实盘不需要 fit
    required_features=set(top_factors)  # 只计算需要的因子
)

# 4. 使用因子
factors = df_features[top_factors]
```

**优势**：
- ✅ 完全控制，可以自定义逻辑
- ✅ 不需要命令行工具
- ✅ 可以直接集成到你的实盘系统
- ✅ 更灵活，可以添加缓存、错误处理等

---

### 方式 2：使用 factor-compute（适合脚本化场景）

```bash
# 1. 从 top_factors.json 提取因子列表
top_factors=$(cat results/dim_compare/.../top_factors.json | \
    jq -r '.top_factors[] | if type=="object" then .name else . end' | \
    tr '\n' ' ')

# 2. 调用 factor-compute
make factor-compute \
    FACTOR_COMPUTE_FACTORS="$top_factors" \
    FACTOR_COMPUTE_SYMBOL=BTCUSDT \
    FACTOR_COMPUTE_OUTPUT=realtime_factors.csv
```

**优势**：
- ✅ 命令行工具，方便脚本化
- ✅ 统一的输出格式处理
- ✅ 不需要写 Python 代码
- ✅ 适合一次性任务或批处理

**劣势**：
- ❌ 需要命令行环境
- ❌ 灵活性较低
- ❌ 不适合集成到复杂的实盘系统

---

## 实际使用建议

### 如果你的实盘系统是 Python 代码

**直接调用代码**（推荐）：

```python
# 实盘因子计算模块
class RealtimeFactorComputer:
    def __init__(self, top_factors_path: str):
        # 加载 top factors（只需要加载一次）
        with open(top_factors_path) as f:
            data = json.load(f)
            self.top_factors = [
                item['name'] if isinstance(item, dict) else item
                for item in data.get('top_factors', data)
            ]
        
        # 创建特征工程器（只需要创建一次）
        self.engineer = ComprehensiveFeatureEngineer(
            feature_types="comprehensive"
        )
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算因子（实盘调用）"""
        return self.engineer.engineer_all_features(
            df,
            fit=False,  # 实盘不需要 fit
            required_features=set(self.top_factors)
        )[self.top_factors]  # 只返回需要的因子

# 使用
computer = RealtimeFactorComputer("results/.../top_factors.json")
factors = computer.compute(latest_data)
```

### 如果你需要脚本化或批处理

**使用 factor-compute**：

```bash
# 批处理脚本
for symbol in BTCUSDT ETHUSDT SOLUSDT; do
    make factor-compute \
        FACTOR_COMPUTE_FACTORS="$(cat top_factors.json | jq -r '...')" \
        FACTOR_COMPUTE_SYMBOL=$symbol \
        FACTOR_COMPUTE_OUTPUT="factors/${symbol}_$(date +%Y%m%d).csv"
done
```

---

## 总结

### factor-compute 的价值

1. **命令行工具**：适合脚本化和批处理
2. **快速原型**：不需要写代码就能测试
3. **统一接口**：标准化的输入输出格式

### 但如果你已经有代码

**直接调用特征工程代码更简单**：
- 不需要命令行工具
- 更灵活
- 更容易集成
- 性能相同（底层都是调用 `engineer_all_features`）

### 建议

- **实盘系统** → 直接调用代码（方式 1）
- **脚本/批处理** → 使用 factor-compute（方式 2）
- **快速测试** → 使用 factor-compute（方式 2）

---

## 代码示例：直接调用（推荐）

```python
# 实盘因子计算（完整示例）
import json
import pandas as pd
from pathlib import Path
from src.data_tools.comprehensive_feature_engineering import ComprehensiveFeatureEngineer
from src.data_tools.data_loader import MarketDataLoader

def load_top_factors(json_path: str) -> list[str]:
    """加载 top factors"""
    with open(json_path) as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        if 'top_factors' in data:
            return [item.get('name', item) if isinstance(item, dict) else item 
                   for item in data['top_factors']]
        elif 'features' in data:
            return data['features']
    elif isinstance(data, list):
        return data
    
    return []

def compute_realtime_factors(
    df: pd.DataFrame,
    top_factors: list[str],
    feature_type: str = "comprehensive"
) -> pd.DataFrame:
    """计算实盘因子"""
    engineer = ComprehensiveFeatureEngineer(feature_types=feature_type)
    df_features = engineer.engineer_all_features(
        df,
        fit=False,  # 实盘不需要 fit
        required_features=set(top_factors)
    )
    return df_features[top_factors]

# 使用
top_factors = load_top_factors("results/dim_compare/.../top_factors.json")
loader = MarketDataLoader("data/parquet_data")
df = loader.load_data(symbol="BTCUSDT", start_date="2024-01-01", end_date="2024-12-31")
factors = compute_realtime_factors(df, top_factors)
```

**这就是全部了！** 不需要 `factor-compute` 工具。

