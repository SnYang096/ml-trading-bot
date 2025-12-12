# VPIN vs Trade Clustering 详细对比分析

## 📊 一、核心概念

### VPIN (Volume-Weighted Price Imbalance)
**定义**：基于**交易量**的买卖不平衡度指标

**核心思想**：
- 按**固定交易量**（或 USD 价值）将 tick 数据分组为 "buckets"
- 每个 bucket 填满后，计算该 bucket 内的买卖不平衡度
- 最终通过滚动平均得到 VPIN 序列

### Trade Clustering
**定义**：基于**成交顺序**的连续同向交易聚集性指标

**核心思想**：
- 识别连续同方向的成交序列（称为 "run"）
- 在滑动窗口内统计这些 runs 的特征（最大长度、平均长度、数量等）
- 捕捉市场微观结构中的订单流模式

---

## 🔄 二、相同点（Similarities）

### 1. **数据源相同**
✅ 都使用 **tick 数据**（逐笔成交数据）
- 需要 `timestamp`（时间戳）
- 需要 `side`（买卖方向：1=buy, -1=sell）
- 需要 `volume`（交易量，VPIN 必需，Trade Clustering 可选）

### 2. **都支持流式处理**
✅ 都支持**跨批次连续性**（cross-batch continuity）
- **VPIN**：通过 `initial_state` 传递未完成的 bucket 状态
  ```python
  initial_state = {
      "current_buy": float,      # 当前 bucket 的买入量
      "current_sell": float,     # 当前 bucket 的卖出量
      "filled_value": float      # 当前 bucket 已填充的总量
  }
  ```

- **Trade Clustering**：通过 `initial_state` 传递窗口状态
  ```python
  initial_state = {
      "current_run_side": int,           # 当前 run 的方向
      "current_run_length": int,         # 当前 run 的长度
      "window_runs": deque,              # 窗口内的 runs
      "window_total_ticks": int,         # 窗口内总 tick 数
      "buy_runs_in_window": deque,       # 窗口内 buy runs
      "sell_runs_in_window": deque       # 窗口内 sell runs
  }
  ```

### 3. **都支持按月缓存**
✅ 都实现了**按月缓存机制**，提高计算效率
- 支持标准缓存（只存 final_state）
- 支持状态缓存（存完整结果 + final_state）
- 缓存键独立于 `start/end` 时间，可跨时间窗口复用

### 4. **都关注买卖不平衡**
✅ 都试图捕捉**买卖力量的不平衡**
- **VPIN**：直接计算 `|buy_volume - sell_volume| / bucket_volume`
- **Trade Clustering**：通过 `imbalance_ratio = (buy_run_count - sell_run_count) / total_runs` 间接反映

### 5. **都输出时间序列特征**
✅ 都输出**时间序列特征**，可以对齐到 K 线数据
- **VPIN**：输出 `(timestamp, vpin_value)` 序列
- **Trade Clustering**：输出包含多个特征的 DataFrame（每个 tick 一行）

---

## 🔀 三、不同点（Differences）

### 1. **分组方式（Grouping Method）**

| 维度 | VPIN | Trade Clustering |
|------|------|-----------------|
| **分组依据** | **交易量**（volume-based） | **成交顺序**（sequence-based） |
| **分组单位** | 固定交易量的 bucket（如 1000 个币或 10 万美元） | 连续同方向的 run（如连续 10 笔都是 buy） |
| **分组特点** | 每个 bucket 的交易量相同，但时间跨度可能不同 | 每个 run 的时间跨度相同（单笔成交），但交易量可能不同 |

**示例**：
```
VPIN:
  Bucket 1: [1000 BTC 的交易] → 计算不平衡度
  Bucket 2: [1000 BTC 的交易] → 计算不平衡度
  （每个 bucket 的交易量相同，但可能跨越不同时间）

Trade Clustering:
  Run 1: [buy, buy, buy, buy, buy] → 5 笔连续买入
  Run 2: [sell, sell, sell] → 3 笔连续卖出
  （每个 run 的成交顺序相同，但交易量可能不同）
```

### 2. **时间维度（Time Dimension）**

| 维度 | VPIN | Trade Clustering |
|------|------|-----------------|
| **时间敏感性** | **低**：bucket 按交易量填充，时间跨度可变 | **高**：run 按成交顺序识别，时间跨度固定（单笔成交） |
| **时间窗口** | 无固定时间窗口，bucket 完成时间取决于流动性 | 有固定滑动窗口（如最近 100 笔成交） |
| **时间对齐** | bucket 完成时输出一个值 | 每个 tick 都输出一个值 |

**示例**：
```
VPIN:
  高流动性时段：1000 BTC 的 bucket 可能在 1 分钟内完成
  低流动性时段：1000 BTC 的 bucket 可能需要 1 小时完成
  → 时间跨度不固定

Trade Clustering:
  窗口大小 = 100 笔成交
  高流动性时段：100 笔 ≈ 几毫秒
  低流动性时段：100 笔 ≈ 数小时
  → 时间跨度不固定（但窗口大小固定为 tick 数）
```

### 3. **计算粒度（Calculation Granularity）**

| 维度 | VPIN | Trade Clustering |
|------|------|-----------------|
| **输出频率** | **低**：每个 bucket 完成时输出一次 | **高**：每个 tick 都输出一次 |
| **输出密度** | 稀疏（取决于 bucket 完成频率） | 密集（每个 tick 一行） |
| **特征数量** | 少（主要是 VPIN 值及其衍生特征） | 多（8+ 个基础特征，37+ 个衍生特征） |

**示例**：
```
VPIN:
  如果 bucket_volume = 1000 BTC
  一天交易 10000 BTC → 约 10 个 VPIN 值
  → 输出稀疏

Trade Clustering:
  一天有 100000 笔成交 → 100000 行特征
  → 输出密集
```

### 4. **关注点（Focus）**

| 维度 | VPIN | Trade Clustering |
|------|------|-----------------|
| **主要关注** | **交易量的不平衡** | **成交顺序的模式** |
| **捕捉信息** | 买卖双方的总交易量差异 | 连续同向交易的聚集性 |
| **应用场景** | 衡量整体买卖压力 | 识别订单流模式、冰山单、算法交易痕迹 |

**示例**：
```
VPIN:
  场景：一个大单（1000 BTC）被拆分成 100 个小单执行
  → VPIN 关注的是：这 1000 BTC 中，有多少是 buy，多少是 sell
  → 结果：如果 600 BTC buy + 400 BTC sell → VPIN = 0.2

Trade Clustering:
  场景：一个大单（1000 BTC）被拆分成 100 个小单执行
  → Trade Clustering 关注的是：这 100 笔成交的顺序模式
  → 结果：如果连续 50 笔都是 buy → max_buy_run = 50
```

### 5. **计算复杂度（Computational Complexity）**

| 维度 | VPIN | Trade Clustering |
|------|------|-----------------|
| **时间复杂度** | O(N) - 线性扫描 | O(N) - 线性扫描（使用增量更新） |
| **空间复杂度** | O(1) - 只维护当前 bucket 状态 | O(W) - 维护滑动窗口（W = window_size） |
| **计算开销** | 低（只需累加买卖量） | 中（需要维护窗口和统计量） |

### 6. **特征类型（Feature Types）**

| 维度 | VPIN | Trade Clustering |
|------|------|-----------------|
| **基础特征** | VPIN 值（不平衡度） | max_buy_run, max_sell_run, avg_buy_run, avg_sell_run, buy_run_count, sell_run_count, imbalance_ratio, directional_entropy |
| **衍生特征** | VPIN 的滚动统计（mean, max, std, zscore, skewness, trend 等） | 各种滚动统计、zscore、MA 等（共 37+ 个特征） |
| **特征维度** | 低（约 30 个特征） | 高（约 37 个特征） |

### 7. **对流动性的敏感性（Liquidity Sensitivity）**

| 维度 | VPIN | Trade Clustering |
|------|------|-----------------|
| **流动性影响** | **中等**：bucket 完成时间受流动性影响，但 bucket 大小固定 | **高**：窗口大小固定为 tick 数，流动性直接影响时间跨度 |
| **跨时段可比性** | **高**：bucket 大小固定，不同时段可比 | **低**：窗口大小固定为 tick 数，不同时段时间跨度差异大 |
| **跨品种可比性** | **高**：可以使用 USD 价值统一 bucket 大小 | **低**：窗口大小固定为 tick 数，不同品种时间跨度差异大 |

---

## 🎯 四、互补性（Complementarity）

### 为什么需要两者？

1. **VPIN** 关注**总量不平衡**，适合：
   - 衡量整体买卖压力
   - 识别大单执行
   - 捕捉市场情绪

2. **Trade Clustering** 关注**顺序模式**，适合：
   - 识别算法交易痕迹
   - 检测冰山单（iceberg orders）
   - 捕捉微观结构异常

### 组合使用示例

```
场景：一个大单被拆分成多个小单执行

VPIN 视角：
  - 如果 1000 BTC 的大单被拆成 100 个小单
  - VPIN 会显示：这 1000 BTC 中，买卖比例如何
  - 结果：VPIN = 0.3（表示买卖不平衡度为 30%）

Trade Clustering 视角：
  - 如果这 100 个小单中，连续 50 笔都是 buy
  - Trade Clustering 会显示：max_buy_run = 50
  - 结果：存在明显的买入聚集性

组合分析：
  - VPIN 高 + max_buy_run 高 → 可能是大单主动买入
  - VPIN 低 + max_buy_run 高 → 可能是算法交易（分散执行）
  - VPIN 高 + max_buy_run 低 → 可能是正常交易（买卖交替）
```

---

## 📈 五、实际应用建议

### 1. **特征选择**
- **VPIN**：适合作为**宏观**市场情绪指标
- **Trade Clustering**：适合作为**微观**订单流模式指标

### 2. **时间对齐**
- **VPIN**：对齐到 K 线时，使用 bucket 完成时间
- **Trade Clustering**：对齐到 K 线时，使用每个 tick 的时间（或聚合统计）

### 3. **特征工程**
- **VPIN**：可以计算滚动统计（mean, max, std, zscore, skewness, trend）
- **Trade Clustering**：可以计算各种滚动统计、zscore、MA 等

### 4. **模型训练**
- **VPIN**：适合作为**趋势**和**反转**策略的特征
- **Trade Clustering**：适合作为**微观结构**和**订单流**策略的特征

---

## 🔍 六、代码实现对比

### VPIN 核心逻辑
```python
# 按交易量填充 bucket
for tick in ticks:
    if side == 1:
        current_buy += tick_value
    else:
        current_sell += tick_value
    filled_value += tick_value
    
    # bucket 填满后，计算不平衡度
    if filled_value >= bucket_volume:
        imbalance = abs(current_buy - current_sell)
        vpin = imbalance / bucket_volume
        # 输出 VPIN 值
```

### Trade Clustering 核心逻辑
```python
# 识别连续同方向的 run
for tick in ticks:
    if side == current_run_side:
        current_run_length += 1  # 同向，增加长度
    else:
        # 方向改变，结束当前 run，开始新 run
        window_runs.append((current_run_side, current_run_length))
        current_run_side = side
        current_run_length = 1
    
    # 在滑动窗口内统计 runs
    # 计算 max_buy_run, max_sell_run, imbalance_ratio 等
    # 每个 tick 都输出特征
```

---

## 📝 七、总结

| 维度 | VPIN | Trade Clustering |
|------|------|-----------------|
| **分组方式** | 按交易量（volume-based） | 按成交顺序（sequence-based） |
| **时间维度** | 时间跨度可变 | 时间跨度固定（但窗口大小固定为 tick 数） |
| **计算粒度** | 稀疏（bucket 完成时输出） | 密集（每个 tick 输出） |
| **关注点** | 交易量的不平衡 | 成交顺序的模式 |
| **特征数量** | 少（约 30 个） | 多（约 37 个） |
| **流动性敏感性** | 中等 | 高 |
| **跨时段可比性** | 高 | 低 |
| **应用场景** | 宏观市场情绪 | 微观订单流模式 |

**结论**：VPIN 和 Trade Clustering 是**互补**的指标，从不同角度捕捉市场信息，组合使用可以提供更全面的市场洞察。

