# 当前架构 vs Qlib 对比分析

本文档对比当前 ML Trading Bot 的特征管道架构与 Microsoft Qlib 的设计，并提出统一研究/回测/实盘的采用计划。

## 1. 架构对比

### 1.1 数据层

| 组件 | 当前架构 | Qlib |
|------|---------|------|
| 数据加载 | `DataHandler.load_ohlcv()` | `D.features()` / `DataHandler` |
| 数据格式 | Parquet 按月分片 | 自定义二进制格式 (`.bin`) |
| 时间对齐 | 手动 resample | 自动对齐到 calendar |
| 多资产 | `_symbol` 列 + concat | 原生支持 (instrument 维度) |

**当前优势**：
- Parquet 格式通用，易于调试和外部工具读取
- 灵活的 resample 规则

**Qlib 优势**：
- 更高效的二进制存储
- 原生多资产支持，无需 concat

### 1.2 特征层

| 组件 | 当前架构 | Qlib |
|------|---------|------|
| 特征定义 | YAML + Python 函数 | Expression DSL (`$close/Ref($close,1)-1`) |
| 依赖管理 | `feature_dependencies.yaml` DAG | Expression 自动解析依赖 |
| 计算引擎 | `ParallelFeatureComputer` | `ExpressionProvider` |
| 缓存 | 月度 Parquet + Memory cache | `DatasetCache` |

**当前优势**：
- YAML 配置可读性强，易于版本控制
- 支持复杂特征（VPIN、Trade Clustering、DTW）
- 窄 IO 设计减少内存

**Qlib 优势**：
- Expression DSL 简洁，适合简单特征
- 自动依赖解析，无需手动管理 DAG
- 成熟的缓存机制

### 1.3 预处理层

| 组件 | 当前架构 | Qlib |
|------|---------|------|
| 预处理 | `processors.py` (可选) | `Processor` 链 (必须) |
| 填充 | `FillNaProcessor` | `DropnaLabel`, `CSZFillna` |
| 归一化 | `ClipOutlierProcessor` | `CSRankNorm`, `RobustZScoreNorm` |
| 类型转换 | `DtypeDowncastProcessor` | 无（默认 float32） |

**当前优势**：
- 特征函数内部已处理，processor 可选
- 更灵活的配置

**Qlib 优势**：
- 标准化的 Processor 链
- 更多内置归一化方法（横截面排名等）

### 1.4 存储层 (Feature Store)

| 组件 | 当前架构 | Qlib |
|------|---------|------|
| 存储 | `FeatureStore` (Parquet 分区) | `DatasetCache` + 本地二进制 |
| 分区 | `layer/symbol/timeframe/YYYY-MM.parquet` | `instruments/fields` |
| 增量 | 月度增量 | 天级增量 |
| 共享 | 跨策略共享 (`base_v1`, `heavy_v1`) | 跨实验共享 |

**当前优势**：
- Parquet 格式通用
- 层级分区支持策略间共享

**Qlib 优势**：
- 更成熟的缓存失效机制
- 天级增量更精细

## 2. 当前架构的独特能力

Qlib **不支持**或**不擅长**的领域：

| 能力 | 当前实现 | Qlib 状态 |
|------|---------|----------|
| **Tick 级特征** | VPIN, Trade Clustering, Footprint | ❌ 不支持 |
| **订单流分析** | CVD, TBR, OFI | ❌ 不支持 |
| **WPT 多尺度分解** | 价格/成交量/CVD 小波分解 | ❌ 不支持 |
| **DTW 模式匹配** | 历史形态识别 | ❌ 不支持 |
| **增量状态缓存** | VPIN bucket state 跨月传递 | ⚠️ 需自定义 |

**结论**：当前架构在**高频/订单流**领域有独特优势，Qlib 更适合**日频/因子投资**。

## 3. 统一架构方案

### 3.1 目标

统一 研究 / 回测 / 实盘 三个场景，共享：
- 数据加载 (DataHandler)
- 特征计算 (Feature DAG)
- 特征存储 (Feature Store)

### 3.2 当前状态

```
┌─────────────────────────────────────────────────────────────┐
│                      研究 (Research)                         │
│  scripts/train_strategy_pipeline.py                         │
│  → DataHandler → Feature DAG → Model Training               │
└─────────────────────────────────────────────────────────────┘
                              ↓ 共享 DataHandler + Feature DAG
┌─────────────────────────────────────────────────────────────┐
│                      回测 (Backtest)                         │
│  VectorBTBacktest (同一 feature 流程)                        │
└─────────────────────────────────────────────────────────────┘
                              ↓ TODO: 统一
┌─────────────────────────────────────────────────────────────┐
│                      实盘 (Live)                             │
│  TODO: Nautilus Trader 集成                                 │
│  → 需要增量特征计算                                          │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 缓存策略

#### Plan A: 增量缓存 (已实现)

适用于：**状态依赖特征** (VPIN, Trade Clustering)

```python
# 月度状态传递
prev_state = load_state(month-1)
features, new_state = compute_features(ticks, prev_state)
save_state(month, new_state)
```

- ✅ VPIN 增量缓存已实现
- ✅ Trade Clustering 增量缓存已实现
- ✅ 跨月状态传递

#### Plan B: Feature Store (已实现)

适用于：**无状态特征** (Baseline, TA-Lib, WPT)

```python
# 写入
store.write_month(spec, "2024-01", df_features)

# 读取
df = store.read_range(spec, "2024-01", "2024-12")
```

- ✅ 层级分区 (`base_v1`, `heavy_v1`)
- ✅ 跨策略共享
- ✅ 元数据记录

### 3.4 实盘集成方案 (TODO)

```
┌─────────────────────────────────────────────────────────────┐
│  Nautilus Trader                                            │
│  ├── on_bar() / on_tick()                                   │
│  │   ↓                                                      │
│  ├── FeatureActor (自定义 Actor)                            │
│  │   ├── DataHandler.load_ohlcv() (历史回填)                │
│  │   ├── IncrementalFeatureComputer (增量计算)              │
│  │   └── Feature Store (读取缓存)                           │
│  │   ↓                                                      │
│  └── Strategy.on_feature_update()                           │
└─────────────────────────────────────────────────────────────┘
```

关键组件：
1. `IncrementalFeatureComputer`: 支持单 bar 增量计算
2. `FeatureActor`: Nautilus Actor 封装特征计算
3. 状态管理: 内存中维护 rolling window state

## 4. 采用计划

### Phase 1: 当前状态 ✅

- [x] DataHandler 统一数据加载
- [x] Feature DAG 窄 IO
- [x] Feature Store 分区存储
- [x] VPIN/Trade Clustering 增量缓存
- [x] Processor Chain (可选工具)

### Phase 2: 代码质量优化

- [ ] 装饰器特征注册 (`@register_feature`)
- [ ] Console Script 入口
- [ ] 单元测试覆盖率提升

### Phase 3: 实盘集成

- [ ] Nautilus Trader FeatureActor
- [ ] 增量特征计算器
- [ ] 实时 Feature Store 更新

### Phase 4: 可选 Qlib 借鉴

如果需要：
- [ ] Expression DSL (简单特征快速定义)
- [ ] 横截面归一化 (CSRankNorm)
- [ ] 更精细的日级增量

## 5. 结论

| 维度 | 建议 |
|------|------|
| 数据层 | 保持当前 DataHandler，不迁移 Qlib 格式 |
| 特征层 | 保持 YAML + Python，支持复杂特征 |
| 缓存层 | 保持 Feature Store + 增量缓存 |
| 预处理 | Processor Chain 作为可选工具 |
| 实盘 | 自研 Nautilus 集成，不依赖 Qlib |

**核心原则**：
1. 保持当前架构的**订单流/高频能力**
2. 借鉴 Qlib 的**缓存/归一化思想**，不照搬实现
3. 统一 研究/回测/实盘 的**数据和特征流程**

