# OOD 和 Safety 模块状态分析

**生成时间**: 2026-01-25  
**目的**: 分析 `config/ood` 目录和相关模块的当前状态和使用情况

---

## 📋 总结

### ✅ 有实现，但处于研究阶段

**当前状态**：
- ✅ **配置文件存在**：`config/ood/` 目录下有完整的配置
- ✅ **代码实现存在**：有完整的 OOD/Survival 模块实现
- ✅ **CLI 命令存在**：可以通过命令训练和使用
- ⚠️ **实盘集成**：**未完全集成到实盘系统**（主要在研究和诊断阶段）

### ✅ 已落地的 Safety v1（硬停机 + 日内限制 + 冷却恢复）

**当前状态**：
- ✅ **实时拦截**：在 live enforcement 层对新开仓进行 Safety 过滤
- ✅ **持久化**：Safety 状态写入 SQLite（重启可恢复熔断状态）
- ✅ **路径**：`config/constitution/constitution.yaml` → `safety_state.persist_to`
- ✅ **冷却恢复**：冷却时间 + 指标恢复双条件
- ✅ **日亏损恢复**：仅跨日恢复（当日即使回归也不恢复）
- ⚠️ **EVT 极端风险**：仅做软告警，不触发硬停机

---

## 📁 config/ood 目录内容

### 1. `ood_config.yaml` ✅

**用途**：OOD Head / Survival Head 的安全控制配置

**关键配置项**：
- `labels`: OOD 和 Survival 的 horizon 定义
- `thresholds`: 操作阈值（degrade/halt）
- `revive`: 恢复阈值和阶段
- `size_cap`: 仓位上限公式
- `dashboard`: LiveDashboard 的 5 个关键指标

**使用位置**：
- `src/time_series_model/diagnostics/ood_config.py` - 配置加载
- `src/time_series_model/diagnostics/live_dashboard.py` - Dashboard 构建
- `src/time_series_model/ops/state_snapshot.py` - 状态快照

**状态**：✅ **已实现，有代码使用**

---

### 2. `ood_to_archetype_table.yaml` ✅

**用途**：条件生存表（Conditional Survival Table），学习 `survival_rate(archetype | ood_bin)`

**关键配置项**：
- `archetypes`: TC/TE/FR/ET 列表
- `bins`: OOD 分箱定义（low/mid/high）
- `weights`: 权重转换参数

**使用位置**：
- `src/time_series_model/diagnostics/ood_to_archetype_table.py` - 表构建逻辑
- CLI 命令：`mlbot diagnose ood-to-archetype-weights`

**状态**：✅ **已实现，有 CLI 命令**

---

### 3. `survival_head_mlp.yaml` ✅

**用途**：Survival Head（小 MLP）的训练配置

**关键配置项**：
- `data`: 特征列定义
- `split`: 时间分割参数
- `train`: 训练超参数
- `loss`: Loss 函数配置

**使用位置**：
- `src/time_series_model/diagnostics/survival_head_mlp.py` - 训练实现
- CLI 命令：`mlbot diagnose survival-head-train`

**状态**：✅ **已实现，有 CLI 命令**

---

## 💻 代码实现状态

### ✅ 已实现的模块

#### 1. OOD 配置加载
**文件**: `src/time_series_model/diagnostics/ood_config.py`
- 完整的配置类定义
- YAML 加载函数
- 配置验证

#### 2. Survival Head 训练
**文件**: `src/time_series_model/diagnostics/survival_head_mlp.py`
- 完整的 MLP 模型定义
- 训练循环
- 评估指标（AUC, AP, Calibration）
- 报告生成

#### 3. OOD → Archetype 映射表
**文件**: `src/time_series_model/diagnostics/ood_to_archetype_table.py`
- 表构建逻辑
- 权重计算
- YAML 导出

#### 4. Live Dashboard
**文件**: `src/time_series_model/diagnostics/live_dashboard.py`
- 使用 `OODConfig` 构建 Dashboard payload
- 包含 OOD/Survival 相关指标

---

## 🔧 CLI 命令状态

### ✅ 可用的命令

#### 1. `mlbot diagnose survival-head-train`
**用途**：训练 Survival Head（小 MLP）

```bash
mlbot diagnose survival-head-train --no-docker \
  --logs results/nnmultihead/my_run_name/logs_3action.parquet \
  --labels results/extinction_replay/my_run/labels.parquet \
  --out results/survival_head/my_run \
  --config config/ood/survival_head_mlp.yaml
```

**状态**：✅ **可用**

---

#### 2. `mlbot diagnose ood-to-archetype-weights`
**用途**：学习 OOD → Archetype 权重表

```bash
mlbot diagnose ood-to-archetype-weights --no-docker \
  --logs results/nnmultihead/my_run_name/logs_3action.parquet \
  --labels results/extinction_replay/my_run/labels.parquet \
  --out results/ood_to_archetype/my_run \
  --config config/ood/ood_to_archetype_table.yaml
```

**状态**：✅ **可用**

---

#### 3. `mlbot diagnose extinction-replay-3action`
**用途**：灭绝回放（生成 Survival 标签）

```bash
mlbot diagnose extinction-replay-3action --no-docker \
  --logs results/nnmultihead/my_run_name/logs_3action.parquet \
  --out results/extinction_replay/my_run
```

**状态**：✅ **可用**（这是 Survival Head 训练的前置步骤）

---

## ⚠️ 实盘集成状态

### ❌ 未完全集成到实盘系统

**检查结果**：
- `src/time_series_model/live/event_driven_strategy.py` - **未使用 OOD/Survival**
- `src/time_series_model/live/nautilus_strategy_*.py` - **未使用 OOD/Survival**

**当前实盘系统**：
- 主要使用 Gate 层进行过滤
- 使用 Execution Manager 进行风险控制
- **没有集成 OOD Head 或 Survival Head**

---

## 📚 架构文档中的定位

### 根据 `docs/ARCHITECTURE.md`

**版本划分**：
- **v0**：可无 Safety Head（靠 Gate + 保守 PCM/slot 控制保证生存）
- **v1**：Unified Safety Head（合并 OOD/Survival/Safety）是必选
- **v2**：视证据保留

**当前状态**：
- 系统处于 **v0 阶段**
- OOD/Survival 模块已实现，但**主要用于研究和诊断**
- **未作为生产决策层集成**

---

## 🧠 关于“简化为 Unified Safety + 自指”的补充说明

### 1) 是否应简化为 Unified Safety

**可以，且是合理的工程收敛方向**，但建议保留**最少的三件事**：

- **OOD 检测**：判断是否处于“训练世界”之外（进入降速/停机逻辑）
- **极端风险检测**：极端波动/流动性断崖/相关性塌缩（硬停机）
- **统一风险输出**：影响 `risk` / `size_cap` / `halt` / `revive`

> 简化不是删除“风险感知”，而是把风险感知统一成一个可执行的 Safety 接口。

---

### 2) 没有 Unified Safety，系统会不会“无法反省自身”

**会变弱，但不至于完全失去反省能力**：

- **没有 Unified Safety**：系统只能依赖 Gate/PCM 做硬性限制，无法形成**连续的风险记忆**。
- **有 Unified Safety**：系统能够把 OOD/Survival/极端事件统一为**连续风险概率**，形成**“自指的自我约束”**。

这就是“自指”的含义：  
> **系统用自己过去的生存结果来调整自己未来的行为强度**。

---

### 3) 如果改成“全部概率输出 + river 实时更新”，算不算自指？

**不完全算**。  
在线更新只是“动态适配”，不是“自指反省”。区别在于：

- **在线更新（river）**：参数随数据漂移，**不知道自己为什么错**  
- **自指反省（Unified Safety）**：把**“错误后果”**变成可执行约束（记仇）

---

### 4) 自指的好处是什么

**核心好处：降低“重复死法”的概率**。具体表现为：

- **错误具有记忆**：相同场景再次出现时，系统更早降仓或停机
- **风险反射更快**：不需要重训即可“反射式收缩”
- **更稳的长期资本曲线**：收益变少但更难被极端事件击穿

---

### 5) 长期反应系统的缺点

**主要缺点是“过度保守”**：

- **错杀机会**：风险过高时会放弃可盈利机会
- **收益上限降低**：长期期望收益变“钝”
- **系统变慢**：复活/恢复速度可能偏慢

但对实盘来说，这是**以收益换生存**，通常是正确的交换。

---

### 6) 为什么归因很重要？是否影响最终收益？

**归因会直接影响最终收益**，原因是：

1. **能精准修复问题层**  
   - 不归因：只能全量重训或盲调阈值  
   - 归因：只修 OOD / Survival / Gate / Execution 的具体层  
   → 降低过拟合、避免收益波动

2. **减少无效改动**  
   - 没有归因时，改动常常是“错误修正”
   - 归因让改动目标明确，改动更少但更有效

3. **缩短修复周期**  
   - 发现问题 → 定位 → 修复 → 验证  
   这个闭环越短，系统越能在真实市场中“活下来”

**结论**：  
归因不是“研究习惯”，而是**决定最终收益稳定性的工程能力**。

---

### 7) Safety 模块与其他层的关系（简化版）

**建议使用以下简化结构：**

```
Path/Regime Heads  →  OOD/Extremes  →  Unified Safety  →  Execution
```

**职责边界**：
- **Path/Regime**：给出机会空间（不负责安全）
- **OOD/Extremes**：提供风险信号（不下决策）
- **Unified Safety**：唯一决策（allow/size/halt/revive）
- **Execution**：把收益落地（安全之下）

## 🎯 使用建议

### ✅ 当前可以使用的场景

1. **研究阶段**：
   - 训练 Survival Head 模型
   - 学习 OOD → Archetype 映射表
   - 分析灭绝风险

2. **诊断阶段**：
   - 使用 Live Dashboard 监控 OOD/Survival 指标
   - 生成诊断报告

3. **准备 v1 升级**：
   - 提前训练好 Survival Head
   - 准备好 OOD → Archetype 映射表
   - 为 v1 集成做准备

### ⚠️ 当前不能使用的场景

1. **实盘交易决策**：
   - OOD/Survival 分数**不会影响实盘交易决策**
   - 实盘系统主要依赖 Gate 层

2. **自动仓位控制**：
   - `size_cap` 公式**未在实盘中应用**
   - 实盘使用 Execution Manager 的仓位控制

---

## 📊 模块依赖关系

```
extinction-replay-3action (生成 labels.parquet)
    ↓
survival-head-train (训练 Survival Head)
    ↓
ood-to-archetype-weights (学习映射表)
    ↓
[未来] 集成到实盘系统 (v1)
```

---

## 🔮 未来规划（根据架构文档）

### v1 阶段（计划中）

**Unified Safety Head**：
- 合并 OOD/Survival/Safety 为一个统一模块
- 作为生产决策层集成
- 影响仓位控制和交易决策

**集成点**：
- 实盘策略需要加载 Survival Head 模型
- 实盘策略需要加载 OOD → Archetype 映射表
- 实盘策略需要应用 `size_cap` 公式

---

## 📝 结论

### `config/ood` 目录的状态

1. **✅ 有用**：配置完整，代码实现完整
2. **✅ 可用**：有 CLI 命令可以使用
3. **⚠️ 未集成**：未完全集成到实盘系统
4. **🎯 研究阶段**：主要用于研究和诊断，为 v1 做准备

### 建议

1. **保留配置**：这些配置是 v1 升级的基础
2. **继续研究**：可以继续使用这些工具进行研究
3. **准备集成**：为 v1 阶段的集成做准备
4. **文档化**：记录研究结果，为集成做准备

---

## 🔗 相关文档

- [系统架构（统一版）](./ARCHITECTURE.md) - 了解 v0/v1/v2 的划分
- [最终简化架构](./architecture/FINAL_SIMPLIFIED_ARCHITECTURE_2026_01.md) - 了解详细设计
- [完整命令速查表](./完整命令速查表.md) - 查看相关命令

---

**最后更新**: 2026-01-25  
**状态**: ✅ 当前版本
