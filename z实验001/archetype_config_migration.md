# Archetype 配置迁移方案

> **目标**：从 `config/nnmultihead/execution_archetypes.yaml` 迁移到 `config/strategies/{strategy}/archetypes/` 的三层解耦配置体系，废弃 NN 多头架构，建立 Gate / Evidence / Execution 清晰分层。

---

## 📋 背景与动机

### 当前问题

#### 问题 1：NN 多头架构已成技术债（建议完全废弃）

**现状**：
- 当前代码保留了 `nnmultihead` 相关模块，但实际上已被"单头分类 archetype + structure-driven 架构"完全取代：
  - **`dir` 头（方向预测）**：已由 `direction_policy.structure_direction` 根据结构（trend_sign / breakout_sign / failed_breakout）直接决定，无需模型预测
  - **`mfe/mae` 头（目标/止损预测）**：已由 archetype 的 `execution_constraints.fixed_rr` 固定（如 BPC 固定 1.0R 止损 / 2.5R 目标）
  - **`mtt` 头（持仓时间预测）**：已用 `max_holding_bars` + mean/trend regime 分类替代

**问题本质**：
- NN 多头模型是为了"预测多个连续值目标"设计的，但现在所有目标都**不需要预测**：
  - Direction 由结构规则确定（不是预测，是检测）
  - RR 由 archetype 固定（不需要预测）
  - Holding time 由 regime 分类决定（离散分类，不是回归）
- 继续维护 `nnmultihead` 代码是**纯粹的技术债**，增加理解成本和维护成本

**决策**：**彻底废弃 NN 多头架构**，清理所有相关代码和配置

---

#### 问题 2：配置位置不合理（应放到 strategies/ 下）

**现状**：
- `execution_archetypes.yaml` 放在 `config/nnmultihead/` 下，但已经不是 NN 多头模型的配置
- 这个配置实际上是各 **archetype 的规则定义**（如 BPC / HTF / ME 等策略的交易规则）

**问题本质**：
- Archetype 配置应该和策略训练配置（features / labels / backtest）放在同一层级
- 当前放在 `nnmultihead/` 下误导读者，以为这是模型的一部分

**正确位置**（两种理解，推荐理解 A）：

**理解 A（推荐）**：每个策略有单一 archetype 目录
```
config/strategies/
  bpc/                          # BPC 策略所有配置
    features_gate.yaml          # Gate 训练特征
    features_evidence.yaml      # Evidence 训练特征
    labels_rr_extreme.yaml      # 训练标签
    labels_return_tree.yaml     # 训练标签
    archetypes/                 # 实战规则（从训练产物提炼）
      bpc_gate.yaml             # Gate 规则
      bpc_evidence.yaml         # Evidence 规则
      bpc_execution.yaml        # Execution 约束
```

**理解 B（不推荐）**：每个训练实验独立目录
```
config/strategies/
  bpc_rr_extreme/               # BPC failure 训练实验
    features.yaml
    labels.yaml
    archetypes/
      bpc_gate.yaml
  bpc_gate/                     # BPC gate 训练实验
    features.yaml
    labels.yaml
    archetypes/
      bpc_gate.yaml
```

**推荐理解 A**，理由：
- BPC 策略就是 BPC，不应该因为训练标签不同就拆成多个目录
- `bpc_rr_extreme` / `bpc_gate` 是训练**实验的输出产物**（放在 `results/` 下），不是配置源
- 一个策略可以有多个训练实验，但实战规则只有一份（从最佳实验提炼）

---

#### 问题 3：职责混杂（违反三层解耦架构）

**现状**：
- 当前 `execution_archetypes.yaml` 把 safety / exclusions / preconditions / evidence / execution_constraints 全混在一个文件里
- 一个 YAML 文件超过 800 行，难以维护和理解

**问题本质**：
- 违反了 `Gate模块宪法级职责与三层解耦架构` 的设计原则：
  - **Gate**：只管 safety + exclusions + 结构合法性（硬 veto）
  - **Evidence**：只管语义发现 + 置信度调整（软约束）
  - **Execution**：只管 RR、持仓约束、方向策略（仓位放大）

**正确做法**：拆分成三个独立配置文件，各司其职

### 目标架构

```
config/strategies/
  bpc/
    # 训练配置（已有）
    features_gate.yaml           # Gate 训练特征（≤10个）
    labels_rr_extreme.yaml       # Failure 标签
    labels_return_tree.yaml      # GOOD 样本标签
    features_evidence.yaml       # Evidence 训练特征
    backtest.yaml
    model.yaml
    
    # Archetype 实战配置（新增）- 从训练产物提炼
    archetypes/
      bpc_gate.yaml              # Gate 规则（从 risk_gate_draft.yaml 提炼）
      bpc_evidence.yaml          # Evidence 规则（从 Return Tree 提炼）
      bpc_execution.yaml         # Execution 约束（RR / holding bars / direction_policy）
  
  # 其他策略同理
  htf/archetypes/
    htf_gate.yaml
    htf_evidence.yaml
    htf_execution.yaml
  me/archetypes/
    me_gate.yaml
    me_evidence.yaml
    me_execution.yaml
  # ... 等
```

**废弃的目录**（迁移后删除）：
```
config/nnmultihead/              # ❌ 完全废弃
  execution_archetypes.yaml      # ❌ 迁移到 strategies/{strategy}/archetypes/
  strategies/                    # ❌ 已无用（NN 多头架构废弃）
```

---

## 🗺️ 迁移方案（分 6 步执行）

> **预计总工时**：6-8 小时，分两周完成  
> **关键决策**：采用**理解 A** - 每个策略单一 archetype 目录，而非按训练实验拆分  
> **向后兼容**：先保留 `config/nnmultihead/execution_archetypes.yaml` 兼容性，打印 deprecation warning，逐步迁移

---

### Step 0: 前置验证（5 分钟）

**目标**：确认当前哪些策略已有完整的训练配置

```bash
# 检查已有策略目录
ls -la config/strategies/

# 确认 BPC 目录结构
tree config/strategies/bpc/
```

**验证清单**：
- [ ] `config/strategies/bpc/` 包含 `features_gate.yaml`, `labels_rr_extreme.yaml`, `risk_gate.yaml`
- [ ] `config/strategies/htf/`, `me/`, `fbf/`, `lsr/`, `aer/` 目录存在
- [ ] `config/nnmultihead/execution_archetypes.yaml` 包含 6 个 archetype

---

**操作**：在每个策略目录下创建 `archetypes/` 子目录

```bash
mkdir -p config/strategies/bpc/archetypes
mkdir -p config/strategies/aer/archetypes
mkdir -p config/strategies/fbf/archetypes
mkdir -p config/strategies/lsr/archetypes
mkdir -p config/strategies/me/archetypes
mkdir -p config/strategies/htf/archetypes
```

**验证**：目录创建成功，符合 `BPC Evidence配置目录规范`。

---

### Step 2: 拆分 BPC Archetype 配置（30 分钟）

从 `config/nnmultihead/execution_archetypes.yaml` 中提取 `BreakoutPullbackContinuation` 的配置，按三层拆分。

> **注意**：`config/strategies/bpc/risk_gate.yaml` 已经是更精炼的 Gate 规则（从树模型训练产物提炼），应该作为 `bpc_gate.yaml` 的基础，**不是从 execution_archetypes.yaml 提取**。

#### 2.1 创建 `config/strategies/bpc/archetypes/bpc_gate.yaml`

**职责**：只含 Gate 的 system_safety / hard_gate / soft_filter

**来源**：直接复制 `config/strategies/bpc/risk_gate.yaml`（这是从树模型训练产物提炼的最新版本）

```bash
# 复制现有的 risk_gate.yaml 作为基础
cp config/strategies/bpc/risk_gate.yaml config/strategies/bpc/archetypes/bpc_gate.yaml
```

**说明**：
- `risk_gate.yaml` 已经符合 Gate 三层架构（system_safety / hard_gate / soft_filter）
- Gate 特征 ≤10 个，只含硬约束项（如 `vpin_ma20`、`bpc_dir_consistency_long`、`jump_risk_pct`）
- VPIN / CVD 等订单流特征已在 `soft_filter` 中正确定位，不需要移到 Evidence
- Attribution Tags 已定义完整，支持月度复盘

---

#### 2.2 创建 `config/strategies/bpc/archetypes/bpc_evidence.yaml`

**职责**：只含 Evidence 的 preconditions / evidence（语义发现 + 置信度调整）

**来源**：从 `execution_archetypes.yaml` 的 `preconditions` / `evidence` 阶段提取

```yaml
version: 1
name: bpc_evidence_rules
description: "BPC Evidence 规则 - 只负责影响置信度/排序/仓位，不做硬 veto"

# Evidence 职责：语义发现 + 置信度调整
# - preconditions: 必要条件（require），但不是硬 veto
# - evidence: 正面证据（amplify），用于仓位调整

preconditions:
  - id: bpc_path_efficiency_high
    priority: 3
    reason: "强语义: 趋势结构有效（结构再次被市场证明）"
    when:
      path_efficiency_pct:
        quantile_gte: 0.6
    then:
      action: require
      confidence_boost: 0.1

  - id: bpc_dir_consistency_high
    priority: 3
    reason: "强语义: 方向一致性"
    when:
      price_dir_consistency_pct:
        quantile_gte: 0.6
    then:
      action: require
      confidence_boost: 0.1

  - id: bpc_compression_score
    priority: 3
    reason: "强语义: 结构压缩→解压（compression score > 0.75）"
    when:
      path_efficiency_pct:
        quantile_gte: 0.55
    then:
      action: require
      confidence_boost: 0.05

evidence:
  - id: bpc_orderflow_present
    priority: 4
    reason: "证据: 订单流支撑（pullback 未被反向接管）"
    when:
      vpin:
        quantile_gt: 0.55
    then:
      action: amplify
      weight: 0.8

  - id: bpc_pullback_not_reversed
    priority: 4
    reason: "证据: 反向单没有真的掌控订单簿（cvd_change_5_pct 不极端负）"
    when:
      cvd_change_5_pct:
        value_gt: 0.3
    then:
      action: amplify
      weight: 0.7
```

**说明**：
- Evidence 仅用于影响置信度、排序和仓位大小，不得替代 Gate veto
- 使用 `confidence_boost` / `weight` / `action: amplify` 等软约束机制
- 这些规则应该从 **Return Tree 训练产物** 提炼（通过 `mlbot train export-rules --evidence` 命令）

---

#### 2.3 创建 `config/strategies/bpc/archetypes/bpc_execution.yaml`

**职责**：只含 Execution 的 RR、holding bars、direction_policy

**来源**：从 `execution_archetypes.yaml` 的 `execution_constraints` / `direction_policy` 提取

```yaml
version: 1
name: bpc_execution_constraints
description: "BPC Execution 约束 - RR、持仓时间、方向策略"

execution_constraints:
  allow_add_on: false
  min_order_interval_minutes: 60
  fixed_rr:
    stop_loss_r: 1.0
    take_profit_r: 2.5
    max_holding_bars: null  # 不限制持仓时间，由市场决定

direction_policy:
  direction_source: structure  # 从结构推断，不是从模型预测
  structure_direction:
    method: trend_sign         # 根据 trend_sign 特征决定方向
    lookback_bars: 5
    min_consistency: 0.6       # 至少 60% 的 bar 方向一致
```

**说明**：
- Execution 负责仓位放大，通过 `exec_multiplier` 公式实现风险可控的交易频次提升（后续可扩展）
- `direction_policy.structure_direction.method` 支持：
  - `trend_sign`: 根据 5 bar 趋势方向
  - `breakout_sign`: 根据突破方向（ME archetype 使用）
  - `failed_breakout`: 根据假突破反转方向（FBF archetype 使用）
- 这是 NN 多头模型被废弃的原因：**方向由结构规则决定，不需要模型预测**

---

### Step 3: 更新代码以支持三层配置加载

**需要修改的核心模块**：

#### 3.1 修改 `load_execution_archetypes_registry()` 函数

**位置**：`src/time_series_model/nnmultihead/strategy_profile.py:41`

**当前实现**：
```python
def load_execution_archetypes_registry(
    path: str | Path = "config/nnmultihead/execution_archetypes.yaml",
) -> Dict[str, ExecutionArchetype]:
    # 从单个 YAML 加载所有 archetype
    obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    archetypes = obj.get("archetypes")
    # ...
```

**新实现**（支持三层配置）：
```python
def load_execution_archetypes_registry(
    strategies_root: str | Path = "config/strategies",
    legacy_path: str | Path = "config/nnmultihead/execution_archetypes.yaml",
) -> Dict[str, ExecutionArchetype]:
    """
    从 config/strategies/{strategy}/archetypes/ 加载三层配置：
    - {strategy}_gate.yaml
    - {strategy}_evidence.yaml
    - {strategy}_execution.yaml
    
    向后兼容：如果新路径不存在，回退到 legacy_path（打印 deprecation warning）
    """
    strategies_root = Path(strategies_root)
    archetypes: Dict[str, ExecutionArchetype] = {}
    
    # 遍历所有策略目录
    for strategy_dir in strategies_root.iterdir():
        if not strategy_dir.is_dir():
            continue
        
        archetypes_dir = strategy_dir / "archetypes"
        if not archetypes_dir.exists():
            continue
        
        strategy_name = strategy_dir.name
        
        # 加载三层配置
        gate_path = archetypes_dir / f"{strategy_name}_gate.yaml"
        evidence_path = archetypes_dir / f"{strategy_name}_evidence.yaml"
        execution_path = archetypes_dir / f"{strategy_name}_execution.yaml"
        
        if not all([gate_path.exists(), evidence_path.exists(), execution_path.exists()]):
            continue
        
        # 解析并合并
        gate_rules = yaml.safe_load(gate_path.read_text()) or {}
        evidence_rules = yaml.safe_load(evidence_path.read_text()) or {}
        execution_config = yaml.safe_load(execution_path.read_text()) or {}
        
        # 组装 ExecutionArchetype
        archetypes[strategy_name] = ExecutionArchetype(
            name=strategy_name,
            when_then_rules=_merge_when_then_rules(gate_rules, evidence_rules),
            execution_constraints=execution_config.get("execution_constraints"),
            direction_policy=execution_config.get("direction_policy"),
            # ...
        )
    
    # 如果没有找到新配置，回退到旧路径（向后兼容）
    if not archetypes and Path(legacy_path).exists():
        warnings.warn(
            f"⚠️  DEPRECATED: {legacy_path} is deprecated. "
            f"Please migrate to config/strategies/{{strategy}}/archetypes/",
            DeprecationWarning,
            stacklevel=2,
        )
        return _load_legacy_archetypes(legacy_path)
    
    return archetypes
```

#### 3.2 修改所有调用点

**需要修改的文件**（grep 搜索结果）：
1. `src/time_series_model/core/meta_router_core.py:77`
2. `src/time_series_model/live/meta_router_config.py:46-52`
3. `src/time_series_model/live/live_feature_plan.py:49, 147`
4. `src/cli/main.py:205`
5. `scripts/optimize_gate_plateau_hard_gate.py:161`
6. 其他脚本文件（约 10+ 个）

**修改策略**：
- 所有调用 `load_execution_archetypes_registry()` 的地方，改成传 `strategies_root="config/strategies"`
- 保留 `legacy_path` 参数的向后兼容性，打印 deprecation warning
- 第一周只修改核心文件（meta_router / live），脚本文件下周再改

---

### Step 4: 迁移其他 Archetype（2 小时）

按 Step 2 的模板，依次拆分其他 archetype（HTF / ME / FBF / LSR / AER）：

#### 4.1 批量创建 archetype 目录

```bash
mkdir -p config/strategies/htf/archetypes
mkdir -p config/strategies/me/archetypes
mkdir -p config/strategies/fbf/archetypes
mkdir -p config/strategies/lsr/archetypes
mkdir -p config/strategies/aer/archetypes
```

#### 4.2 拆分每个 archetype 配置

**来源**：从 `config/nnmultihead/execution_archetypes.yaml` 提取各 archetype 的配置

**脚本化工具**（可选，建议手工拆分以确保理解）：
```python
# scripts/split_archetypes_config.py
import yaml
from pathlib import Path

# 读取原配置
with open("config/nnmultihead/execution_archetypes.yaml") as f:
    config = yaml.safe_load(f)

for name, arch in config["archetypes"].items():
    strategy_name = name.lower().replace("breakoutpullbackcontinuation", "bpc")
    archetype_dir = Path(f"config/strategies/{strategy_name}/archetypes")
    archetype_dir.mkdir(parents=True, exist_ok=True)
    
    # 拆分 gate / evidence / execution
    # ...
```

**人话**：这一步工作量较大，建议先完成 BPC 验证后再批量执行

---

### Step 5: 废弃 `config/nnmultihead/` 目录（30 分钟）

**操作**：

1. **备份旧配置**：
   ```bash
   mv config/nnmultihead config/nnmultihead_deprecated_$(date +%Y%m%d)
   ```

2. **在代码中移除所有对 `config/nnmultihead/execution_archetypes.yaml` 的引用**：
   - 移除 `legacy_path` 参数的默认值
   - 移除所有 `nnmultihead` 相关的模块（可选，分步执行）

3. **更新文档和 README**：
   - 在 `ARCHITECTURE.md` 中更新配置路径说明
   - 在 `cmd: 树模型到archetype.md` 中更新路径
   - 在 `完整命令速查表.md` 中更新路径

**验证清单**：
- [ ] 代码中没有 `config/nnmultihead/execution_archetypes.yaml` 的硬编码路径
- [ ] 所有文档中的配置路径已更新为 `config/strategies/{strategy}/archetypes/`
- [ ] `config/nnmultihead/` 目录已备份并从主分支删除
1. `src/time_series_model/core/meta_router_core.py:77`
2. `src/time_series_model/live/meta_router_config.py:46-52`
3. `src/time_series_model/live/live_feature_plan.py:49, 147`
4. `src/cli/main.py:205`
5. `scripts/optimize_gate_plateau_hard_gate.py:161`
6. 其他脚本文件（约 10+ 个）

**修改策略**：
- 所有调用 `load_execution_archetypes_registry()` 的地方，改成传 `strategies_root="config/strategies"`
- 保留 `legacy_path` 参数的向后兼容性，打印 deprecation warning

#### 3.3 废弃 NN 多头相关代码

**需要删除的模块**（可选，建议分步执行）：
1. `src/time_series_model/nnmultihead/` 目录（除了 `strategy_profile.py` 保留作为 archetype 加载器）
2. `config/nnmultihead/strategies/` 目录（完全无用）
3. 所有 `nnmultihead_inference` 相关配置和代码

**代码改动点**：
- 先注释掉相关代码，确保回测不受影响
- 验证通过后再物理删除文件

---

### Step 4: 迁移其他 Archetype（AER / FBF / LSR / ME / HTF）

按 Step 2 的模板，依次拆分其他 archetype：

```bash
config/strategies/
  aer/archetypes/
    aer_gate.yaml
    aer_evidence.yaml
    aer_execution.yaml
  fbf/archetypes/
    fbf_gate.yaml
    fbf_evidence.yaml
    fbf_execution.yaml
  # ... 其他策略同理
```

**批量操作**：可以写一个脚本自动拆分 `execution_archetypes.yaml` 的各 archetype 配置。

---

### Step 5: 废弃 `config/nnmultihead/` 目录

**操作**：

1. 备份旧配置：
   ```bash
   mv config/nnmultihead config/nnmultihead_deprecated_$(date +%Y%m%d)
   ```

2. 在代码中移除所有对 `config/nnmultihead/execution_archetypes.yaml` 的引用；

3. 更新文档和 README，说明新的配置位置。

---

### Step 6: 验证与回测（30 分钟）

**验证清单**：

- [ ] **加载测试**：`load_execution_archetypes_registry()` 能成功加载所有 archetype
  ```python
  from src.time_series_model.nnmultihead.strategy_profile import load_execution_archetypes_registry
  archetypes = load_execution_archetypes_registry("config/strategies")
  print(f"加载了 {len(archetypes)} 个 archetype: {list(archetypes.keys())}")
  ```

- [ ] **Gate 规则生效**：`safety` / `hard_gate` / `soft_filter` 正确触发 deny / downweight
- [ ] **Evidence 规则生效**：置信度调整正确（`confidence_boost` / `weight` 生效）
- [ ] **Execution 约束生效**：RR / holding bars 符合配置
- [ ] **回测输出一致**：Trades 数量与旧配置一致（或按预期变化）
- [ ] **Failure Analysis 正常**：lift 曲线符合预期

**回测命令**（示例）：

```bash
# BPC 策略回测
mlbot train final --no-docker \
  --config config/strategies/bpc \
  --features config/strategies/bpc/features_gate.yaml \
  --labels config/strategies/bpc/labels_rr_extreme.yaml \
  --symbol BTCUSDT,ETHUSDT,BNBUSDT \
  --timeframe 240T \
  --holdout-start-date 2024-05-01 \
  --holdout-end-date 2025-11-30
```

验证 HTML 报告中的：
- Failure Analysis 中的 lift 曲线
- Lift vs Coverage 曲线
- Attribution Tags 统计（如果已实现 Phase 5）

**预期结果**：
- BPC 的 failure_rr_extreme 过滤率 ~70%+
- 保留 trade 的 lift: 0.36x
- 剩余 ~8% 不可预测风险（news/liquidity shock）

---

## 🎯 迁移优先级与时间表

### 本周内完成（4 小时）

| 步骤 | 内容 | 工时 | 负责人 |
|------|------|------|--------|
| **Step 0** | 前置验证 | 5 分钟 | ✅ |
| **Step 1** | 创建目录结构 | 5 分钟 | 待执行 |
| **Step 2** | 拆分 BPC archetype | 30 分钟 | 待执行 |
| **Step 3.1** | 修改 `load_execution_archetypes_registry()` | 1 小时 | 待执行 |
| **Step 3.2** | 修改核心调用点（meta_router / live） | 1 小时 | 待执行 |
| **Step 6** | 验证 BPC 回测 | 30 分钟 | 待执行 |

### 下周完成（3 小时）

| 步骤 | 内容 | 工时 | 负责人 |
|------|------|------|--------|
| **Step 4** | 迁移其他 archetype（HTF/ME/FBF/LSR/AER） | 2 小时 | 待执行 |
| **Step 3.2** | 修改脚本调用点（10+ 脚本） | 30 分钟 | 待执行 |
| **Step 5** | 废弃 `config/nnmultihead/` | 30 分钟 | 待执行 |

---

## 📚 相关文档

- [树模型训练流程](./cmd:%20树模型到archetype.md)
- [Return Tree KPI 框架](./return_tree_kpi_framework.md)
- [Gate模块宪法级职责与三层解耦架构](memory: Gate模块宪法级职责与三层解耦架构)
- [BPC Evidence配置目录规范](memory: BPC Evidence配置目录规范)

---

## ⚠️ 注意事项与风险控制

### 1. 向后兼容性（Critical）

**必须保证**：
- 代码改动时先保留对旧路径的兼容性，打印 deprecation warning
- 第一周只改核心模块（meta_router / live），脚本文件下周再改
- 如果新配置不存在，自动回退到 `config/nnmultihead/execution_archetypes.yaml`
- **验证方式**：分别测试有无新配置的情况下，系统能否正常运行

### 2. 配置路径理解（Important）

**明确采用理解 A**：
```
config/strategies/
  bpc/                          # BPC 策略
    archetypes/                 # 实战规则（单一）
      bpc_gate.yaml
      bpc_evidence.yaml
      bpc_execution.yaml
```

**不是理解 B**（❌ 错误）：
```
config/strategies/
  bpc_rr_extreme/               # ❌ 训练实验不是配置
  bpc_gate/                     # ❌ 训练实验不是配置
```

**原因**：
- `bpc_rr_extreme` / `bpc_gate` 是训练实验的**输出产物**（在 `results/` 下），不是配置源
- 一个策略可以有多个训练实验，但实战规则只有一份（从最佳实验提炼）
- 配置应该"面向策略"，而不是"面向实验"

### 3. Gate vs Evidence 职责边界（Important）

**关键区别**：

| 层 | 职责 | 特征数量 | 作用 | 数据来源 |
|----|------|---------|------|----------|
| **Gate** | 硬 veto（safety + regime + structure） | ≤10 | deny | failure_rr_extreme 训练产物 |
| **Evidence** | 软约束（置信度调整） | 无限制 | amplify/weight | return_tree 训练产物 |
| **Execution** | RR / 持仓 / 方向 | N/A | 仓位放大 | archetype 定义 |

**注意**：
- BPC 的 `risk_gate.yaml` 已经是正确的 Gate 配置，**不需要从 execution_archetypes.yaml 提取**
- VPIN / CVD 在 BPC `risk_gate.yaml` 中作为 `soft_filter`（降权），这是正确的，**不需要移到 Evidence**
- Evidence 应该从 Return Tree 训练产物提炼（通过 `mlbot train export-rules --evidence`）

### 4. 测试覆盖（Critical）

**每个 archetype 迁移后都要跑一次回测**：
- 确保 Trades 数量符合预期（±5% 可接受）
- 重点验证 Gate 的 lift 曲线（failure_rr_extreme 过滤率 ~70%+）
- 验证 Evidence 的置信度调整是否生效（通过 Sharpe plateau 分析）
- 检查 HTML 报告中的 Attribution Tags 统计（如果已实现）

**失败处理**：
- 如果 Trades 数量差异 >10%，立即回滚并排查问题
- 如果 lift 曲线异常，检查 Gate 规则是否正确加载
- 如果置信度调整无效，检查 Evidence 规则的 `confidence_boost` / `weight` 参数

### 5. 文档更新（Important）

**迁移完成后，必须更新的文档**：
- [ ] `README.md` - 更新配置路径说明
- [ ] `ARCHITECTURE.md` - 更新架构图和配置说明
- [ ] `cmd: 树模型到archetype.md` - 更新所有命令的配置路径
- [ ] `完整命令速查表.md` - 更新 `mlbot train` 系列命令
- [ ] `docs/guides/` 下所有相关文档

**文档中需要修改的内容**：
- 所有 `config/nnmultihead/execution_archetypes.yaml` → `config/strategies/{strategy}/archetypes/`
- 所有 `nnmultihead` 相关描述 → `structure-driven archetype`
- 添加"三层配置拆分"的说明（Gate / Evidence / Execution）

### 6. NN 多头清理（Optional）

**建议分步执行**：
- **Phase 1**（本周）：只废弃 `config/nnmultihead/execution_archetypes.yaml`，代码保留向后兼容
- **Phase 2**（下周）：注释掉 `nnmultihead` 相关代码，确保回测不受影响
- **Phase 3**（后续）：物理删除 `src/time_series_model/nnmultihead/` 目录（保留 `strategy_profile.py` 作为 archetype 加载器）

**风险控制**：
- 不要一次性删除所有 `nnmultihead` 代码，避免引入不可预见的问题
- 每一步都要跑一次完整回测，确保系统稳定

---

## ✅ 完成标志

### 技术指标

- [ ] 所有 archetype 配置迁移到 `config/strategies/{strategy}/archetypes/`
- [ ] `load_execution_archetypes_registry()` 支持三层配置加载
- [ ] 所有核心模块（meta_router / live）调用点已更新
- [ ] 回测验证通过，Trades / lift 曲线符合预期
- [ ] `config/nnmultihead/` 目录已备份并从主分支删除
- [ ] 所有文档引用路径已修正

### 质量指标

- [ ] BPC 策略回测：failure_rr_extreme 过滤率 ~70%+，lift ~0.36x
- [ ] 其他 5 个策略回测通过（HTF / ME / FBF / LSR / AER）
- [ ] 向后兼容性验证通过（无新配置时能回退到旧路径）
- [ ] Attribution Tags 正常输出（如果已实现 Phase 5）
- [ ] 代码中无 hardcoded `config/nnmultihead/execution_archetypes.yaml`

### 文档指标

- [ ] `README.md` 已更新
- [ ] `ARCHITECTURE.md` 已更新
- [ ] `cmd: 树模型到archetype.md` 已更新
- [ ] `完整命令速查表.md` 已更新
- [ ] 所有 `docs/` 下相关文档已更新

---

## 📊 实施检查表（Checklist）

### 第一周（本周内完成）

**Day 1-2: 配置迁移**
- [ ] 0.1 前置验证：确认当前策略目录结构
- [ ] 1.1 创建 6 个 archetype 目录（bpc/htf/me/fbf/lsr/aer）
- [ ] 2.1 复制 `risk_gate.yaml` 到 `bpc/archetypes/bpc_gate.yaml`
- [ ] 2.2 从 `execution_archetypes.yaml` 提取 BPC 的 Evidence 规则到 `bpc_evidence.yaml`
- [ ] 2.3 从 `execution_archetypes.yaml` 提取 BPC 的 Execution 配置到 `bpc_execution.yaml`

**Day 3-4: 代码修改**
- [ ] 3.1 修改 `load_execution_archetypes_registry()` 函数，支持三层配置加载
- [ ] 3.2 修改 `meta_router_core.py` 调用点
- [ ] 3.3 修改 `meta_router_config.py` 调用点
- [ ] 3.4 修改 `live_feature_plan.py` 调用点
- [ ] 3.5 修改 `cli/main.py` 调用点

**Day 5: 验证回测**
- [ ] 6.1 加载测试：验证 `load_execution_archetypes_registry()` 能成功加载 BPC
- [ ] 6.2 BPC 完整回测：验证 Trades / lift 曲线符合预期
- [ ] 6.3 HTML 报告检查：验证 Failure Analysis / Attribution Tags

### 第二周（下周完成）

**Day 1-3: 批量迁移**
- [ ] 4.1 迁移 HTF archetype（gate / evidence / execution）
- [ ] 4.2 迁移 ME archetype
- [ ] 4.3 迁移 FBF archetype
- [ ] 4.4 迁移 LSR archetype
- [ ] 4.5 迁移 AER archetype
- [ ] 4.6 验证 5 个策略的回测

**Day 4: 脚本修改**
- [ ] 3.6 修改 `scripts/optimize_gate_plateau_hard_gate.py`
- [ ] 3.7 修改其他 10+ 脚本文件的调用点

**Day 5: 清理与文档**
- [ ] 5.1 备份 `config/nnmultihead/` 目录
- [ ] 5.2 从主分支删除 `config/nnmultihead/execution_archetypes.yaml`
- [ ] 5.3 更新所有文档（README / ARCHITECTURE / cmd 等）
- [ ] 5.4 更新 `archetype_config_migration.md`（标记完成）

---

**版本**: v1.0  
**创建时间**: 2026-02-06  
**负责人**: 架构重构  
**预计工时**: 6-8 小时（分两周完成）
