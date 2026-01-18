# Live Feature Contract & Evidence（为什么需要“缺失策略”、证据字段怎么来的）

> Feature Contract 目前集成到 feature_plan 配置文件终，因为基本不会变

本文回答三个常见疑问：

- **Q1：同一套缺失策略有没有必要？不是“不可能缺”吗？**
- **Q2：`has_orderflow` / `has_sr_quality` 这些证据字段现在实现在哪里？要手写吗？**
- **Q3：`live_feature_contract.yaml` 是不是给实盘用的？**

---

## 1) 为什么“缺失策略”有必要（即便你认为“不可能缺”）

你说“理论上不可能缺”，在离线研究里经常成立；但在实盘里，**缺失是必然会发生的**，区别只在于：
- 是以“显式、可控、可审计”的方式发生（可接受）
- 还是以“隐式、不可控、误下单”的方式发生（不可接受）

### 1.1 缺失的来源不是“你代码写错”，而是“实盘世界不稳定”

下面是几个真实且常见的 **“feature 变缺”** 场景（不是 bug，而是环境现实）：

- **例子 A：订单流断流 / tick 丢包**
  - 现象：交易所 websocket 抖动、重连、某段时间 trade tick 为空
  - 结果：`vpin/cvd/delta/footprint` 等 orderflow 特征不可用
  - 风险：如果系统“默认当作 0”继续交易，会把“没有 orderflow 证据”当成“orderflow 证据弱”，语义错了

- **例子 B：计算延迟（timer 到了但特征还没更新）**
  - 现象：你每 10min timer 决策一次，但上一周期的 bar/ tick 还没完整写入缓存
  - 结果：本周期 features 不完整（“短缺”）
  - 风险：模型输入等价于“随机缺列”，推理输出不稳定

- **例子 C：版本升级/配置变更导致列名变更**
  - 现象：策略升级把 `ofi_short` 改名为 `ofi_15m`
  - 结果：证据 DSL 的 `any_key_contains` 或 `value_gt` 对不上 key
  - 风险：白名单要求的证据没满足却误通过，或相反误拒绝

### 1.2 “缺失策略”的目标：让缺失变成“明确的、可验证的降级”

实盘最重要的是：当缺失发生时，系统必须能回答：
- **缺了什么？**
- **为什么缺？**
- **因此我采取了什么降级？（NO_TRADE / THROTTLE / VETO）**

这就是“同一套缺失策略”的意义：**不是为了日常运行**，而是为了把极端/意外从“不可控风险”降级为“可控行为”。

---

## 2) `has_orderflow` / `has_sr_quality` 证据字段：实现在哪里？需要手写吗？

结论：**不需要手写 if/else**，大部分是 **YAML 配置 → Evidence DSL → runtime 计算** 的链路。

### 2.1 证据字段在哪里定义？

它们同时出现在两处配置（职责不同）：

- **A) Execution Archetype Registry（定义“证据怎么计算”）**
  - 文件：`config/nnmultihead/execution_archetypes.yaml`
  - 每个 archetype 有：
    - `required_evidence`: 这个 archetype 必须满足哪些证据
    - `evidence_rules`: 证据如何从 `features: Dict[str, Any]` 推导出来（DSL）

- **B) Execution Whitelist Constitution（定义“证据不满足就硬拒绝”）**
  - 文件：`config/constitution/execution_whitelist.yaml`
  - 每个 regime 下每个 strategy 的 `required_evidence`：**硬门禁**

### 2.2 证据字段如何计算（代码入口）

核心实现：
- `src/time_series_model/core/constitution/execution_evidence.py`
  - 函数：`compute_execution_evidence(features=..., rules=...) -> Dict[str, bool]`
  - 支持 rule kinds：
    - `any_key_contains`（当前 `has_orderflow/has_sr_quality` 就是用这个）
    - `key_exists`
    - `value_gt/gte/lt/lte`
    - `abs_gt`
    - `on_missing` 策略（`false|true|error`）

### 2.2.1 目前是不是“字符串匹配”？是的（但这是可演进的）

当前 repo 里 `has_orderflow/has_sr_quality` 的默认规则使用的是：
- `kind: any_key_contains`

它的含义是：**只要 features 字典里出现过某类“关键字风格”的 key，就认为这个证据成立**。

这不是最终形态，而是一个“先跑通、严格保守”的 v1：
- 好处：不依赖具体某个指标的精确数值与归一化细节；更像“数据通路存在性检查”
- 风险：如果 key 命名误匹配、或者语义漂移（同名但定义不同），证据可能“形式上成立但实际无效”

因此后续建议把关键证据逐步升级成 value-level 的 DSL：
- 例如：`value_gt(key=vpin, threshold=0.4)` 才算 `has_orderflow`
- 或：`key_exists(key=sr_score)` + `value_gte(threshold=...)` 才算 `has_sr_quality`

这种升级不需要改 Python 代码，只需要改 YAML（`execution_archetypes.yaml`）并补测试即可。

在 live 策略里调用位置：
- `src/time_series_model/live/meta_router_strategy.py`
  - 在下单前会计算 `evidence = compute_execution_evidence(...)`
  - 然后传给 `enforce_before_order(..., execution_evidence=evidence)`

### 2.3 证据字段如何被宪法强制执行（硬拒绝）

门禁实现：
- `src/time_series_model/core/constitution/execution_whitelist.py`
  - `enforce_execution_whitelist(...)`
  - 会检查：
    - regime 是否允许这个 strategy
    - 是否命中 forbidden keyword
    - **required_evidence 是否存在且为 True（缺 or False 都拒绝）**

live enforcement hook：
- `src/time_series_model/live/enforcement.py`
  - `enforce_before_order()` 内部调用 `executor.validate_execution_strategy(...)`

### 2.4 一个具体例子（`has_orderflow` / `has_sr_quality`）

在 `config/nnmultihead/execution_archetypes.yaml` 中（简化理解）：

- `has_orderflow`：只要 `features` 的 key 里出现过 `vpin/cvd/delta/...` 任意一个子串，就认为有 orderflow 证据
- `has_sr_quality`：只要 key 里出现 `sr_` / `sqs_` / `poc_` 这类子串，就认为有 SR 质量证据

然后 whitelist 会要求例如：
- `FailedBreakoutFade` 必须同时满足 `has_orderflow` 和 `has_sr_quality`
  - 否则：**硬拒绝下单**

---

## 3) `live_feature_contract.yaml` 是给实盘用的吗？

是的，定位是：**实盘侧的“特征输入契约”**（类似于 API contract）。

它的作用不是“训练消融”，而是：
- 明确 **live 必须提供哪些特征**（按来源：on_tick/on_bar/on_timer）
- 明确 **窗口/对齐/更新时间**（避免语义不一致）
- 明确 **缺失处理策略**（硬拒绝、降级、或允许缺但打 mask）
- 明确 **证据字段所依赖的特征集合**（否则证据会漂）

你可以把它理解为：把“训练时 features.yaml 的要求”迁移成“实盘时 features 必须满足的运行时约束”，并且提供可审计的降级行为。

### 3.1 为什么 Evidence/Whitelist 已经实现了，还需要 Live Feature Contract？

这两者解决的是不同层级的问题：

- **Evidence DSL + Whitelist enforcement**：解决“下单前能否执行某个 archetype”（硬门禁）
  - 输入假设：已经有一个 `features: Dict[str, Any]`（但不保证它完整/一致）
- **Live Feature Contract**：解决“features 本身是否可信/一致”（运行时契约）
  - 目标：把 *哪些特征必须存在、怎么更新、缺失如何降级、窗口怎么对齐、哪些证据依赖哪些输入* 变成可配置、可审计、可回放的 contract

如果没有 live contract，会出现一种最危险的情况：
- whitelist 看起来在工作
- 但 features 的语义已经漂了（例如窗口不同、对齐不同、断流导致“默认为 0”）
- 最终门禁变成“形式正确、实际无效”

> 备注：仓库里已提供 `live_feature_contract.yaml` 作为默认样例；如果你的 live 特征集不同，请按需裁剪/扩展。

### 3.2 当前仓库的落地状态（v1）

已落地的最小版本：
- **配置**：`config/live/live_feature_contract.yaml`
- **校验器**：`src/time_series_model/live/live_feature_contract.py`
- **接入点**：`src/time_series_model/live/meta_router_strategy.py` 的 timer 决策入口（`_on_signal_check`）
  - 行为：若 contract 不满足 → **强制 NO_TRADE**（本周期不下单）并记录原因到日志
  - 环境变量覆盖：`MLBOT_LIVE_FEATURE_CONTRACT_YAML`（默认指向上述 YAML）
