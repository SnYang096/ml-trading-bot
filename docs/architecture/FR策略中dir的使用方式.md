# FR策略中dir的使用方式

> **相关文档**: 参见 [最终简化架构](FINAL_SIMPLIFIED_ARCHITECTURE_2026_01.md) 了解完整的系统架构。

## 一、核心结论

在现有的**NN多头（dir / mfe / mae / mtt）+ Gate + Archetype（TC / TE / FR / ET）**框架中：

- ✅ **TE / TC：方向可以主要用dir（没问题）**
- ⚠️ **FR：方向不能只靠dir，否则一定会错一部分关键交易**
- ❌ **完全不加任何启发式方向判断 → 系统一定会"失去SR反身性"**

👉 **正确做法不是"加 or 不加"，而是：只在FR（SR Reversal）这一支，加极轻量的"结构方向修正"**

不是回到老规则系统，也不是推翻NN。

---

## 二、逐条映射判断

### 1️⃣ SR Breakout / Compression Breakout → TE

**方向用dir大概率不会错，对吗？**

👉 **是的，这里判断是对的，而且是最"安全"的地方。**

原因很简单：

- **TE的本质**：
  - 波动压缩后 → 单边扩展
  - 或新趋势启动
- **这类行情**：
  - dir的统计优势非常明显
  - mfe / mtt也天然配合

在TE里：

```python
TE = 结构允许 + dir置信度 + mfe潜力
```

👉 **不需要额外启发式方向判断**，dir在这里就是"该往哪边炸"。

**结论**：
- ✅ SR breakout ≈ TE
- ✅ Compression breakout ≈ TE
- ✅ 用dir定方向，没问题

---

### 2️⃣ TC（趋势续航）→ 方向用dir

**这个也OK，而且比TE还稳**

TC的前提是：
- 已经在趋势里
- 只是"是否继续持有 / 加仓"

这里dir的角色本来就该是：
- 趋势信念强弱
- 是否值得继续暴露方向风险

👉 **TC = dir的"舒适区"**

**结论**：
- ✅ TC完全可以不加任何启发式方向
- dir + mfe + mae + mtt足够

---

### 3️⃣ FR（SR Reversal）→ 用dir有没有问题？

**这里是关键，也是现在最不确定的地方。**

直接答案：

> ⚠️ **FR只用dir，会"统计上还行，但结构上必然漏掉最值钱的反转"。**

#### 为什么？

因为FR（SR Reversal）本质是：

> **"结构失败后的条件性反转"**

而你现在的dir是：

> **"在当前特征条件下，未来更可能往哪边走"**

这两件事在很多时候是**冲突的**。

---

## 三、一个非常典型的错误场景

* 价格上行，逼近强SR
* dir = 0.65（向上）
* TE / TC都不成立
* 出现：
  * 假突破
  * 上方成交吸收
  * wick + CVD背离

**这时候**：
- 结构上：**最优交易 = FR short**
- 统计上：dir仍然偏多

👉 如果你 **FR方向 = dir**，你会得到：
- 要么不开FR（错过最好反转）
- 要么FR方向错（反而做多）

这不是模型弱，这是**信息层级不同**。

---

## 四、FR的正确机制（极简版）

不是回到老SR系统，而是👇

### Step 1：FR是否允许（现在已经在Gate做得很好）

* adx低
* sr_distance小
* sqs高
* reflexivity风险不过高

### Step 2：FR方向来源（这里是关键）

```python
# 结构方向（非常轻）
if near_upper_sr and rejection_confirmed:
    fr_direction = SHORT
elif near_lower_sr and rejection_confirmed:
    fr_direction = LONG
else:
    return NO_FR
```

👉 **只回答一个问题："如果反转，应该往哪边反？"**

### Step 3：NN dir只做"是否值得信"的裁判

```python
dir_risk = abs(dir_logit)  # 只关心方向性强度，不关心方向

if dir_risk > dir_risk_max:
    skip_fr("strong directional risk")  # veto
else:
    fr_size = base_size * (1 - dir_risk)  # 风险缩放
```

**注意**：
- dir不再决定方向
- dir只决定：**敢不敢做、做多大**

---

## 五、用一句话把TE / TC / FR分清楚（很重要）

* **TE / TC**：
  > "世界接下来更可能往哪边走？" → **dir说了算**

* **FR**：
  > "刚才那条路失败了，现在应该往哪边走？" → **结构说了算，dir只表态度**

---

## 六、dir的正确用法

### ❌ 错误方式（直觉已经觉得不对）

> "dir不支持我这个方向 → 那我FR少做/不做"

这等于在问：

> "趋势模型觉得要涨，我是不是就不能做回调？"

这在策略上是**自我否定**。

---

### ✅ 正确方式：dir只用于**风险缩放（risk scaling）**

FR的逻辑应该是：

> **FR的胜率来自结构，不来自方向**
> 但 **FR的最大亏损来自方向性展开**

所以dir只回答一个问题：

> **"现在顺着某个方向被单边拖走的风险有多大？"**

---

### ✅ 方式一：用 |dir|，不是 dir[fr_direction]

```python
dir_risk = abs(dir_logit)   # 或 abs(dir_prob - 0.5)

if dir_risk > dir_risk_high:
    size = small   # 趋势风险大，FR缩仓
elif dir_risk < dir_risk_low:
    size = normal  # 市场无方向性，FR舒服区
else:
    size = reduced
```

👉 **不关心方向对不对，只关心"有没有方向性"**

---

### ✅ 方式二：dir只作为FR的veto（不是sizing）

```python
if abs(dir_logit) > dir_risk_max:
    skip_fr("strong directional risk")
```

这在极端趋势启动时非常有效（也是现在缺的）。

---

## 七、那FR的"方向"到底从哪来？

**不是dir。**

FR的方向应来自：

* SR几何（在上轨只能做空，在下轨只能做多）
* 距离（normalized sr_distance）
* 结构状态（sqs / compression release）
* 失败形态（fake breakout）

👉 这正是原来的规则类优势区。

**现在的系统进化方向不是"用NN替代规则"，而是：**

> **用NN管风险
> 用规则定结构
> 用execution决定是否真下单**

---

## 八、一句话把现在的直觉"翻译成系统设计语言"

> 之所以觉得
> "dir可能错，那是不是不该用它"
>
> 是因为已经正确地意识到：
> **FR不该用方向模型来确认方向**

这不是bug，是系统成熟的标志。

---

## 九、Mental Model表格

| Archetype | 用不用dir | 怎么用 |
|-----------|----------|--------|
| TE | ✅ 强用 | 方向 + size |
| TC | ✅ 用 | 方向 + risk |
| FR | ⚠️ **不用来定方向** | **只用于风险缩放/veto** |
| ET | ❌ 不用 | 只看风险信号 |

---

## 十、验证方式：如何证明改进有效

### 10.1 对比实验设计

#### Baseline版本（FR直接用dir定方向）

```python
# 错误方式：FR方向 = dir方向
if dir_prob > 0.5:
    fr_direction = LONG
    fr_size = base_size * dir_prob
else:
    fr_direction = SHORT
    fr_size = base_size * (1 - dir_prob)
```

#### 改进版本（FR用结构方向+dir做风险缩放）

```python
# 正确方式：FR方向 = 结构方向，dir只做风险缩放
fr_direction = get_structure_direction(sr_type, rejection_confirmed)  # 结构决定方向
dir_risk = abs(dir_logit)  # 只关心方向性强度，不关心方向

if dir_risk > dir_risk_max:
    skip_fr("strong directional risk")  # veto
else:
    fr_size = base_size * (1 - dir_risk)  # 风险缩放
```

---

### 10.2 评估指标

| 指标 | 说明 | 预期改进 |
|------|------|---------|
| **Sharpe Ratio** | 风险调整后收益 | 改进版应更高（更少错误方向交易） |
| **胜率（Win Rate）** | 盈利交易占比 | 改进版应更高（结构方向更准确） |
| **平均R/R** | 风险回报比 | 改进版应更高（更少被单边拖走） |
| **最大回撤（Max Drawdown）** | 最大亏损幅度 | 改进版应更低（dir风险缩放起作用） |
| **关键场景命中率** | 假突破、SR反转等场景的准确率 | 改进版应显著更高 |
| **方向错误率** | FR方向与最终价格方向相反的比例 | 改进版应显著更低 |

---

### 10.3 具体验证方法

#### 方法1：Ablation实验（推荐）

在同一数据集上，对比两种FR方向判断方式：

```bash
# 运行baseline版本（FR直接用dir定方向）
python scripts/apply_archetype_gate.py \
  --config config/nnmultihead/execution_archetypes_baseline.yaml \
  --logs logs_3action.parquet \
  --out results/fr_dir_validation/baseline_fr_dir.parquet

# 运行改进版本（FR用结构方向+dir做风险缩放）
python scripts/apply_archetype_gate.py \
  --config config/nnmultihead/execution_archetypes_improved.yaml \
  --logs logs_3action.parquet \
  --out results/fr_dir_validation/improved_fr_structure_dir.parquet

# 对比分析
python scripts/compare_fr_direction_methods.py \
  --baseline results/fr_dir_validation/baseline_fr_dir.parquet \
  --improved results/fr_dir_validation/improved_fr_structure_dir.parquet \
  --output results/fr_dir_validation/comparison_report.md
```

---

#### 方法2：关键场景分析

重点分析以下场景的表现：

**1. 假突破场景**：
- 价格上行，逼近强SR
- dir = 0.65（向上）
- 出现假突破、上方成交吸收、wick + CVD背离
- **预期**：改进版应能正确识别FR short，baseline可能错过或做错方向

**2. SR反转场景**：
- 价格在SR附近，结构显示反转信号
- dir与结构方向相反
- **预期**：改进版应遵循结构方向，baseline可能被dir误导

**3. 趋势启动场景**：
- 结构显示FR信号，但dir显示强趋势
- **预期**：改进版应通过dir风险缩放/veto避免逆势交易

---

#### 方法3：归因分析

分析哪些交易是结构方向判断带来的，哪些是dir风险缩放带来的：

```python
# 分析FR交易的归因
fr_trades = get_fr_trades(logs)

# 结构方向判断带来的交易
structure_driven = fr_trades[
    (fr_trades['structure_direction'] != fr_trades['dir_direction']) &
    (fr_trades['structure_direction'] == fr_trades['actual_direction'])
]

# dir风险缩放带来的保护
dir_protected = fr_trades[
    (fr_trades['dir_risk'] > threshold) &
    (fr_trades['size'] < base_size * 0.5)
]

# 对比分析
compare_attribution(structure_driven, dir_protected)
```

---

### 10.4 验证报告模板

验证报告应包含：

1. **整体KPI对比表**（baseline vs 改进版）
2. **关键场景命中率对比**
3. **方向错误率分析**（按场景分类）
4. **归因分析**（结构方向 vs dir风险缩放）
5. **典型案例分析**（展示改进的具体案例）
6. **结论与建议**

---

### 10.5 验证成功的标准

改进版应满足以下标准：

- ✅ **Sharpe Ratio提升 ≥ 0.3**（或baseline为负时，改进版转正）
- ✅ **方向错误率降低 ≥ 30%**
- ✅ **关键场景（假突破、SR反转）命中率提升 ≥ 20%**
- ✅ **最大回撤降低或持平**（不能以牺牲风险控制为代价）
- ✅ **交易数量合理**（不能过度减少，导致样本不足）

---

## 十一、总结

### 核心观点

> **dir是"世界会怎么走的概率"，
> FR是"这次尝试失败后该怎么走的逻辑"。**
>
> 在TE / TC里它们是一致的，
> 在FR里它们天生就不一样。

### 设计原则

- ✅ **需要，但只在FR，而且是"方向修正器"，不是"方向生成器"**
- ❌ **不需要在TE / TC加**
- ❌ **不需要回到完整规则类系统**

现在的框架**本身是对的**，只是差了这一块"SR反身性方向"的最小补丁。

---

## 参考文档

- [最终简化架构](FINAL_SIMPLIFIED_ARCHITECTURE_2026_01.md) - 完整的系统架构说明
- [FR/ET和MEAN_REGIME优化实验](../experiments/EXP_FR_ET_MEAN_REGIME_OPTIMIZATION_2026_01.md) - 相关实验协议
