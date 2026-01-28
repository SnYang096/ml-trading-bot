# Volume Profile WPT 边界划分设计决策

## 一、核心结论（可直接使用）

### 短答案

> **如果你用 Volume Profile 的核心目的是「边界划分」，那——是的，你"可以用" WPT，但前提是：你把它当成"边界稳定器"，而不是默认输入。**

### 一句话判断法（可写在代码注释里）

> **"如果这个 VP 特征是用来'画线'的，可以 WPT；如果是用来'踩线'的，不要 WPT。"**

### 系统公理

> **"当结构的不同尺度解释无法达成一致时，系统不应基于结构进行精细决策。"**

---

## 二、边界划分的本质

### 2.1 VP 中的三类边界

你现在所有和 VP 相关的"边界"，本质上只有三类：

| 边界类型 | 代表特征 | 语义 |
|---------|---------|------|
| **价值边界** | HAL high / HAL low | "多数成交愿意接受的价格区间" |
| **流动性边界** | LVN 的位置 | "成交断层 / 易穿越区" |
| **结构边界** | POC 作为锚点 | "结构中心 vs 偏离" |

⚠️ **共同点**：

> **它们都不是瞬时的，而是统计形成的**

### 2.2 边界最怕什么？

不是偏一点，而是：

- 今天 HAL 在这
- 明天因为几个 tick 抖动，HAL 跳到那
- LVN 数量 / 位置不稳定
- Router / Execution 层收到的是"结构抖动"

**这是边界系统的大敌。**

---

## 三、WPT 在边界划分场景下的价值

### 3.1 WPT 的作用

WPT 在这里做的事非常"对位"：

> **减少价格在相邻 bins 之间的无意义往返**

这带来的直接效果是：

- HAL 上下界更连续
- LVN 更"块状"，而不是锯齿状
- POC 不会因为几根噪音 K 线来回跳 bin

📌 对"边界划分"来说：

> **稳定性 > 灵敏度**

所以在**边界识别这一层**，WPT 是加分项。

### 3.2 边界 vs 信号：关键分界线

🚨 **边界 ≠ 入场信号**

你要特别小心这一点：

- **边界是"场地划线"**
- **信号是"球在哪、往哪踢"**

**WPT 适合**：

- ✅ 划场地
- ✅ 定缓冲区
- ✅ 定风险区 / 允许区 / 禁区

**🚫 不适合**：

- ❌ 判断"刚不刚好碰到边界"
- ❌ 判断"是否发生突破"

👉 这正好对应你现在的系统分层：

- VP → router / execution 参数
- 而不是 → entry trigger

这在架构上是**完全正确的使用方式**。

---

## 四、WPT 在 VP 中的本质作用：什么时候有意义，什么时候多此一举

### 4.1 一句话结论

> **在「Volume Profile」里对 *价格轴* 做 WPT 去高频——不是多此一举，但它只在少数、非常明确的目标下才有意义；如果目标没对齐，确实可能是"工程感很强、alpha 很弱"。**

你这套代码已经不是"随手加个小波"，而是**体系化地把 VP 当成一个"结构估计器"**在用，所以值得认真拆清楚。

---

### 4.2 WPT 在 VP 里到底在干嘛

**Volume Profile 本质是**：

> **对「价格轴」做一个带权密度估计（权重 = volume）**

你现在做的是：

```
price_raw(t)
   ↓ WPT（去高频）
price_smooth(t)
   ↓ histogram(price_smooth, weights=volume)
VP(price)
```

⚠️ **注意**：
**你没有对 volume 去噪，而是只动了 price。**

这意味着 WPT 的作用不是"平滑成交量"，而是：

> **减少价格在相邻 bin 间的抖动跳跃，从而稳定 VP 的"结构形态"**

---

### 4.3 什么时候这是「有意义的」

#### ✅ 情况 A：你把 VP 当成【结构锚点】，不是微观信号

在你现在的使用方式里，VP 主要产出的是：

- POC（长期成交锚）
- HAL（价值区间）
- HVN / LVN（流动性结构节点）
- 以及它们的**相对位置关系**（distance / density）

这些特征本质上是：

> **低频、结构性、区间级别的信息**

👉 在这种前提下：

- 原始价格里的 **微小锯齿**
- 会被 FD bins + histogram **放大成"假 LVN / 假 HVN"**
- 尤其是在 bins=auto 时更明显

此时你用 WPT 去掉最高频 25% 子带，本质是在做：

> **"结构层面的核函数平滑"**

不是为了"更准"，而是为了**更稳、更可复现**

✅ 这是**有意义的**

---

#### ✅ 情况 B：你在做「跨市场 / 跨标的」的 VP 归一化特征

你后面大量用了：

- `vp_lvn_distance / price_range`
- `vp_volume_density`
- `price_to_poc_pct`

这些都是**无量纲、可迁移特征**。

但有个前提：

> **POC / LVN 本身要稳定，否则"无量纲"只是在放大噪音**

WPT 在这里的价值是：

- 减少不同 market microstructure
  （BTC vs 山寨 vs 外汇）
- 在 VP 结构层引入的随机性

👉 如果你的目标是 **structure-generalization**
WPT 是合理的工程手段。

---

### 4.4 什么时候它就「真的多此一举」

#### ❌ 情况 A：你把 VP 当作短线 timing / 微结构信号

如果你希望 VP 回答的是：

- "这里是不是刚形成一个 LVN？"
- "刚刚有没有新的成交真空？"
- "当前 price 是否在刚刚形成的 volume gap 边缘？"

那 **WPT 是反作用的**：

- 它会**延迟结构形成**
- 把本来有用的微观 discontinuity 抹平
- 让 VP 更像一条"慢热的历史统计"

👉 在 **≤5min / tick / footprint** 语境下
**WPT ≈ alpha killer**

---

#### ❌ 情况 B：你已经用 bins / rolling / forward-fill 控制了噪音

你现在已经有：

- FD bins（IQR 抗异常）
- window = 100 ~ 160（不是短窗）
- ffill（结构延续假设）

在这种情况下：

> **WPT 带来的"结构稳定增益"很可能是二阶的**

而它的代价是：

- 引入 wavelet / level / drop_ratio 超参数
- 隐含时间尺度假设（2^level）
- 增加调参自由度（= 回测幻觉风险）

---

### 4.5 站在「你这个系统」的角度，我的判断

结合你以往的讨论（router / regime / structure first），我会给你一个**不模糊的建议**：

#### 🎯 **WPT 不该是 VP 的默认，而应该是一个 regime-aware option**

**推荐策略**：

1. **默认 VP = raw price**
2. **仅在"结构决策层"启用 WPT VP**

   - 用于：

     - regime 判定
     - 执行模式选择（MEAN vs TREND）
     - position sizing / risk shaping

3. **不要让 WPT VP 直接参与 entry timing**

你现在的代码已经很好地做到了这一点：

- WPT 封装在 `compute_wpt_volume_profile`
- 上层只消费结构特征
- 没把 denoised price 当交易价格用

这是**成熟系统的写法** 👍

---

### 4.6 一个很关键但容易忽略的点（⚠️）

你现在是：

```python
freq_nodes = wp.get_level(actual_level, "freq")
drop top 25% 高频子带
```

这 **隐含了一个强假设**：

> **"价格噪音 ≈ 最高频子带"**

在 crypto 的某些 regime（尤其是流动性薄、跳变多的币）：

- **真实信息本身就在高频**
- 被你当噪音丢掉了

👉 如果你哪天发现：

- VP 结构"过于平滑"
- LVN 变少、distance 变大
- mean-reversion 信号失效

**第一个该怀疑的，不是 bins，也不是 window，而是这个 drop_high_freq**

---

### 4.7 最终总结（可执行结论）

- ❌ **不是普适 alpha enhancer**
- ✅ **是结构稳定器**
- ✅ **对你这种"VP → 派生无量纲结构特征 → router"体系是合理的**
- ⚠️ **不要默认启用**
- 🎯 **把它当成：结构视角滤镜，而不是信号生成器**

如果你愿意，下一步我可以直接帮你设计一个：

> **raw VP vs WPT VP 的结构一致性诊断指标（非收益）**

用来判断：
👉 *"WPT 到底是在帮你，还是在替你决定市场该不该波动"*

---

## 五、vp_boundary_stability_score：结构一致性指标

### 4.1 核心思想

> **比较「raw VP」与「WPT VP」得到的边界是否一致，用一致性来判断：当前市场是"结构主导"还是"噪音 / 过渡主导"。**

你要的不是方向，而是 **"现在边界靠不靠谱"**。

### 4.2 推荐的最小完备定义

只用 **HAL 边界**，不要一开始就混 LVN：

```python
Δ_high = |HAL_high_raw − HAL_high_wpt|
Δ_low  = |HAL_low_raw  − HAL_low_wpt|
```

归一化（非常关键）：

```python
range = price_max − price_min

d = (Δ_high + Δ_low) / (2 * range)
```

最后映射成一个 **稳定性分数（越大越稳）**：

```python
vp_boundary_stability_score = exp(− d / τ)
```

- `τ`：温度参数（建议 0.05 ~ 0.1）
- score ∈ (0, 1]

### 4.3 直觉解释

- raw ≈ WPT → d 很小 → score ≈ 1 → **结构一致**
- raw ≠ WPT → d 大 → score → 0 → **噪音 / 过渡**

### 4.4 计算架构

在 **同一个 window** 内：

```python
raw_vp  = compute_volume_profile(price_raw)
wpt_vp  = compute_volume_profile(price_wpt)

score = boundary_consistency(raw_vp, wpt_vp)
```

⚠️ **注意**：

- **这个 score 不参与交易**
- 它是 **router 的 meta-input**

---

## 五、系统语义映射

### 5.1 score 区间的市场状态语义

| score 区间 | 市场状态语义 |
|-----------|------------|
| > 0.75 | 结构稳定、边界可信 |
| 0.4–0.75 | 结构在，但开始松动 |
| < 0.4 | 微观噪音主导 / regime 过渡 |

这不是技术判断，这是**行为判断**。

### 5.2 映射到 6 种 archetype

根据系统语义，6 种 archetype 可以分成三大类：

---

#### 🧱 A 类：边界依赖型（MEAN / RANGE / FADE）

**Archetypes**：
- `FailedBreakoutFade` (FBF)
- `LiquiditySweepRejection` (LSR)
- `AuctionExhaustionReversal` (AER)

**典型行为**：
- 在 HAL / LVN 附近做事
- 假设边界是"力场"

**路由规则（强）**：

```python
if vp_boundary_stability_score < 0.6:
    禁用 A 类 archetype
```

或者软一点：

```python
weight_A = sigmoid((score − 0.6) / 0.1)
```

📌 **解释**：

> 边界都不一致了，你还在边界上做 mean，是自杀。

---

#### 🚀 B 类：方向 / 趋势型（BREAK / TREND / EXPAND）

**Archetypes**：
- `BreakoutPullbackContinuation` (BPC)
- `MomentumExpansion` (ME)
- `HTFBiasLTFEntry` (HTF)

**典型行为**：
- 假设结构正在被破坏
- 对边界不敏感，甚至"吃掉边界"

**路由规则（反而偏好低分）**：

```python
if vp_boundary_stability_score < 0.5:
    提高 B 类 archetype 权重
```

📌 **解释**：

> raw 和 WPT 不一致，本身就是"结构撕裂"的信号。

---

#### 🧊 C 类：防御 / 缩手型（NO_TRADE / WAIT / REDUCE）

**典型行为**：
- 不赌方向
- 不信结构

**路由规则（双低区间）**：

```python
if score < 0.3 and volatility ↑:
    进入 C 类（NO_TRADE）
```

📌 **解释**：

> 这是结构真空期，不是机会期。

---

## 七、集成到现有 Router（不推翻任何东西）

### 6.1 最佳位置

你现在已经有：

- regime score
- path head
- execution constraints

**vp_boundary_stability_score 的最佳位置是：**

> **作为"结构可信度调制因子"**

### 6.2 示例实现（概念）

```python
# 对 A 类（边界依赖型）
score_mean *= stability_score

# 对 B 类（趋势型）
score_trend *= (1 − stability_score)  # 或使用其他映射

# 最终路由决策
final_score = base_archetype_score * modulation_factor
```

### 6.3 关键理解

> **你不是用 VP 预测市场，而是用 VP 判断："我该不该相信我自己现在用的结构假设。"**

这一步极其关键：

> **系统是否允许自己"知道何时不知道"**

`vp_boundary_stability_score` 正是那个"知道自己不知道"的信号。

---

## 七、可执行的使用准则

### ✅ 建议你这样用 WPT VP

**1️⃣ 只用于边界类输出**

- `vp_hal_high / low`
- `vp_lvn_distance`
- `vp_price_in_lvn`
- 结构状态判断

**2️⃣ window 要 ≥ 一个"结构周期"**

- 你现在 100–160 是合理的
- 太短会让 WPT 反而引入伪结构

**3️⃣ 永远保留 raw VP 作为对照**

- 不是给模型
- 是给你自己做 sanity check

### ❌ 不建议你这样用

- ❌ 用 WPT VP 的 HAL 做精确止损
- ❌ 用 WPT VP 判断"刚刚突破"
- ❌ 在 execution tick 级别依赖它

---

## 九、实现建议

### 8.1 代码位置

**当前实现**：
- `src/features/time_series/utils_volume_profile.py`：WPT VP 计算
- `config/nnmultihead/execution_archetypes.yaml`：gate rules 中使用 VP 特征

**需要添加**：
- `vp_boundary_stability_score` 特征计算函数
- Router 中的稳定性调制逻辑

### 8.2 实现步骤

**第一步：添加 `vp_boundary_stability_score` 特征**

在 `utils_volume_profile.py` 中添加：

```python
def compute_boundary_stability_score(
    raw_vp: VolumeProfileResult,
    wpt_vp: VolumeProfileResult,
    price_min: float,
    price_max: float,
    tau: float = 0.075,
) -> float:
    """
    计算 raw VP 和 WPT VP 的边界一致性分数。
    
    Args:
        raw_vp: Raw volume profile result
        wpt_vp: WPT volume profile result
        price_min: Minimum price in window
        price_max: Maximum price in window
        tau: Temperature parameter (default: 0.075)
    
    Returns:
        Stability score in (0, 1], where 1 = perfect consistency
    """
    # 计算 HAL 差异
    delta_high = abs(raw_vp.hal_high - wpt_vp.hal_high)
    delta_low = abs(raw_vp.hal_low - wpt_vp.hal_low)
    
    # 归一化
    price_range = price_max - price_min
    if price_range <= 0:
        return 0.0
    
    d = (delta_high + delta_low) / (2.0 * price_range)
    
    # 映射到稳定性分数
    score = np.exp(-d / tau)
    return float(np.clip(score, 0.0, 1.0))
```

**第二步：在 Router 中集成**

在 `meta_router_strategy.py` 或 `meta_router_core.py` 中：

```python
# 获取稳定性分数
stability_score = features.get("vp_boundary_stability_score", 1.0)

# 根据 archetype 类型调制
if archetype_name in ["FailedBreakoutFade", "LiquiditySweepRejection", "AuctionExhaustionReversal"]:
    # A 类：边界依赖型
    if stability_score < 0.6:
        continue  # 跳过这个 archetype
    confidence *= stability_score
elif archetype_name in ["BreakoutPullbackContinuation", "MomentumExpansion", "HTFBiasLTFEntry"]:
    # B 类：趋势型（低稳定性反而偏好）
    if stability_score < 0.5:
        confidence *= (1.0 - stability_score * 0.5)  # 适度提升
```

**第三步：添加到特征计算流程**

在 `compute_unified_volume_profile_features` 中同时计算 raw 和 WPT VP，然后计算稳定性分数。

---

## 九、验证方法

### 9.1 观察指标

1. **边界稳定性**
   - 统计 HAL 位置的跳跃频率
   - 对比 raw VP 和 WPT VP 的 HAL 差异分布

2. **Archetype 路由效果**
   - 低稳定性时，A 类 archetype 是否被正确禁用
   - 高稳定性时，A 类 archetype 的表现是否改善

3. **系统整体表现**
   - 在结构过渡期（低稳定性）是否减少无效交易
   - 在结构稳定期（高稳定性）是否提高交易质量

### 9.2 对比实验

| 实验组 | 控制组 |
|--------|--------|
| 启用 `vp_boundary_stability_score` 调制 | 不使用稳定性分数 |

**观察**：
- A 类 archetype 在低稳定性时的交易频率
- 整体 Sharpe 和盈亏比变化
- 结构过渡期的回撤控制

---

## 十一、当前系统状态

### 10.1 已有实现

- ✅ WPT Volume Profile 计算（`utils_volume_profile.py`）
- ✅ Volume Profile 特征在 gate rules 中使用（`fbf_volume_profile_rejection`）
- ✅ 6 种 archetype 已定义并配置

### 10.2 待实现

- ⏳ `vp_boundary_stability_score` 特征计算
- ⏳ Router 中的稳定性调制逻辑
- ⏳ 特征计算流程中同时计算 raw 和 WPT VP

---

## 十二、总结

### 11.1 核心判断

**WPT 在边界划分场景下是有价值的，但必须明确其角色：**

- ✅ 作为"边界稳定器"
- ❌ 不作为默认输入或精确信号

### 11.2 系统层级理解

> 这一步问得**非常到位**。你现在其实已经站在「**结构 → regime → archetype**」这一层，而不是"再加一个指标"。

`vp_boundary_stability_score` 会让你的 6 archetype **第一次真正有"使用条件"**，而不是并列竞争。

### 11.3 最终目标

> **系统是否允许自己"知道何时不知道"**

`vp_boundary_stability_score` 正是那个"知道自己不知道"的信号。

---

## 十二、相关文档

- `src/features/time_series/utils_volume_profile.py`：Volume Profile 计算实现
- `config/nnmultihead/execution_archetypes.yaml`：Archetype 配置和 gate rules
- `docs/experiments/EXP_ET_VOLUME_PROFILE_EFFECTIVENESS_2026_01.md`：ET 为什么需要 Volume Profile
- `docs/architecture/NNMULTIHEAD_FEATURE_CONTRACT_BLOCK_GATING.md`：Feature contract 和 block gating

---

**文档版本**：v1.1  
**创建日期**：2026-01-27  
**最后更新**：2026-01-27
