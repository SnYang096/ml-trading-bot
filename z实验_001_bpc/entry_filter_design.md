# Entry Filter 正交设计原理

> 结论：Entry Filter 不需要树模型 + plateau 机制，用启发式规则 + 少量阈值扫描更合理。
> 这不是"省事"，而是层级职责 + 可学习性边界判断的结果。

---

## 一、四层架构正交性

| 层 | 核心问题 | 输出类型 | 优化机制 | 主 KPI |
|---|---------|---------|---------|--------|
| Gate | "能不能做？" | veto / 降权 | 树模型 → Lift plateau | Lift, pass_rate |
| Entry Filter | "现在该进吗？" | 等 / 不等（硬二值） | 启发式 + 阈值 plateau 扫描 | snotio, loss_rate, stop_rate |
| Evidence | "下多少注？" | 连续强度 [0,1] | 树模型 → bad_suppression plateau | bad_suppression |
| Execution | "怎么执行？" | SL/TP/trailing 参数 | 网格搜索 plateau | Sharpe |

### 正交性保证

```
Gate:         状态 → 好/坏 的映射（需要学习，非直觉切分）
Evidence:     状态 → 程度 的映射（需要学习，连续空间找 bins）
Entry Filter: 结构前提 → 是/否（不需要学习，语义已完成）
Execution:    参数空间 → 稳健组合（网格搜索，不需要模型）
```

**关键区分**：Gate / Evidence 用树模型，是因为映射是非直觉的（如 wpt_ignition 高反而坏）、需要在连续空间找切分点。Entry Filter 的特征已经是"语义完成态"（如 `bpc_was_in_pullback`, `cvd_absorption_detected`），不需要模型再学一遍。

---

## 二、Entry Filter 为什么不适合树模型

### 2.1 特征是"语义完成态"

Entry Filter 使用的特征：
- `bpc_was_in_pullback` — bool，是否在回踩中
- `bpc_pullback_depth` — 回踩深度（0-1，已归一化）
- `bpc_cvd_absorption` — bool，CVD 吸收是否确认
- `bpc_vol_pullback_confirm` — 缩量确认程度
- `wick_absorption_score` — 影线吸收得分

这些不是 raw feature，而是**已经把结构理解编码进去的指标**。树模型最怕的就是"你已经告诉我答案，我却还要假装学习"。

### 2.2 Label 设计困境 — 不可学习

| 候选 Label | 问题 |
|-----------|------|
| forward_rr | 退化成 Evidence（RR 是 Evidence 的职责） |
| Sharpe-based | 无法定义在单 bar 上 |
| TP-first | 退化成 Gate（TP 到达是结构问题） |

**这是一个"不可学习"的任务**：入场好不好不是入场当下的属性，而是整个交易路径的结果。这个结果已经被 Evidence（RR/结构质量）和 Execution（止损/追踪）建模了。

### 2.3 "必要但非充分"的本质

Entry Filter 回答的是：
> "如果现在不满足这个条件，几乎一定不该进；但满足了，也不保证一定值得进。"

用 BPC 场景说明：

- ❌ 没 pullback + 吸收 → 几乎一定是追高/噪音 → **不该进**
- ✅ 有 pullback + 吸收 → 结构允许入场，但 RR 可能差、上方 SR 很近、大周期不一致…

满足 ≠ 一定好，只是"有资格被考虑"。

| 层 | 逻辑地位 |
|---|---------|
| Gate | 必要条件（否定式）：不满足 → 禁止 |
| Entry Filter | 必要条件（资格式）：不满足 → 等待 |
| Evidence | 强度条件（程度式）：满足程度 → 下多少 |

Entry Filter 不需要预测未来，只需要回答：**"现在进，会不会在结构上显得很愚蠢？"**

---

## 三、Entry Filter vs Guardrail 边界

机制相同（feature + threshold → binary），语义职责不同：

| 维度 | Guardrail | Entry Filter |
|------|-----------|-------------|
| 问的问题 | "违反策略前提了吗？" | "现在是最佳入场时机吗？" |
| 执行位置 | Gate 阶段（hard gate 之后） | Execution 阶段（gate 放行之后） |
| 否决语义 | **永久否决** — 不该做 | **暂时等待** — 可以做但等更好时机 |
| 特征类型 | 策略前提特征 | 入场时机特征 |
| 阈值来源 | 策略定义的常识（固定） | 回测优化（plateau 扫描） |
| 是否参与优化 | 不参与（同义反复） | 可参与 |

同一条件轴上的不同层级：

```
bpc_volume_compression_pct:
[0] ---- [0.3 Guardrail: deny] ---- [0.7 Entry Filter: 缩量确认] ---- [1.0]
          ↑ 违反策略前提               ↑ 最佳入场时机
```

**不应合并**：Gate 层的 deny 影响所有下游（包括 Evidence 评分），Entry Filter 只影响入场时机。

---

## 四、Entry Archetype 层级结构

当前系统实际的层级：

```
Strategy Archetype（策略大类: BPC / SR / Trend）
└── Entry Archetype（入场动机类型）
    └── Sub-Entry Archetype（微结构变体 = entry_filters.yaml 中的 filter）
```

### 4.1 Entry Archetype（动机分类）

| # | Entry Archetype | 入场动机 |
|---|----------------|---------|
| 1 | Pullback Continuation | 突破后回踩底部续涨 |
| 2 | Compression Breakout | 压缩收敛后爆发 |
| 3 | Absorption Fade | 卖方力量被吸收后反转 |
| 4 | Failed Breakdown Reversal | 假跌破后反弹 |
| 5 | Trend Ignition | 趋势启动点 |
| 6 | Mean-Reversion Exhaustion | 极端偏离后回归 |

### 4.2 BPC Sub-Entry Archetype（当前已实现）

在 Pullback Continuation 下的微结构变体：

| Sub-Archetype | 核心附加条件 | Sharpe | Trades |
|--------------|-----------|--------|--------|
| deep_pullback | pullback ≥ 0.6 | 0.326 | 962 |
| deep_pullback_cvd | + CVD 吸收 | 0.332 | 527 |
| deep_pullback_vol | + 缩量确认 | 0.474 | 224 |
| deep_pullback_wick | + 影线吸收 | 0.414 | 150 |
| deep_pullback_bb | + BB 压缩 | 0.395 | 356 |
| deep_pullback_wpt | + WPT 吸收 | 0.391 | 178 |
| deep_pullback_momentum | + 动量恢复 | 0.413 | 177 |
| deep_pullback_liq_void | + 流动性缺口 | 0.455 | 193 |
| deep_pullback_full | 全条件组合 | 0.355 | 78 |

这些 Sub-Archetype：
- **共享入场动机**（都是 pullback continuation）
- 但**微结构不同**（CVD 吸收 vs 缩量 vs 影线）
- 不应再拆成树模型（差异是结构语义，不是连续特征的最佳切分）

---

## 五、阈值 Plateau 扫描（已实现）

Entry Filter 的阈值优化使用 plateau 扫描（不是树模型）：

```bash
python scripts/optimize_entry_filter_plateau.py \
    --logs results/train_final_*/bpc/predictions.parquet \
    --strategy bpc
```

### 5.1 Plateau 判定机制

**双 CV 约束**（Entry Filter 特有）：
- Sharpe CV < 0.3 — 收益稳定性
- Trades CV < 0.4 — 执行节奏稳定性（Sharpe 稳但 trades 剧变 → 实盘节奏异常）

当双 CV 均不满足时，回退到仅 Sharpe CV 并标注 `⚠️ Trades CV>0.4`。

### 5.2 Recommended 选点：偏宽容侧 20%

**不取中点，取 plateau 20% 处**：

| 条件类型 | 计算方式 | 语义 |
|---------|---------|------|
| `>=` / `>` | `start + 20% * width` | 低阈值更宽容，偏左取 |
| `<=` / `<` | `end - 20% * width` | 高阈值更宽容，偏右取 |

**为什么 Entry Filter 要偏宽容侧？**

plateau 的存在证明了「严格 ≠ 更好」。三个具体例子：

**例 1: `bpc_pullback_depth >= x`**
```
plateau = [0.49, 0.71], Sharpe ≈ 0.32 ± 0.01
中点 = 0.60 → "必须是相当深的回踩"
20%点 = 0.53 → "确认在深回踩区就够了" ← 更合理
```
pullback depth 是估计量，回踩底部经常在 0.50-0.60 报动。设 0.60 会挡掉很多「结构上已经对了」的 K 线，但 Sharpe 并没因此显著提高。

**例 2: `bpc_vol_pullback_confirm > x`**
```
plateau = [0.70, 0.90], Sharpe ≈ 0.47 ± 0.03
0.70 → 600 trades,  0.90 → 180 trades
中点 = 0.80 → "必须非常明显的缩量"
20%点 = 0.74 → "缩量已经开始就够了" ← trades 翻倍，Sharpe 不变
```

**例 3: `wick_absorption_score > x`**（影线 = 最容易报动的特征）
```
plateau = [0.15, 0.35], Sharpe ≈ 0.45 ± 0.03
中点 = 0.25 → "必须是很标准的锤子线"
20%点 = 0.19 → "有吸收迹象就够了" ← 实盘更稳
```

**核心原则**：Entry Filter 的错误成本是「错过」，不是「多等一次确认」。在 Sharpe 无本质差别的区间内，选更容易触发入场的一侧。

### 5.3 置信度等级

Plateau 宽度 → 置信度（宽度越大 = decision boundary 曲率越低 = 越可部署）：

| 宽度 | 置信度 | 含义 |
|------|--------|------|
| ≥ 0.3 | HIGH | 可直接部署，阈值抗扰动强 |
| ≥ 0.15 | MEDIUM | 可部署，建议定期监控 |
| < 0.15 | LOW | 谨慎使用，易受市场状态变化影响 |

### 5.4 扫描结果（2026-02-09，双 CV + 20% 偏宽容）

| Filter | 条件 | 当前 | 推荐 | 高原范围 | width | conf |
|--------|------|------|------|---------|-------|------|
| deep_pullback | pullback_depth ≥ | 0.6 | 0.543 | [0.49, 0.71] | 0.228 | MED |
| deep_pullback_cvd | pullback_depth ≥ | 0.6 | 0.543 | [0.49, 0.71] | 0.228 | MED |
| deep_pullback_vol | pullback_depth ≥ | 0.6 | 0.486 | [0.43, 0.66] | 0.228 | MED |
| deep_pullback_vol | vol_confirm > | 0.7 | 0.750 | [0.70, 0.90] | 0.200 | MED |
| deep_pullback_wick | pullback_depth ≥ | 0.6 | 0.543 | [0.49, 0.71] | 0.228 | MED ⚠️ |
| deep_pullback_wick | wick_absorption > | 0.3 | 0.200 | [0.15, 0.35] | 0.200 | MED |
| deep_pullback_bb | pullback_depth ≥ | 0.6 | 0.600 | [0.54, 0.77] | 0.228 | MED |
| deep_pullback_bb | bb_compression > | 0.7 | 0.750 | [0.70, 0.90] | 0.200 | MED |
| deep_pullback_wpt | pullback_depth ≥ | 0.6 | 0.486 | [0.43, 0.66] | 0.228 | MED |
| deep_pullback_wpt | wpt_absorption > | 0.2 | 0.171 | [0.13, 0.30] | 0.171 | MED ⚠️ |
| deep_pullback_momentum | pullback_depth ≥ | 0.6 | 0.429 | [0.37, 0.60] | 0.229 | MED |
| deep_pullback_liq_void | pullback_depth ≥ | 0.6 | 0.429 | [0.37, 0.60] | 0.229 | MED |

⚠️ = Trades CV > 0.4（仅用 Sharpe CV 回退判定）

核心发现：当前 `pullback_depth ≥ 0.6` 在多数 filter 中处于高原中央偏右，推荐值在 0.43-0.54 — **当前 0.6 偏严格，可下调至 0.5 左右以捕捉更多有效入场点**。

---

## 六、KPI 体系

Entry Filter 的失败模式不是"亏钱"，而是"过滤太干净，系统失血"。

### 层级 KPI 定位

| 层 | 主 KPI | 辅助 | 说明 |
|---|--------|------|------|
| Entry Filter | **snotio** (mean R) | loss_rate, stop_rate | 不受 trade count 影响 |
| Evidence | failure rate / drawdown contribution | bad_suppression | 下注质量 |
| Execution | Sharpe / Calmar / MDD | per-symbol equity | 参数稳健性 |
| 全系统 | OOS Sharpe | — | 最后看 |

### Entry Filter 主 KPI: snotio

```
snotio = mean(R-multiples) = 平均每笔交易的风险调整收益
```

**为什么不用 Sharpe?**
1. Sharpe 被 √(trades) 人为抬高，trades 极多时几乎只在度量 "execution 磨正期望的能力"
2. Entry Filter 目标是 "避免低性价比交易"，snotio 与此目标对齐
3. snotio 不受 trade count 影响，与 Execution 正交

### Entry Filter 辅助 KPI: Loss Rate / Stop Rate

```python
loss_rate = mean(R < 0)        # 亏损交易占比
stop_rate = mean(R <= -SL+eps)  # 触发止损占比
```

**为什么需要这两个?**
- worst_10% 在固定 SL 下全部 = -SL（止损封顶），无法区分尾部差异
- loss_rate 和 stop_rate 是真正的区分力指标：
  - **好的 Entry**: snotio 高 + stop_rate 明显低于 baseline
  - **伪改进**: snotio 还行 + stop_rate ≈ baseline（只是挑了 easy trade）

**实测数据 (BPC, SL=2.0R):**

| Filter | snotio | Loss% | Stop% | Trades | 判定 |
|--------|--------|-------|-------|--------|------|
| bb | **16.43** | 42.7% | **40.3%** | 330 | 真正好 Entry (snotio↑ + stop↓) |
| wpt | 15.47 | 50.1% | 47.6% | 437 | 部分 execution 友好 |
| full+bb (N=2) | 15.03 | 43.1% | 41.1% | 401 | 优质组合 |
| full+bb+wpt (N=3) | 14.13 | 48.0% | 45.7% | 725 | 当前选择 — 平衡 |
| Baseline | 13.20 | **53.4%** | **50.6%** | 11107 | 对照 |

**结论**: bb 的 stop_rate 比 baseline 低 10.3pp，是唯一同时大幅提升 snotio 和降低 stop_rate 的 filter。

---

## 六 B、正交维度搜索 — 从 bb-only 到 bb OR liquidity_silence (2026-02-10)

### 问题起源

bb 是唯一的 snotio↑+stop↓ filter，但现有 filter 库中其余 15 个都无效。原因已经分析：

1. cvd/wick/wpt 等“吸收型”信号与 pullback_depth 高度共线
2. bb_compression 是唯一正交维度（GARCH 波动率聚类）

那么问题是：**除了波动率压缩，还有哪些正交维度是空白的？**

### 三个新维度假设

| 维度 | 特征 | 计算方式 | 与 bb 的区别 |
|------|------|----------|-------------|
| 波动率动态 | `ef_vol_regime_shift` | `bb_width_normalized_pct.diff(5)` | bb=水平(静态)，这个=方向(动态) |
| 流动性枯竭 | `ef_liquidity_silence` | `vol_percentile_approx` | bb=布林带宽度，这个=成交量绝对百分位 |
| 时间压缩 | `ef_consolidation_bars` | 连续 `was_in_pullback==1` bar 数 | bb 完全没有时间维度 |

实现位置：[compute_derived_entry_features()](scripts/backtest_execution_layer.py) — 运行时从已有列派生，不在特征工程层新增。

### 全量 19 Filter 测试结果

| Rank | Filter | snotio | Stop% | Win% | Trades | 判定 |
|------|--------|--------|-------|------|--------|------|
| 4 | **liquidity_silence** | **18.29** | **42.2%** | 54.5% | 211 | **snotio↑ + stop↓** |
| ★ | bb | 16.43 | 40.3% | 57.3% | 330 | snotio↑ + stop↓ |
| 8 | vol_regime_shift | 13.34 | 45.0% | 52.7% | 442 | snotio≈baseline |
| 10 | consolidation_time | 11.64 | **37.9%** | 55.6% | 169 | **stop最低**，但snotio差 |
| — | baseline | 13.20 | 50.6% | 46.6% | 11107 | 基线 |

**发现：`liquidity_silence` 是继 bb 之后第二个同时 snotio↑ + stop↓ 的 filter。**

### 重叠分析

```
bb total:   330
liq total:  212
bb ∩ liq:    86  (仅 19% 重叠)
bb ONLY:    244
liq ONLY:   126
bb ∪ liq:   456
```

bb 和 liq 的交集只有 86 bar — **它们确实在捕捉不同的市场状态**：
- bb: 波动率已经压缩到极端低位 (GARCH squeeze)
- liq: 成交量在历史底部 20% (参与者枯竭)

### OR 组合测试结果 (帕累托改进)

| Config | snotio | Loss% | Stop% | Win% | Trades |
|--------|--------|-------|-------|------|--------|
| bb only | 16.43 | 42.7% | 40.3% | 57.3% | 330 |
| liq_silence only | 18.29 | 45.5% | 42.2% | 54.5% | 211 |
| **bb OR liq_silence** | **19.18** | **41.3%** | **39.3%** | **58.7%** | **455** |
| bb OR liq OR consol | 17.82 | 42.1% | 38.9% | 57.9% | 563 |
| bb AND liq_silence | 6.45 | 57.0% | 50.0% | 43.0% | 86 |
| baseline | 13.20 | 53.4% | 50.6% | 46.6% | 11107 |

**bb OR liquidity_silence 在所有维度上都优于 bb-only**：

| 指标 | bb only | bb OR liq | 变化 |
|------|---------|-----------|------|
| snotio | 16.43 | **19.18** | **+17%** |
| Loss% | 42.7% | **41.3%** | **-1.4pp** |
| Stop% | 40.3% | **39.3%** | **-1.0pp** |
| Win% | 57.3% | **58.7%** | **+1.4pp** |
| Trades | 330 | **455** | **+38%** |

这是一个「帕累托改进」——没有任何一个维度变差。原因是 liq ONLY 的 126 笔交易补充了 bb 未覆盖的高质量入场机会。

### 为什么不加更多？

- `bb OR liq OR consolidation` → snotio=17.82 (降低)，因为 consolidation 的 snotio < baseline，加入后稀释了质量
- `vol_regime_shift` → snotio=13.34 ≈ baseline，边际贡献为零
- **正确做法：只加同时 snotio↑+stop↓ 且重叠低的 filter**

### 结论

- bb + liquidity_silence 是仅有的两个同时 snotio↑ + stop↓ 的 filter
- bb: GARCH 波动率聚类（低波动→爆发）
- liq: 流动性枯竭（极低成交量→大动作）  
- 两者正交（重叠 19%），OR 组合是帕累托改进
- 当前配置: `bb OR liquidity_silence` (enabled in entry_filters.yaml)

### 为什么其他 filter 无效？——候选池与正交性分析

**误解**：「baseline 11107 bar 太多，导致其他 filter 被稀释」

**事实**：所有 filter 都已要求 `was_in_pullback==1`，实际候选池是 ~1202 个 pullback bar（deep_pullback），而非 11107。

| Filter | 条件 | 实际候选池 |
|--------|------|------------|
| deep_pullback (base) | `was_in_pullback==1` + `depth>=0.55` | 1202 |
| bb | base + `bb_compression>0.72` | 330 |
| wpt | base + `wpt_absorption>0.17` | 437 |
| cvd | base + `cvd_absorption==1` | 650 |

**即使以 deep_pullback 为 baseline，排名不变**：

| Filter | snotio | vs baseline(13.20) | vs deep_pullback(13.16) | stop_rate |
|--------|--------|---------------------|--------------------------|----------|
| bb | 16.43 | +24% | +25% | 40.3% |
| wpt | 15.47 | +17% | +18% | 47.6% |
| cvd | 13.55 | +3% | +3% | 41.2% |

**真正原因：信号正交性**

- cvd_absorption、wick_absorption、wpt_absorption 与 `pullback_depth` + `pullback_quality` 高度共线——回踩越深的 bar 往往也伴随 CVD 吸收、影线吸收，加上去不提供新信息
- `bb_compression`（波动率压缩）是真正正交的维度：回踩深 ≠ 波动率低，波动率低 ≠ 回踩深
- bb 有效的底层因果是 **GARCH 波动率聚类**（低波动→爆发），这是市场物理事实，不依赖 Donchian 检测早晚

### snotio vs N 曲线解读

在 OR 逻辑下（多 filter = 更宽松 = 更多 trades）：
- N 越小 → 越严格 → snotio 越高但 trades 越少
- N 越大 → 越宽松 → snotio 趋近 baseline
- 选择标准：snotio 仍明显高于 baseline 的最大 N（当前 N=3 → +7%）

### 评估工具

```bash
python scripts/optimize_entry_filter_snotio.py \
  --logs results/.../predictions.parquet \
  --strategy bpc

# 输出: entry_filter_snotio_combo.html
# 内容: snotio vs N 图、Loss Rate/Stop Rate 柱状图、Top30 表、Best-per-N
```

---

## 七、什么时候 Entry Filter 才"值得模型化"

**只有一种情况**：

> 你开始怀疑："哪些 entry archetype 在当前 regime 下更该启用？"

那时候学的不是"现在该不该进"，而是"该用哪一套 entry filter"。
这是 **meta-selection**（regime → filter routing），不是 entry 本身。

当前系统还没到这个阶段。触发条件：
- 有 3+ 个 Entry Archetype 各自回测表现与 regime 强相关
- 单一 filter 在不同 regime 下的 Sharpe 方差 > 0.1

---


## TODO（待实现）
3️⃣ 订单簿 / 微观流动性（理论上正交，工程上不值）

如果你问学术意义上，当然还有：

spread collapse

depth imbalance

cancel / trade ratio

但你现在是：

多交易所

中频（非 tick-level）

已经有结构 alpha

工程复杂度 >> 边际收益。

不值。
### 高优先级
- [x] **KPI 迁移到 snotio** — 完成。Entry Filter 主 KPI 改为 snotio + loss_rate/stop_rate
- [x] **filter 组合优化** — 完成。从 9 filter OR → bb only → bb OR liquidity_silence，snotio 从 12.42 提升至 19.18
- [x] **正交维度搜索** — 完成。实现 3 个新维度 (vol_regime_shift/liquidity_silence/consolidation_time)，发现 liq 是第二个 snotio↑+stop↓ 的 filter
- [ ] **Entry Filter 工程规范模板** — 每个 filter 必须满足的最小 trades/day、plateau 最小宽度、禁用条件

### 中优先级
- [ ] **Entry Archetype 扩展** — 当 BPC 以外的策略（SR/Trend）需要 entry filter 时，按 archetype 分类管理

### 低优先级 / 观察
- [ ] **Meta-selection（regime → filter routing）** — 等有 3+ archetype 且 regime 相关性明确时再实现
- [ ] **Entry Pattern Catalog** — 整理所有 Entry Archetype + Sub-Archetype 的触发前置条件、最适 Execution 模板、反条件
