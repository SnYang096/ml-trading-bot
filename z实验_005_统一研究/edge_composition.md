# 边际优势复合：每层只要一点小优势，最终涌现 Sharpe

> 更新日期：2026-02-20

## 核心认知

**没有任何一层能"每次都对"，也不需要。**

这个系统不是一个超级预测器。它是一个 **条件概率递减管线**：
每一层只做一件小事 — 把失败概率降低一点点。

```
P(亏损) = P(方向错) × P(环境差|方向) × P(质量低|环境) × P(时机坏|质量) × P(执行差|时机)

每个因子 < 1 → 乘积可以很小 → 即使单个因子只有微弱优势
```

---

## 每层的真实优势（基于实际数据）

### Layer 0: Baseline（什么都不做）

全量数据的起点：

| 策略 | 样本量 | Bad Rate (rr < -0.8R) | Median RR |
|---|---|---|---|
| BPC (4H) | 5910 | 48.3% | -0.50 |
| FER (4H) | 6210 | 48.3% | -0.50 |
| ME (1H) | 25674 | 44.8% | +0.01 |

**起点接近抛硬币。** 这是正常的 — 市场大部分时间是噪声。

---

### Layer 1: Direction — "跟对方向"

| 指标 | 说明 | 典型贡献 |
|---|---|---|
| 职责 | 确定 long/short | 确定性规则，不是预测 |
| 优势来源 | sign(momentum/CVD/energy) | 5-15% 方向正确率提升 |
| 失败模式 | 震荡市方向频繁翻转 | 此时 Gate 负责 veto |

**Direction 不需要"猜对方向"。它只需要：**
- 在趋势市中跟对方向（这很容易，sign(momentum) 就够了）
- 在震荡市中不要太害（Gate 会 veto 震荡市的大部分 bar）

**类比**：Direction 像是开车时的方向盘 — 你不需要每秒都精确对准，大方向对了就行。
方向盘偶尔偏一下（方向错），刹车系统（Gate）会保护你。

---

### Layer 2: Gate — "排除必输局"

Gate 不挑赢家，只移除输家。

| 策略 | Gate 规则 | Lift (失败率降低) | PassRate | 效果 |
|---|---|---|---|---|
| BPC | `dir_consistency > 0.55` | 11.4% | 74.3% | 移除方向混乱的 bar |
| BPC | `wpt_ignition > 0.209` | 21.2% | 35.8% | 移除无触发的 bar |
| ME | `me_atr_pct < 0.65` | 18.1% | 29.9% | 移除低波动环境 |
| FER | `trapped_longs < 2.75` | 31.0% | 30.3% | 移除无反转燃料的 bar |

**Gate 的数学效果**：

```
BPC 举例：
  Gate 前:  bad rate = 48.3%
  Gate 后:  bad rate ≈ 48.3% × (1 - 0.18) ≈ 39.6%

  从"几乎抛硬币"变成"有一定优势"
  绝对降低 ~9%，相对降低 18%
```

**Gate 不需要完美。** 它允许漏网（部分坏交易通过），也允许误杀（部分好交易被拒）。
只要 Lift > 0 且 PassRate 合理（不杀太多），Gate 就在做正确的事。

---

### Layer 3: Evidence — "好坏排序"

Evidence 不做 allow/deny，它把通过 Gate 的交易按质量排序。

```
Evidence Score ∈ [0, 1]

Score = w1×Structure + w2×Orderflow + w3×Regime + ...

高分 → 高质量机会 → 更大仓位、更紧止损、更长持仓
低分 → 低质量机会 → 更小仓位、更宽止损、更快退出
```

**Evidence 的数学效果**：

```
假设通过 Gate 后 bad rate = 40%

Evidence 把交易分成 3 档:
  强证据 (score ≥ 0.70): bad rate ≈ 25%, median RR ≈ +1.5R
  中等证据 (score ≥ 0.50): bad rate ≈ 35%, median RR ≈ +0.5R
  弱证据 (score ≥ 0.30): bad rate ≈ 45%, median RR ≈ -0.2R

加权平均比不分档好 → 因为强证据给了更大仓位
```

---

### Layer 4: Entry Filter — "等待确认"

Entry 不改变"做不做"，只改变"什么时候做"。

```
Gate = allow, Evidence = 0.7, 但 Entry Filter 说"等一等"

等待条件: VPIN >= 阈值 / CVD burst / Volume spike

效果: 减少 MAE (Max Adverse Excursion)
     = 入场后的最大浮亏更小
     = 止损被触发的概率更低
```

---

### Layer 5: Execution — "管理已有交易"

```
Execution Tiers (按 Evidence 分档):

  强证据: SL=0.8R, Trail=0.5R, TimeStop=200bars, Size=1.2x
  中等:   SL=1.0R, Trail=0.8R, TimeStop=150bars, Size=1.0x
  弱证据: SL=1.2R, Trail=1.0R, TimeStop=100bars, Size=0.8x
```

**Execution 的效果**：
- 强证据交易 → 更紧止损（减少单笔最大亏损）+ 更大仓位（放大收益）
- 弱证据交易 → 更宽止损（给更多空间）+ 更小仓位（控制风险）

这创造了 **不对称收益**：赢了赚多，输了亏少。

---

### Layer 6: PCM — "只选最好的一个"

当多个 archetype 同时触发时，PCM 仲裁：

```
同一 bar, BTCUSDT:
  BPC: evidence=0.6, AOS=0.62×0.6=0.372
  ME:  evidence=0.7, AOS=0.85×0.7=0.595  ← 选这个

AOS = Archetype Opportunity Score = base_priority × evidence_score
```

PCM 确保同一时刻只有一个最优 archetype 在运行 → 避免冲突和过度暴露。

---

## 复合效果：小优势如何涌现为 Sharpe

### 条件概率链

```
Unconditional Bad Rate:        ~48%  (接近抛硬币)
  → Direction 过滤:            ~45%  (方向减少 3% 坏交易)
    → Gate veto:               ~38%  (Gate 再减少 ~7%)
      → Evidence 强档:         ~25%  (只取高分交易)
        → Entry timing:        ~20%  (时机确认再降 5%)
          → Execution 不对称:  有效 bad rate ~15% (止损更紧+仓位调整)
```

**从 48% 到 15%：没有一层做了什么惊天动地的事。每层只降了几个百分点。**

### Win Rate × RR → Sharpe

```
最终: Win Rate ≈ 75-80%, 平均赢/平均亏 ≈ 1.5-2.0

期望收益 = 0.75 × 1.5R - 0.25 × 1.0R = 0.875R per trade

年化 Sharpe ≈ 期望收益 / 收益标准差 × √(trades_per_year)

即使每笔交易只有 0.3-0.5R 的微弱期望:
  50 笔/年: Sharpe ≈ 0.4 × √50 / σ ≈ 1.0-1.5
  100 笔/年: Sharpe ≈ 0.4 × √100 / σ ≈ 1.5-2.0
```

---

## 关键认知：为什么每层不需要"每次都对"

### 1. 错误被后续层修正

```
Direction 方向错了:
  → Gate 发现"环境不支持"→ veto (很多方向错误发生在不利环境)
  → Evidence 分数低 → 小仓位 (即使入场，亏损有限)
  → Execution 止损 → 截断损失
```

### 2. 各层错误独立

Direction 用 momentum，Gate 用 archetype 特征，Evidence 用结构/订单流。
它们基于不同信息做判断 → 同时都错的概率远低于单独都错的概率。

```
P(Direction错 AND Gate漏 AND Evidence高分 AND Entry确认) << P(Direction错)
```

### 3. 不对称设计放大正确、缩小错误

```
正确时: 跟对方向 + 环境好 + 证据强 → 大仓位 + 紧止损 + 长持仓 → 赚 2-3R
错误时: 证据弱或 Gate veto → 不做 / 小仓位 + 宽止损 + 短持仓 → 亏 0.5-1R

即使正确率只有 55%:
  0.55 × 2.5R - 0.45 × 0.8R = 1.015R per trade (正期望)
```

---

## 类比：德州扑克

一个好的扑克玩家不需要"每手都赢"：

| 系统层 | 扑克类比 | 边际优势 |
|---|---|---|
| Direction | 看到自己的牌 | 知道大致方向 |
| Gate | 弃掉烂牌 | 只玩有基础的牌 |
| Evidence | 读对手强弱 | 知道这手牌的质量 |
| Entry | 等待好位置 | Position play |
| Execution | 控制下注量 | Pot management |
| PCM | 选择上哪张桌 | Table selection |

**一个赢家每手只赢 2-3% 的边际优势，但打 1000 手后稳定盈利。**

---

## 验证框架：如何确认每层都在做正贡献

每层的贡献可以独立验证：

| 层 | 验证方法 | 通过标准 |
|---|---|---|
| Direction | `median(rr_in_dir) > median(rr_long)` | 优于 always-long |
| Gate | `Lift > 0%` (bad rate 降低) | 失败率显著降低 |
| Evidence | 强档 bad rate < 全局 bad rate | 分档有区分力 |
| Entry | 有 Entry vs 无 Entry 的 MAE 对比 | MAE 降低 |
| Execution | 有 trailing vs 无 trailing 的 PnL | PnL 提升 |
| PCM | 多 archetype vs 单 archetype 的 Sharpe | Sharpe 提升 |

**每层只需要"比没有它更好"就够了。不需要完美。**

---

## 总结

```
单层优势: 微小 (5-20% 的条件改善)
复合优势: 显著 (从抛硬币到稳定盈利)
涌现机制: 条件概率递减 × 不对称收益 × 独立错误源
```

这就是为什么：
- Direction 只要"不有害"就够（后面有 Gate 兜底）
- Gate 只要 Lift > 0 就够（后面有 Evidence 排序）
- Evidence 只要能区分好坏就够（后面有 Execution 放大差异）
- 每层都不需要"每次都对"

**这不是一个预测系统，是一个概率过滤系统。**
