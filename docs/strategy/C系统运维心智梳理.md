# C 系统运维心智梳理

C 系统已经相当合理，不需要大改。下面分块说明哪里可以简化、哪里不用动，以及 `trend_scalp` 与 B 系统（trend swing）的分工与实验结论。

---

## 先说结论

**C 系统的架构是对的，设计哲学清晰，和 A/B 的分工也是对的。唯一需要简化的是维护心智，不是系统本身。**

---

## 合理的部分

**chop_grid 和 trend_scalp 的互斥设计是正确的。** 两者 edge source 相反，同一 symbol 同时跑会自相矛盾，强制互斥是唯一合理的做法。

**与 B 系统的 regime 互补是真实的。** B 在震荡市亏损，C 的 chop_grid 在震荡市赚钱，这是真正的 edge source 分散，不是人为构造的。

**fee-aware TP 是必须的。** trend_scalp 的数据显示费用约占 gross 的 35%，不做费用感知止盈会严重侵蚀收益。

---

## 可以简化的地方（仅一处）

**trend_scalp 和 B 系统（TPC）有部分 regime 重叠**，两者都需要趋势，只是执行模型不同。

这会产生一个实际问题：趋势来了，B 在开 TPC 仓位，C 也在开 trend_scalp 腿，两个账户都在同一方向暴露，相关性上升。

简化方法是加一个**软性协调规则**：

同一 symbol，若 B 已有 TPC 仓位在跑，C 的 trend_scalp 降低 `unit_notional` 或暂停新开腿。不需要跨账户复杂通信，可在 trend_scalp 的 prefilter 里查询「B 系统是否有活跃仓位」。

---

## 不需要动的地方

| 项 | 说明 |
| --- | --- |
| **chop_grid 参数** | spacing、max_levels、ATR 倍数，回测已验证，不需再优化 |
| **trend_scalp 加仓** | max_adds 2→3 边际已很小，当前配置合理 |
| **regime 阈值** | semantic_chop 0.50 开 / 0.32 关；trend_confidence 0.7 开 / 0.4 关；慢变量，定期检查、少改 |

C 系统不需要 ML、SHAP、反复找特征。唯一定期工作：确认 semantic_chop 与 trend_confidence 分界仍有效。

---

## C 系统的维护工作（与 A/B 对比）

| 系统 | 维护重点 |
| ---- | -------- |
| A | 人工判断每个周期的卖出倍数目标 |
| B | regime 划分（EMA1200）+ 异常时查 gate / entry filter |
| C | **只看 chop / trend regime 阈值是否仍有效** |

---

## 一句话（架构）

**C 系统不需要简化架构，只需加与 B 的软协调规则，避免趋势段两账户同向过度暴露；其余保持现状。**

---

## trend_scalp 与 trend swing 有区别吗？

**问题**：trend swing 在 2h 级别、持仓较长；trend_scalp 快速止盈、像小蚂蚁啃利润——是否本质不同？

**答**：有区别，且很本质。

### 本质差异

| | TPC（trend swing） | trend_scalp |
| --- | --- | --- |
| 入场 | 等高置信结构（pullback 到位） | regime 确认即开腿 |
| 持仓 | 数天～数周，单笔盈亏比高、次数少 | 数小时，单笔小、次数多、靠累积 |
| 出场 | 结构失效 | 快速 basket 止盈 |

### 比喻

- **TPC**：狙击手——等最佳时机，一枪一个目标。
- **trend_scalp**：割草机——趋势段内持续收割小利润。

### 为何可以共存

时间颗粒度不同：TPC 等 pullback 时，trend_scalp 可能已止盈重开多次；TPC 持仓时，trend_scalp 可再跑数轮。节奏不同，但都需趋势。

### 需注意

软协调仍重要：两者同需趋势，反转时都会亏。trend_scalp 靠快止盈控单次亏损，TPC 靠结构止损；同时在场时须控制**同一 symbol 合计暴露**。

---

## trend_scalp 的身份与账户分工

**背景**：C 层主因是频率与账户隔离；早期对冲开局 + 顺势加仓，后发现在强 regime 下直接开方向单更好。

### 重新定义真实身份

不是泛泛的「趋势跟随」，而是 **regime 内的方向确认 + 加仓机制**。

早期：对冲探测方向 → 顺势加仓、清弱势腿。强 regime 下方向已知，**对冲开局冗余**；当前默认 `initial_legs: TREND`。

**真正 edge**：在正确 regime 里顺势加仓，而非对冲本身。

### 与 TPC 的关系

| 维度 | TPC | trend_scalp |
| ---- | --- | ----------- |
| 入场触发 | 结构确认 | regime 确认 |
| 持仓逻辑 | 等结构失效 | 快速止盈；翻转见下节实验 |
| 对 regime | 中等（prefilter） | 强（直接定方向） |
| 错误代价 | 单次大止损 | 多次小亏累积 |

### 账户分离

风险语义不同：TPC = 结构失效的单次损失；trend_scalp = regime 误判后的多腿累积。分账户便于看清各系统在何种 regime 下盈亏。

### 一句话（身份）

**trend_scalp = regime 驱动的顺势加仓；TPC = 结构驱动的单笔 swing；账户分离、regime 部分重叠、执行语义不同，架构合理。**

---

## 翻转时是否应开反向单？

**问题**：为何不在翻转时直接止损，而要段内 reseed 反向？是否应对照实验？（历史来自对冲框架下的遗留行为。）

### 直接说结论

**段内翻转后立即按新方向 reseed，是历史遗留，不是当前主动设计。**

在「仅开方向单、regime 确认才入场」前提下，翻转时开反向单等于：**regime 仍判为原段，价格短期反向，却在未确认新 regime 时入场**——与原则矛盾。

### 两种做法

| 做法 | 逻辑 | 问题 |
| ---- | ---- | ---- |
| 反转开反向单 | 价格反转 = 新方向 | regime 未确认，可能是噪音 |
| 直接止损 | 翻转平逆势腿，等下一 regime 段再开 | 与入场逻辑对称 |

### 实验（已完成）

对比：**翻转 reseed（旧默认）** vs **`reseed_on_flip: false`（翻转平仓，等下一 regime）** vs **`flip_action=keep`（对冲遗留）**。

脚本：`scripts/experiment_dual_add_flip_reseed.py`  
窗口：2022-01-01 → 2026-03-31，5 币，2h 信号 + 1m 执行回放。  
详见 [trend_scalp 翻转 reseed 实验](trend_scalp_翻转reseed实验.md)，数据见 `trend_scalp_翻转reseed_ablation_summary.csv`。

### 实验结论（2026-05-19，已落地）

| 配置 | return_pct | portfolio_cum_dd | worst_segment |
| ---- | ----------: | ----------------: | ------------: |
| **翻转后平仓，等下一 regime** | **1273** | **-3.9%** | **-2.5%** |
| 翻转后立刻 reseed（旧默认） | 1235 | -9.3% | -6.8% |
| flip_action=keep（对冲遗留） | 1272 | -9.0% | -6.5% |

**结论**：去掉段内反向 reseed 更好——PnL 更高，maxDD 与最差段明显更小。

**已改**：

- `config/strategies/trend_scalp/archetypes/execution.yaml` → `reseed_on_flip: false`
- 策略 slug：`dual_add_trend` → **`trend_scalp`**（宪法 `multi_leg.strategies` 已同步）
