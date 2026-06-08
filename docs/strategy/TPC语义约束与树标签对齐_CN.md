# TPC：语义约束 vs 统计阈值 vs 树标签对齐

> **状态**：研究笔记（2026-06-04），来自 TPC 入场语义实验（S50/S51）与 R&D 方法论讨论。  
> **关联**：[`ABC统一研究框架_CN.md`](ABC统一研究框架_CN.md) · [`方法论_R_and_D流程_CN.md`](方法论_R_and_D流程_CN.md) · [`LAYER_PROMOTION_CRITERIA.md`](../../config/experiments/LAYER_PROMOTION_CRITERIA.md) · 实验 [`20260604_tpc_entry_semantic_validate/`](../../config/experiments/20260604_tpc_entry_semantic_validate/)

---

## 1. 核心结论（30 秒）

| 做法 | 在优化什么 | 强项 | 弱项 |
|------|------------|------|------|
| **语义规则**（depth>0.5、chop gate、near EMA） | 「这类 bar 在故事上该不该做 TPC」 | 可解释、可对照地图诊断 | 慢，假设要人定 |
| **纯统计**（plateau / IC / meta 扫 τ） | 「历史上 P(label\|feature) 怎么分」 | 快，可批量扫 | 易过拟合共变；与机制错位时仍「过关」 |
| **树 + label** | 「在**已定子空间**里哪根 bar 更值」 | 阈值扫描可自动化 | **label 与子空间**若未对齐，自动化的是错误空间 |

**Doctrine 不变**（见 ABC 框架 §0）：① 假设人定 → ② `event_backtest` 验因果 → ③ 监控自动；**禁止**同一趟 run 里发现+调参+promote。

---

## 2. 为什么语义约束往往比「只调统计阈值」更有效？

### 2.1 不是在比「特征更少」

更准确的说法：

- **统计流程**常在**单特征边际**上找 τ：`cvd >= 0.629`、`chop <= 0.4`。
- **语义约束**是**少量特征的组合故事**：深回踩 × 非高 chop ×（可选）近 EMA1200 × 非延续区。

有效的是 **假设空间更小、机制更可解释**，而不是维度更低。

### 2.2 prod 上的反面教材（2025-06 深回踩实验记录）

`config/experiments/20260530_tpc_deep_pullback/DECISION.md` 记载：旧 entry（cvd/recovery 松阈值）下约 **76% 入场 `tpc_pullback_depth ≈ 0`**，语义像 BPC **延续腿** 而非 TPC **回踩腿**。

此时统计上可以仍有过线 bar（`cvd_absorption=1`、`chop<0.4`），但 **机制类状态错了**——SOL 图上的「贴局部极值追」即此类。

### 2.3 与「alpha 不做 beta」的关系

- **账本分工**（A=beta、B=alpha）仍成立，见 [`ABC三层收益结构_战略框架_CN.md`](ABC三层收益结构_战略框架_CN.md)。
- **入场语义**解决的是：在 B 账本里，edge 来自 **swing 段内的 continuation timing**，不是 macro 冲顶后的 naked beta。
- 执行层（宽止损、加仓）不能替代 **「先在对的结构状态开仓」**。

---

## 3. 慢、不自动化：代价与收益

### 3.1 慢在哪里？

手工迭代（看图 → 对 trades 特征 → 写 prefilter/gate/entry → variant-grid）主要花在：

1. **命名机制**（深回踩 vs 贴顶延续）
2. **划界**（与 BPC 的 path_efficiency、chop 分工）
3. **分段验收**（canonical bear/bull/recent，见 `LAYER_PROMOTION_CRITERIA.md`）

这不是「调参慢」，而是 **假设形成慢**。

### 3.2 快在哪里？

假设一旦写成变体（如 S50/S51），应用 **同一套** 自动化：

- `event_backtest --variant-grid`
- `mlbot research scan` / plateau（在**子集**上）
- `calibrate_roll` 月 drift（**不改 yaml**）

### 3.3 为何全自动 pipeline 难收敛到「合适空间」？

| 原因 | 说明 |
|------|------|
| 搜索对象 | meta/plateau 偏 **边际 τ**，不是「depth>0.5 ∧ near EMA」这类 **合取假设** |
| KPI 错位 | success_rate / IC 在 **冲顶 K 线** 上仍可能有信号；目标却是 pullback continuation |
| 极性需人给 | 如 `tpc_pullback_depth` 在语义上应是 **下界**，meta polarity  alone 推不出 |
| 多层叠拟合 | 同一段历史先 tune prefilter→gate→entry，无法归因（WORKFLOW §1 问题 #5） |
| 缺反面教材 | 优化器只见 aggregate R，不见「红框追高」；除非显式 veto（bars_since_local_high、path_efficiency） |

**结论**：自动化适合 **验证** 已定假设；**发现**「该做深回踩」仍要人脑 + 地图。

---

## 4. 树模型 + label 对齐：能否更自动化？

### 4.1 可以，但前提是把语义写进 label 或训练子集

仓库内两条通道（ABC §1）：

| | 规则栈 TPC | 树通道 fast_scalp |
|--|-----------|-------------------|
| ① 输入 | `features_labeled.parquet` | 同左 |
| ② 定案 | **event_backtest** | event_backtest + τ plateau |
| label | `success_label` / failure-first（H≈50） | signed `forward_rr`（H=3） |

树会在全 bar 上学 **P(label \| features)**。若 label 未编码「深回踩后才算 good」，树可能学到 **别的可预测因素**（见 §5）。

**对齐做法示例**：

- 仅在 `tpc_pullback_depth > 0.5` 的子集上训练 / 打标；
- 或 label 定义改为「深回踩后 H 内达到 X R / 非 failure」；
- 再用 `mlbot research ic-prune` / plateau **在子集内**自动化。

### 4.2 「大家都用模型吗？」

分三档（行业常见，非唯一标准）：

1. **纯规则结构策略**（许多 CTA / B 系统）— 机制 + 风控，可解释优先。  
2. **模型做 score，执行仍规则**（本仓库 fast_scalp、gate 树）— 自动化 train+τ，promote 仍看回测。  
3. **端到端 ML**— 自动化高，归因与 regime 漂移更难；与 ABC 分层文化常冲突。

**TPC 相对 fast_scalp**：horizon 长、执行复杂（trailing、加仓、structural exit），**label↔入场语义**对齐成本更高；不是不能做树，而是 **要先收紧语义子空间再让树扫**。

### 4.3 推荐混合路线

```text
人（慢）  语义假设 → 规则变体（S50/S51）+ 可选新 label / 训练子集
    ↓
机（快）  features_labeled 上 scan / ic-prune / 浅树；variant-grid 验因果
    ↓
人（审）  canonical 段 + DECISION.md promote
    ↓
机（快）  calibrate_roll 监控（不改 yaml）
```

---

## 5. 「学到 regime 代理」是什么意思？（详解）

### 5.1 定义

**代理（proxy）**：在训练数据里与 **label** 强相关、能解释大量方差，但 **不是** 你关心的 **因果机制** 的变量（或其组合）。

对 TPC 而言：

- **机制（mechanism）**：趋势已存在 → 出现 **足够深的回踩** → 订单流/缩量确认 → 短期 continuation 有 edge。  
  对应特征故事：`tpc_pullback_depth`、`tpc_vol_pullback_confirm`、`bpc_pullback_delta_absorption`、`bars_since_local_high` 等。

- **Regime 代理**：区分「这两年主要是牛还是熊」「波动大还是小」「价格在 EMA1200 上方还是下方」的 **慢变量 / 粗状态**。  
  对应：`ema_1200_position`、`ema_1200_slope`、`atr_percentile`、`funding_rate`、`tpc_semantic_chop`（粗粒度）等。

**「学到代理」** = 树（或 plateau）在 **全周期 bull+bear 混训** 时，优先用 **regime 代理特征** 做分裂，因为它们 **全局** 对 label 的预测力最强；**机制特征** 只在局部/regime 内才有弱边际增益。

### 5.2 是不是「学到了一个更粗的特征」？

**可以这么理解，但不等于「只用一个粗特征」。**

更精确的说法：

1. **时间尺度更粗**  
   Regime 特征慢变（EMA1200、宏观 band），在 2022–2026 混训里，label 的大头方差来自「牛年做多平均赚、熊年做多平均亏」——树的第一层分裂常像 `ema_1200_position > 0.1` 或 `macd_atr sign`，等价于在学 **环境**。

2. **统计上是混淆（confounding）**  
   浅回踩（depth≈0）在 **bull** 里也可能经常 forward_rr>0（beta 拖着走）；深回踩在 **bear** 里常亏。混训后模型可能学到：  
   - 「牛 + 任意入场」→ good  
   - 「熊 + 任意入场」→ bad  
   而不是「深回踩 → good」。

3. **机制特征变成「条件次要」**  
   `pullback_depth` 在 **固定 regime 子集内** 可能有 lift；在全样本上 gain 低于 `ema_1200_position`。树仍可能用到 depth，但是 **第 3、4 层** 且过拟合某段子样本——回测外推差。

4. **与「更粗」的关系**  
   - **粗** = 状态空间划分少、每个盒子样本多、方差解释率高。  
   - **机制特征** = 细、局部、在非平稳全周期上 **信噪比低**。  
   所以不是树「故意选粗」，而是 **优化目标（全样本 label 预测）奖励粗划分**。

### 5.3 小例子（直觉）

假设 2022–2026 混训，label = failure-first success：

| 年份环境 | 浅回踩 depth≈0 做多 | 深回踩 depth>0.5 做多 |
|----------|----------------------|------------------------|
| Bull 2023–24 | 常赚（beta） | 常赚 |
| Bear 2022 | 常亏 | 也常亏 |
| Recent 震荡 | 假突破多 | 略好但仍难 |

全样本树的第一棵分裂往往是：**「当前是否 bull 代理」**（如 `ema_1200_position > 0.1`），而不是 depth。  
因为在全样本上，「牛/熊」对 label 的 **边际** 解释力 > depth 的 **边际** 解释力。

你要的策略是：**在已判 bull 的环境里，只要深回踩才做**。这是 **条件机制**，必须用 **规则合取** 或 **条件训练子集** 显式写出；混训树不会自动等价于该合取。

### 5.4 与「不想 auto-roll 阈值」同一类问题

| | auto-roll 阈值 | 混训树 / 全样本 plateau |
|--|----------------|-------------------------|
| 在拟合什么 | 一段日历上的 **边际分布** | 同左 + 多特征交互 |
| 风险 | recent_6m 的 τ 锁到 2022 bear | 牛年统计鼓励 **浅回踩追高** |
| 表现 | bear 段失效 | bear/recent 追高、SOL 贴顶单 |
| 语义 | 「一个数走天下」 | 「一个模型走天下」 |

二者都是：**把非平稳环境压成一个全局最优**，环境项盖过机制项。

**正确节奏**（方法论 + WORKFLOW）：

- regime / 宏观：**慢改**，分段看，不按月 auto-roll promote。  
- 机制（depth、entry 吸收）：**假设 → variant-grid → 分段 promote**。  
- 树若要用：**在机制子集或语义 label 上训**，holdout 按 `market_segment.yaml` 验。

### 5.5 如何验证「是不是在学代理」？

| 检查 | 做法 |
|------|------|
| 分裂重要性 | 若 Top-1 永远是 `ema_1200_*` / slope，机制特征靠后 → 代理主导 |
| 分桶 lift | 只在 `bull_2023_2024` 上看 `depth` plateau；与全样本 plateau 方向相反 → 混淆 |
| 与规则对照 | S50 规则砍掉的交易，树 score 仍高 → 树在鼓励追高 |
| 分段回测 | 混训 τ 在 bear_2022 maxDD 恶化 → 全局拟合环境 |

### 5.6 读者速查（对话沉淀）

**一句话**：全周期混训时，模型优先用「牛/熊/波动环境」预测 label，而不是「是否深回踩、是否有吸收」——因为前者在全样本上更好分，但不是你要交易的因果故事。

**「是不是更粗的特征」？**

| 概念 | 含义 |
|------|------|
| **机制特征**（细） | `tpc_pullback_depth`、`vol_pullback_confirm`、`bars_since_local_high` — 这一根该不该现在进 |
| **Regime 代理**（粗） | `ema_1200_position`、slope、`atr_percentile` — 这两年做多平均赚还是亏 |
| **学到代理** | 树第一层常在 **环境** 上切；`depth` 退到深层且易过拟合某段子样本 |

不是树「故意选粗」，而是 **全样本预测 label** 的目标函数奖励 **粗划分**（方差解释高）；机制特征在 **固定 bull 子集** 里才有 lift，混进 bear 年后全局 signal 被淹没。

**SOL / 追高与代理的关系**：bull 段浅回踩（depth≈0）也常赚（beta）→ 混训后模型像在说「牛年就多开」而非「深回踩才开」。你要的是 **条件机制**（bull 环境里仍要 depth>0.5），须用 **规则合取**（S50）或 **条件训练子集**，混训树不会自动等价。

**与 auto-roll 同类**：「一个 τ 走天下」≈「一个模型走天下」——都把非平稳环境压成全局最优，**环境项盖过机制项**。

---

## 6. B 系统有没有必要「全部换成树」？

### 6.1 结论：**没有必要，也不建议**

本仓库的 ABC / R&D doctrine 设计是 **B = 规则栈主线，D = 树通道主线**，不是二选一替换关系（见 [`ABC统一研究框架_CN.md`](ABC统一研究框架_CN.md) §1）。

| 若「全部树化」 | 实际代价 |
|----------------|----------|
| 四层 archetype 合成一个 score | 归因消失：bear 亏因说不清是 regime、gate 还是 entry |
| 执行仍要规则 | trailing、加仓、structural exit、PCM slot **无法**被浅树替代 |
| promote 标准不变 | 仍须 `event_backtest` + canonical 段；树训练好看 ≠ 1m 回放赚钱 |
| 混训默认行为 | 更易 **regime 代理**（§5），除非 label/子集先语义对齐 |
| 运维与 live | `features_labeled` / `predictions.parquet` / 特征 bus 全链路改造，四策略 × 多币 |

**树适合做的**（B 系统内 **局部**）：

- 在 **已定语义子空间** 里精标（例如 depth>0.5 后的 gate/entry τ）
- **方向 / gate** 的候选 score（与现网 `signal_match_position_band` 对照），promote 必须赢过规则基线
- **fast_scalp / short_term_swing**（D 通道）——label 与 horizon 对齐成本低

**规则必须坚持做的**（不宜树替代）：

- **Regime / 宏观带**（EMA1200 死区）— 慢变量，人要分段审
- **机制合取**（深回踩下界、chop deny、path_efficiency 划界）— 这是 hypothesis，不是边际 τ
- **执行与宪法**（止损、加仓、slot、trend_pool_guard）
- **分层 promote**（`LAYER_PROMOTION_CRITERIA.md`）

### 6.2 推荐架构（「混合」而非「替换」）

```text
BPC / TPC / ME / SRB（规则栈，继续为主）
  regime / prefilter / 机制合取  →  人定语义 + yaml（慢）
  gate / entry                  →  规则为主；可选「子集内浅树 τ」补精度（快）
  execution / PCM               →  规则 only

fast_scalp / short_term_swing（树通道，独立 slug）
  全链路树 + plateau + 回测

共享
  features_labeled ① 扫描
  event_backtest ② 定案（唯一 promote 依据）
  calibrate_roll ③ 监控（不改 yaml）
```

### 6.3 什么时候才考虑「某一策略更多树」？

同时满足再加大树比重（例如 TPC entry 浅树），否则维持规则：

1. **规则语义已锁**（如 S50 证明深回踩必要）  
2. **label 或训练子集已与语义对齐**（§4.1）  
3. **树在 canonical 三阶段赢 E0**（Total R↑、maxDD 不恶化）  
4. **不是 regime 代理**（§5.5 检查通过）  
5. **live 特征 bus 与回测一致**（因果特征、无泄漏）

### 6.4 和「大部分人用模型」的关系

行业常见是 **score 模型 + 规则执行**，不是把 prefilter→gate→entry→exit 全换成黑盒。  
你现在的路径（语义规则实验 + 分段 grid）符合 B 系统定位；树应 **减少重复扫 τ 的劳动**，而不是 **取代机制假设**。

### 6.5 TPC 是否「更好做树」？

| 维度 | TPC | fast_scalp（D） |
|------|-----|-----------------|
| label 与入场语义 | H≈50 failure-first，与「深回踩」错位风险高 | H=3 forward_rr，与方向近对齐 |
| 执行 | trailing + 加仓 + EMA1200 structural | timeout / tight SL，chassis 简单 |
| 树化优先级 | **后**：先 S50 类规则 / label 子集 | **先**：已是树通道主线 |

**TPC 更好做的是「语义规则 + 分段验证」**；树是可选加速器，不是前提。

---

## 7. 实验变体索引（20260604）

| ID | 语义 |
|----|------|
| E0_prod | 现网 |
| S50 | `tpc_pullback_depth > 0.5` |
| S51 | S50 + `ema_1200_position >= -0.10` + direction `inner_abs=-0.10` + regime 死区缩窄 |
| E1 | depth >= 0.15 |
| E2 | entry AND `bars_since_local_high >= 0.10` |
| E3 | gate `path_efficiency_pct > 0.15` deny |
| E4 | turbo 20260424 execution |

Grid：`config/experiments/20260604_tpc_entry_semantic_validate/tpc_entry_semantic_grid.yaml`

---

## 8. 后续：若 S50/S51 有效

1. **规则 promote**：按 `LAYER_PROMOTION_CRITERIA.md` 写 prefilter 下界（不必等树）。  
2. **label 实验**：`features_labeled` 仅在 depth>0.5 子集重标 failure-first / 新 continuation label。  
3. **浅树 gate/entry**：在子集内 plateau，promote 仍跑 canonical grid。  
4. **禁止**：全周期混训后直接 `--adopt` 进 prod。

---

## 9. 入场窗口 vs 执行层错配

详见 **[B系统入场语义与执行层周期错配_CN.md](B系统入场语义与执行层周期错配_CN.md)**：`tpc_pullback_depth` 基于 `lookback_breakout=20`（≈1.7d），与 ema1200 执行（≈100d）不对齐；抓「大周期回调」应优先新增 `tpc_macro_pullback_pct` 或拉长 lookback，而非先改 timeframe。

---

## 10. 交叉引用

- 入场漏斗与 SOL 追高诊断：对话记录 → 实验 README。  
- Regime 监控 vs R&D：`docs/strategy/为何不做滚动调阈值_与研究节奏_CN.md`  
- Label 口径：`config/strategies/tpc/labels.yaml` vs `tree_strategies/fast_scalp/labels.yaml`
