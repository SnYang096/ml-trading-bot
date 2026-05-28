# 为何不做滚动调阈值 · 研究节奏与监控分工

> **定位**：把「特征寻找 / 阈值调整 / 稳定性验证」为何要分阶段、为何禁止 rolling optimize、规则栈与树通道差异、以及 **月监控 vs 月调优** 的分工，写成可引用的 doctrine 说明。
>
> 配套阅读：
> - [`ABC统一研究框架_CN.md`](ABC统一研究框架_CN.md) — 三阶段总图
> - [`R&D工具矩阵_CN.md`](R&D工具矩阵_CN.md) — 工具与 pipeline yaml 弃用口径
> - [`方法论_R_and_D流程_CN.md`](方法论_R_and_D流程_CN.md) — 命令与维护节奏表（§3）
> - [`WORKFLOW_整体架构与管线改进计划_CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md) — M1 turbo 纯验证、walk-forward overfit 根因
> - [`B系统不变的层.md`](B系统不变的层.md) — 「策略生命周期 = 数据分布稳定性」
> - [`短期树独立策略_设计与落地_CN.md`](短期树独立策略_设计与落地_CN.md) — 树通道 τ 与重训
> - [`C系统.md`](C系统.md) — chop_grid / trend_scalp 互斥与路由
> - [`A系统.md`](A系统.md) · [`A2_spot_fattail_设计稿_CN.md`](A2_spot_fattail_设计稿_CN.md) — A 层 cycle inventory vs 尾部代理

---

## 0. 一句话结论

**滚动 replay（固定 config 看多个月表现）要保留；滚动 optimize（每窗重搜最优阈值）要禁止。**

日常节奏是：

- **月**：监控 drift（IC / PSI / 固定 yaml replay 趋势）→ 只 alert，**不改 yaml**
- **季（或 drift 触发的人审 R&D）**：在离线 parquet 上找特征、扫 plateau、双段验因果 → **人审 promote**

不是「未来不可预测所以随便找个历史不亏的数」，而是 **对慢变量分布的持续性下注**：锁定 τ 与特征，跨大周期验因果，上线后模拟 **drift 来临时系统的反应**（发现 → 告警 → 减仓/不开仓 → 人审 R&D），而不是每月假装分布又变了去追最优。

---

## 1. 三件事必须分开

| 阶段 | 做什么 | 典型工具 | 改生产 yaml？ |
|------|--------|----------|---------------|
| **① 假设** | 找特征、验方向、扫 plateau 是否存在 | `mlbot research scan` / `ic` / `plateau` | 否 |
| **② 验因果** | 双段 R-multiple、by-side | `event_backtest --variant-grid` | 否（人审 promote 才改） |
| **③ 监控** | drift、缺勤、固定 config replay | `regime_watchdog` / `calibrate_roll`（optimize 全关） | 否 |

**绝对禁止**：同一趟 `mlbot pipeline run`（或任何 bundle）里 ① 找特征 + ② 调阈值 + ③ 评分定上线，并 `--adopt`。

原因不是「工具不能串」，而是 **无法归因**：分不清收益来自新特征、新 τ，还是某段噪声上的 overfit。

---

## 2. 为何滚动 replay 可以，滚动 optimize 不行

### 2.1 直觉 vs 陷阱

Walk-forward **听起来**像 honest OOS：用过去训、在下一段验。  
但若 **每个窗口都重新 grid 搜最优 τ**，实际模拟的是：

> 「一个每月都改规则的研究员」在各段历史上的表现。

Live 上线后的行为是：

> 「规则锁死，时间流过，看 drift。」

两者不是同一对象。** stitched 曲线往往比固定规则 replay 乐观**，这是经典的 walk-forward overfit（见 WORKFLOW §1 问题 #3）。

### 2.2 旧管线错在哪

旧 `calibrate_roll.default.yaml` 曾默认：

```yaml
threshold_calibration:
  prefilter:    { optimize: true }
  gate:         { optimize: true }
  entry_filter: { optimize: true }
```

等于 **月月动 locked 规则旁的数值**，与 B 运维心智（gate/entry **统计做一次**）相反。  
M1 已改为 **turbo 纯验证**：optimize 全关，只产 monthly drift report。

### 2.3 正确的「模拟未来」

| 时机 | 做法 | 问的问题 |
|------|------|----------|
| **选 τ 时（①→②）** | 长窗 plateau + `robustness` + **双段** variant-grid（bull + recent） | 多种历史上是否都过得去？ |
| **上线后（③）** | **固定 config** 的 `rolling_sim` 或 cron `event_backtest` | 若这几个月 τ 一直不变，PnL/IC/触发率是否恶化？ |

③ 里若 drift 超标 → **alert → 人决定是否 reopen 季度 R&D**，不是 cron 自动改 τ。

---

## 3. 核心心智：概率下注，不是逐月预测

### 3.1 我们在赌什么

与其说 **预测未来**，不如说 **假定慢特征/慢因子的数据分布在未来会以一定概率重现**——趋势持续、牛熊结构可区分、入场形态反复出现。我们是对 **分布持续性** 下注，不是对「下一个月最优 τ 是多少」下注。

[`B系统不变的层.md`](B系统不变的层.md) 把这一点写成：

> **策略的生命周期 = 所依赖数据分布的稳定性。**

「重现」不等于逐 bar 复刻历史，而是：

- **Regime 级**（牛/熊/震荡）会持续 **季度～年级**，不会每 30 天换一套；
- **Prefilter 语义**（突破回踩、深回调）在趋势参与者存在时会 **反复出现**；
- 数值 τ 应落在 **plateau 宽高原**，对局部噪声不敏感。

这个 bet **可能错**——所以要有 ③ 监控；错了不 silent retune，而 **显式发现、显式决策**。

### 3.2 每月调阈值：隐含先验就错了

按日历窗 **每月 re-optimize τ**，等价于嵌入先验：

> **数据分布大约每 30 天换一次。**

对 B 规则栈的慢变量而言，这个先验 **与 doctrine 相反**。后果是双重的：

| 问题 | 机制 |
|------|------|
| **过拟合** | 在短窗噪声里搜「当月最优 τ」，stitched 曲线偏乐观，live 锁规则后兑现不了 |
| **欠拟合** | 真正缓慢的结构（regime、archetype 语义）还没显现，就被月频抖动 **洗掉了**；系统永远追局部、学不到 durable 结构 |

因此：不是「预测不了未来所以随便找个数」，而是 **分布变化的时间尺度比 1 个月慢得多**——用月频调参，既 overfit 短噪声，又 underfit 慢结构。

### 3.3 真正的「模拟未来」：模拟 drift 时系统怎么反应

诚实的 forward 问题不是「若我每月改规则，历史拼起来多好看」，而是：

> **分布 drift 了（如 bull → bear），锁定的 config 能否及时发现问题，并做出合理反应？**

期望的反应链（由快到慢）：

```
分布开始漂移（IC↓、PSI↑、触发率异常、fixed-config replay 恶化）
        │
        ▼
③ 监控尽早发现（周 watchdog + 月 fixed-config replay）
        │
        ├─→ Regime OFF / 减频：不适合当前分布的策略 **少开仓、不开仓**（regime 层全局开关）
        ├─→ 账户级：PCM kill / 槽位收缩 → **控回撤**
        ├─→ 架构级：B 趋势 vs C 震荡 **路由到更匹配的 slug**（chop 高 → chop_grid；趋势段 → TPC/BPC）
        └─→ 人审 R&D：alert 触发 ①→② 完整链 → 改 τ / 改特征 / 改 regime / drop 策略
```

要点：

- **③ 测的是「锁死规则后的韧性」**，不是「自适应调参的上限」。
- 「更适合当前分布」主要靠 **regime 路由 + 多 slug 分工 + 不开仓**，不是 cron 悄悄改 gate 里第三个小数。
- 「调整阈值和特征变得适合」属于 **alert 之后的季度 R&D**，必须过 plateau + 双段因果 + decision doc，不能月 cron 自动做。

### 3.4 锁定阈值 + 特征、做大周期测试的意义

Phase ①→② 在 **锁定** 候选 τ 与特征的前提下，用 **大周期、跨 regime** 回测，问的是：

| 问题 | 对应做法 |
|------|----------|
| 我们的「分布会重现」这个 bet，在 bull **和** recent 两段是否都成立？ | variant-grid **双段**（+ 可选第三段） |
| τ 是宽 plateau 中心，还是尖峰噪声？ | `plateau` + `robustness` |
| 锁死后，最坏 regime 下会不会大回撤？ | by-side R-multiple、maxDD、regime OFF 触发率 |

这才是 **大周期测试** 的目的：不是求每一段历史都最优，而是验证 **在多种曾出现过的分布下，固定规则仍可接受；真 drift 了，监控链能接住**。

与 §2.3 的关系：选 τ 时用长窗 + 双段；上线后用 fixed-config rolling replay —— 共同服务 §3.3 的 drift 反应叙事，而非「每月追最优」。

### 3.5 实例：C 系统 — `chop_grid` ↔ `trend_scalp` 路由

C 是 §3.3「架构级路由」的 **最清晰落地**：同一 120T 账户下，**两个 slug 互斥、edge 相反**，用 **锁定的语义代理阈值** 决定「当前分布下开哪种引擎」，而不是每月 re-optimize。

> 详述：[`C系统.md`](C系统.md) · 维护节奏：[`方法论_R_and_D流程_CN.md`](方法论_R_and_D流程_CN.md) §3.3

#### 分工（与 A/B 对照）

```text
A  spot_accum_simple     年级 cycle inventory（ensure 在车上）
B  TPC/BPC/ME/SRB       高置信 trend swing（PCM）
C  chop_grid            震荡 mean-reversion（网格）
C  trend_scalp          低 chop 趋势段顺势加仓（库存腿）
```

**宪法约束**：同一 symbol **同时只能跑** `chop_grid` 或 `trend_scalp` 之一——两边 edge 相反，同时开会自相矛盾（[`C系统.md`](C系统.md) §「C 系统内部：为何二选一」）。

#### 路由逻辑（锁定阈值，非月调）

| 市场状态（代理） | 开谁 | 锁定阈值（示例，live config） | 反应 |
|------------------|------|-------------------------------|------|
| **广义震荡** | `chop_grid` | `bpc_semantic_chop >= 0.50` 开网格；`< 0.32` 关网格 + **强平库存** | 无方向来回吃 spacing |
| **趋势延续、低 chop** | `trend_scalp` | `trend_confidence >= 0.7` 开段；`< 0.4` 段结束；入场 `max_semantic_chop_entry <= 0.25` | 顺势 leg + fee-aware TP |
| **高 chop 里赌趋势** | **都不该做** | trend_scalp 的 chop 上限会 veto | 避免手续费 + 翻转吞噬 |
| **单边突破、网格一侧被套** | chop_grid **被动** | `force_exit_on_regime_loss` | 强平 → 等 chop 回归或换 slug |

这就是 drift 下的 **第一层自动反应**：不是改 τ，而是 **关引擎 / 强平 / 切换到更匹配的 slug**（若运维已为该 symbol 配置了另一 leg）。

#### 分布 drift 场景 walk-through

**场景 A：bull 震荡 → 趋势启动（chop 失效）**

```
bpc_semantic_chop 从 0.55 掉到 0.28
  → chop_grid：exit_chop_below=0.32 触发 → 撤单 + 平库存（控单边堆积回撤）
  → 若 trend_confidence 升到 0.75 且 chop<0.25：运维/配置可切 trend_scalp（互斥切换，非同时开）
  → B 侧 TPC：regime ON 时仍吃高置信 pullback（与 C 账户分离，互补而非重复）
```

**场景 B：趋势段结束 → 回到 chop box**

```
trend_confidence 跌破 0.4 → trend_scalp 段结束、close_offside_all
  → semantic_chop 回到 0.5+ 且 box_pos 在中部 → chop_grid 可重新 arm
  → ③ 月监控：固定 0.50/0.32、0.7/0.4 跑 replay，看 adverse_break / 段级 maxDD 是否恶化
```

**场景 C：代理本身 drift（语义列 IC 掉、plateau 中心偏移）**

```
watchdog：bpc_semantic_chop 的 IC sign-flip 或 PSI>0.25
  → alert，不改 yaml
  → 季度 R&D：condition-set 对照 grid KPI（maker 回合期望 / adverse_break）
       + chop_grid_backtest --variant-grid config/experiments/chop_grid_proxy_<日期>.yaml
  → 人审后才改 entry_chop_min / exit_chop_below 或换 entry_feature
```

#### C 的 R&D 节奏（对照 §3.2）

| 频率 | 做什么 | **不**做什么 |
|------|--------|--------------|
| **月** | 固定 config 多腿 replay；段级 PnL / 费用占比趋势 | ❌ 月 optimize `semantic_chop` 双阈 |
| **半年** | regime 双阈 **复核**（plateau 是否仍宽） | ❌ 无 alert 就改 |
| **季** | 语义代理候选 grid（`bpc_semantic_chop` vs `tpc_semantic_chop` vs `chop_not_box`…） | ❌ rolling 里 embed optimize + adopt |

C 的「更适合当前分布」**首先是 slug 级路由 + regime 强平**，其次才是 **季度** 改语义代理或 τ——与 B 的 gate 小数月调完全不同。

---

### 3.6 实例：A 系统 — cycle inventory，不是 2h 突破 timing

用户常把 A 与「EMA1200-2h 突破入场」联系在一起；仓库里需拆成 **三条线**，避免和 B/C 混参：

| 条线 | 状态 | 入场逻辑 | 与「月调 τ」的关系 |
|------|------|----------|-------------------|
| **A1 `spot_accum_simple`（live）** | 生产 | **周线** `weekly_ema_200_position < 0` → 深熊 DCA deploy；5× 阶梯卖 | **几乎不动**；年级 / 极端事件人脑复盘 |
| **A2 `spot_fattail`（设计稿）** | bad-candidates，**默认不上** | OI surge / funding 极端 / 链上巨鲸等 **尾部代理** | 若启用：**季度** tail-proxy R&D，见 [`A2_spot_fattail_设计稿_CN.md`](A2_spot_fattail_设计稿_CN.md) |
| **ABC 研究：`abc_macro_regime_score` / spot_ft** | 研究口径 | 2h 慢变量（含 `ema_1200_position`、`oi_zscore`、`funding_rate_zscore_50`…）合成 **0–5 分 macro gate**：`score>=3` 允许新 deploy；**不是** precision breakout entry | **年级 gate 复核**；非月 optimize |

> A1 详述：[`A系统.md`](A系统.md) · A2：[`A2_spot_fattail_设计稿_CN.md`](A2_spot_fattail_设计稿_CN.md) · macro score：[`ABC三层收益结构_战略框架_CN.md`](ABC三层收益结构_战略框架_CN.md) §3.2

#### 为何 A **不需要**（也不应）做「EMA1200-2h 突破 + 月调阈值」

**1. Edge 来源不同 — 不是 signal optimization，是 inventory persistence**

A 的目标是 **熊市攒 inventory、牛市持有、周期末退出**（[`A系统.md`](A系统.md)、[`config/strategies/bad-candidates/spot_accum/spot_fattail.md`](../../config/strategies/bad-candidates/spot_accum/spot_fattail.md) 讨论稿）。  
真正决定成败的是 **deploy 曲线、cycle death、长期 exposure**，不是「这根 2h bar 突破 EMA1200 第 3 小数」。

**2. 时间尺度 — 比 B/C 还慢一个量级**

| 系统 | 典型分布周期 | 合理 R&D 频率 |
|------|--------------|---------------|
| B gate/entry | 季度～年级 regime 内 | 季度 |
| C semantic_chop | 季度级 chop/trend 切换 | 半年～季 |
| **A cycle** | **减半/采用率/宏观牛熊：年级～多年** | **年级 / 触发** |

对 A 做 **月频 τ optimize**，隐含先验比 B 还离谱：「宏观 cycle 每 30 天换一次」。

**3. 若把 spot_fattail 做成「EMA1200-2h 突破 + 宽 stop」→ 与 B 重叠，且仍用错数据平面**

[`A2_spot_fattail_设计稿_CN.md`](A2_spot_fattail_设计稿_CN.md) 与 bad-candidates 讨论稿一致：

- breakout + EMA + trailing + 宽 stop ≈ **「不加杠杆的 BPC」**，edge source 与 B **高度相关**，危机时 trend correlation → 1。
- 当前 portfolio 已有 A=cycle beta、B=trend alpha、C=micro alpha；**再加 trend-following spot 层会 strategy overlap**。

因此 **不是「A 不需要调参」**，而是：

- **不该调** 2h 量价突破类 τ（那是 B 的活）；
- **该调**（若做 A2）的是 **尾部事件定义 + 非价量数据上的 proxy**。

**4. 行业/周期分布 — 量价订单流 alone 不够**

A2 设计稿的候选特征池刻意 **离开纯 bar/OHLC**：

- `oi_change_zscore`、`funding_rate_extreme`、`liquidation_cluster_score`
- `whale_inflow_z`、`exchange_netflow_spike`

验收 KPI 也是 **事件后 7d/30d 现货 return、`tail_deploy_success`**，不是 `success_no_rr_extreme`。

含义：

> **Cycle / 尾部 / 行业拥挤** 的分布，在 **衍生品持仓、资金费率、链上 flows** 上才有可重复的统计信号；  
> 仅用 2h K 线 + 订单流做 **月调 τ**，既抓不到 tail 分布，也会在 bull 噪声里 overfit。

`abc_macro_regime_score` 把 OI/funding **纳入 macro gate**，正是承认：**A 层需要多数据平面**，但用途是 **「允不允许加仓」的年级 gate**，不是 B 式 entry timing。

#### A 在 drift 下怎么反应（对照 §3.3）

```
宏观 drift（score 从 4 降到 2；OI/funding 结构变化；周线 EMA  regime 翻转）
        │
        ├─→ score <= 2：**停止新 deploy**（不开新仓试错）— 不是月改 deploy 阈值
        ├─→ 已有 inventory：靠 **cycle death / 慢结构退出 / 5× 阶梯卖** — 不因单 bar 波动清仓
        ├─→ ③ 监控：deploy 节奏、已部署比例、macro score 分布 — alert only
        └─→ 年级 R&D（若做 A2）：tail 事件样本够否、proxy lift 是否仍正 → 人审改 gate 规则
```

与 C 类比：A 的「少开仓」靠 **macro regime score / 周线 EMA 死区**，不是 cron 改第三个 grid 点。

#### A 若未来启用 A2，R&D 仍走同一 doctrine

```bash
# ① 尾部代理 — 数据平面是 OI/链上，不是月 rolling 2h breakout τ
quick_layer_scan condition-set \
  --features-parquet results/spot_fattail/features_labeled.parquet \
  --label tail_deploy_success \
  --condition "oi_z: oi_change_zscore>=2.0"

# ② shadow 回测 + 人审 — 非 mlbot train 主路径
# ③ 月监控 tail 事件率、deploy success 率 — 不 auto-promote
```

**明确不做**（A2 设计稿 §5）：rolling turbo / SHAP 写回 live；用 spot_fattail 复制 BPC 突破逻辑。

---

## 4. 规则栈 vs 树通道（「通道略有不同」详述）

两者 **共用 Phase 1 发现工具**（`features_labeled.parquet` + scan / ic / plateau），**共用 Phase 2 双段回测**，**共用 Phase 3 监控 doctrine（月 drift、不 auto optimize）**。  
差异在 **假设来源** 与 **上线物形态**。

### 4.1 规则栈（BPC / TPC / ME / SRB / C 规则部分）

| 维度 | 规则栈 |
|------|--------|
| **假设表达** | 手写 archetype：`prefilter.yaml` / `gate.yaml` / `entry_filters.yaml` 多条 if/else |
| **阈值数量** | 多：每层多条规则、多个 τ |
| **① 找特征** | `scan` / `condition-set` / `feature-plateau`；prefilter **locked**，季度才审 |
| **②b 调阈值** | 每条规则 offline `plateau` → `calibrate` → draft yaml |
| **上线物** | 可读 yaml 规则 + 数值 τ |
| **漂移形态** | 单特征 IC 翻负、plateau 中心偏移（watchdog 可逐特征告警） |
| **R&D 节奏** | gate / entry / regime：**季度**（drift 可提前触发人审） |

规则栈 **不需要** `predictions.parquet` 做 ①（路线 B：`features_labeled` 即可）。

### 4.2 树通道（fast_scalp / short_term_swing）

| 维度 | 树通道 |
|------|--------|
| **假设表达** | LightGBM 非线性组合 → 单列 `score` |
| **阈值数量** | **少**：通常 **单维 τ**（long/short entry threshold 或 gate deny τ） |
| **① 找特征** | `factor-eval` / `ic-decay` → **冻结** 30–50 列（对齐 horizon H） |
| **训练** | `mlbot train final` → `predictions.parquet` + model artifact |
| **②b 调阈值** | `regime_threshold_calibrate.py`：在 **holdout** 上对 **score 一分位** 扫 plateau，写 `backtest.yaml` |
| **上线物** | **模型权重文件 + 单 τ**（**不**导出 if/else 规则进 gate.yaml） |
| **漂移形态** | score 分布整体平移/缩放；单特征 IC 不够，需 score IC + τ 触发率 |
| **R&D 节奏** | IC 对齐 → 训树 → τ plateau：**季度**；**不是每月** |

### 4.3 树是否「变化快、所以要每月调 τ」？

**否 — 文档口径仍是季度重训 + 季度 τ 标定，不是每月自动调。**

树和规则 **快/慢** 指的是 **失效机制不同**，不是 **维护频率不同**：

| | 规则栈 | 树通道 |
|--|--------|--------|
| **什么在变** | 各特征边际 IC、单条规则 lift | **整片 score 分布**（权重 × 特征联合） |
| **典型失效** | 某 gate 列 IC 翻负 | score 与 label 脱钩、τ 上 pass rate 漂移 |
| **修复方式** | 改一行 yaml 或删一条规则 | **整模重训** + holdout 上 **重扫 τ plateau** |
| **为何仍禁止月调** | 多条 τ 滚动搜 → overfit 叠加 | 每月重训 + 重扫 τ = **更高维** rolling optimize，同样 overfit |

树通道 **看起来** τ 只有一个，但 **模型本身是高维对象**；若每月 `train final` + 每月 τ grid，等价于 **每月换一个完整决策面**，比改一条 gate 规则更难归因、更容易过拟合历史。

正确做法：

1. **季度**（或 watchdog ALERT 触发的人审）：IC 对齐 → 重训 → holdout τ plateau → 双段回测 → promote。
2. **周/月**：固定 model + 固定 τ 跑 watchdog / replay，只看 drift。
3. **禁止**：pipeline 里 rolling 同时 `optimize: true` + `--adopt`。

FAQ 对齐 [`短期树独立策略_设计与落地_CN.md`](短期树独立策略_设计与落地_CN.md) §8：**不要**把树导出成可读规则写回 yaml；promote 的是 **τ 中心 + 模型 artifact**。

### 4.4 树 vs 规则：何时选谁（复习）

| 场景 | 选 |
|------|-----|
| 单条件可解释、要审计 | 规则 |
| ≥3 特征强非线性组合、样本够 | 树 |
| 数据少（<5k bar） | 规则 |
| 要逐特征 drift 告警 | 规则 |

---

## 5. 各层「变化速度」与允许动的频率

> 来源：[`B系统不变的层.md`](B系统不变的层.md)、[`方法论_R_and_D流程_CN.md`](方法论_R_and_D流程_CN.md) §3、WORKFLOW §2.2。

| 层 | 语义 / 变量类型 | 变化速度 | R&D 复核 | 月 cron 做什么 |
|----|-----------------|----------|----------|----------------|
| **Regime** | EMA1200、chop/box — **慢变量、全局开关** | 慢 | **季度**（C 可半年） | 监控 IC/PSI/触发率；**不调** EMA/chop 阈值 |
| **Prefilter** | archetype 语义（突破回踩、深回调…）**locked** | 很慢 | **季度**才审；新候选仅 ① 探索 | 监控；**不** SHAP auto-promote |
| **Direction** | 符号公式，与叙事绑定 | 极慢 | 几乎不动 | 监控 |
| **Gate** | 尾部 deny，经 t-test + plateau 双重验证 | 慢 | **季度**；异常才查 | 监控单特征 IC；**不** optimize |
| **Entry** | OR 择时，少而精 | 慢 | **季度**；异常才查 | 同上 |
| **Execution** | SL/TP/trail，payoff 假设 | 极慢 | 年度或触发 | 监控 |
| **树 score + τ** | 非线性组合 + 单阈 | 模型随分布变；τ 随 score 标度变 | **季度**重训+重标 τ | 固定 artifact+τ replay；**不** 月训月调 |

**Prefilter / Regime 变化慢** → 没有统计证据时不值得每月动；每月动只是在噪声里追最优 → overfit。  
**Gate / Entry** 文档写「异常才查」，比 regime 还保守。

---

## 6. 月监控 drift ≠ 月调优（你的理解）

**是的，就是这个意思。**

| | 月 cron（③） | 季度 / 人触发 R&D（①→②） |
|--|--------------|---------------------------|
| **目的** | 早期发现分布/业绩漂移 | 提出并验证新假设 |
| **改 yaml？** | **否**（alert only） | 人审 promote 才改 |
| **工具** | `regime_watchdog`、`regime_drift_monitor`、`calibrate_roll`（optimize 全关）或固定窗 `event_backtest` | `mlbot research *`、`event_backtest --variant-grid`、decision doc |
| **rolling** | ✅ 固定 config 多月 replay | ❌ 不在 rolling 里 embed optimize |
| **产出** | `report.json`、Telegram ALERT、`decisions/*.md` 草稿 | 候选 yaml、EXPERIMENT_INDEX |

WORKFLOW §6 原则：

> 漂移检测不是「自动修复」，是「早期警报」。任何漂移信号都应 **alert → 人审 → 决定是否 R&D**。

因此：

- **不是**「慢变量也要每月调优防 overfit」——反了，是 **慢变量正因为慢，才不该每月调**。
- **是**「每月只看 drift；真要动，走季度级（或 alert 触发的）完整 R&D 链」。
- 与 §3.3 一致：**月监控 = 测锁定系统在 drift 下能否早发现、少开仓、控回撤**；不是月调优。

---

## 7. 推荐离线链路（规则栈示例）

```bash
# Phase 0
mlbot train final --no-docker --prepare-only -c config/strategies/tpc \
  --output-dir results/train_final/tpc/<run_id>
# → features_labeled.parquet

# Phase 1 — 假设（可多次、可并行，不动 yaml）
mlbot research scan condition-set --strategy tpc --layer prefilter ...
mlbot research ic --strategy tpc --features pulse_z ...
mlbot research plateau --strategy tpc --feature tpc_pullback_depth ...

# Phase 1→2b — 阈值 draft（仍不动 live）
mlbot research calibrate --from-plateau results/.../plateau.json \
  --output config/strategies/tpc/gate_draft.yaml

# Phase 2 — 因果（config_experiments only）
python -m scripts.event_backtest --variant-grid config/experiments/<grid>.yaml

# Phase 3 — 人审 promote + docs/decisions/*.md

# Phase 4 持续 — 监控（固定 yaml）
python scripts/regime_watchdog.py --strategies tpc ...
mlbot pipeline run --all \
  --config config/strategies/tpc/research/calibrate_roll.default.yaml \
  --stage rolling_sim --skip-shap
```

树通道在 Phase 1 后插入 `mlbot train final` + `regime_threshold_calibrate.py`，Phase 2 以后相同。

---

## 8. 反模式速查

| 反模式 | 为何 |
|--------|------|
| `research_roll` + `--adopt` | ①②③ 揉在一起 |
| 月 cron 跑 `optimize_gate_unified` | 月月动 locked τ |
| rolling 窗口内 optimize + 用同窗成绩决策 | walk-forward overfit |
| label scan 通过就改 yaml（跳过 variant-grid） | 无因果证据 |
| SHAP 直接进 `features.yaml` | importance ≠ 因果 |
| 树每月重训 + 每月重扫 τ（无 alert、无决策文档） | 高维 rolling optimize |

---

## 9. 与 ML4T「滚动训练」对照

Qlib / ML4T 式 rolling 适合 **因子池 + 模型预测 + 组合权重** 频繁重估。  
本仓库 B 规则栈的主线是 **locked 语义 + 少量经 plateau 验证的 τ**；C 是 **multi-leg 规则**；树是 **独立 slug、季度整模更新**。

可借鉴的是 **任务编排** 与 **固定 config replay**；不应照搬的是 **每窗重搜全层阈值并 auto-adopt**。

---

## 10. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-27 | 初稿：滚动 optimize vs 固定 config replay；规则栈/树通道差异；月监控 vs 季 R&D 分工 |
| 2026-05-27 | §3 扩充：概率下注 vs 逐月预测；过拟合+欠拟合；drift 反应链；大周期锁定测试意义 |
| 2026-05-27 | §3.5–§3.6：C 系统 chop_grid↔trend_scalp 路由实例；A 层 cycle/尾部与「为何不月调 2h 突破 τ」 |
