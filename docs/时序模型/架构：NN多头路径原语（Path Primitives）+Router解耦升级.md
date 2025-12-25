### NN 多头路径原语（Path Primitives）+ Router 解耦升级（生产级定稿）

本文档总结一次完整的“研究系统 → 生产系统”的 NN 多头升级设计：**模型输出市场未来路径的通用原语（path primitives）**，策略只在 Router 层解释与决策，从而实现**可复用、可监控、可扩展、抗 regime shift**。

---

### 1. 目标与非目标

- **目标**
  - **统一底座**：一个 MLP multi-head 学习未来路径原语，可被 SR 反转 / SR 突破 / 压缩突破 / 趋势等多个 Router 复用。
  - **策略解耦**：模型不拟合具体策略“是否赚钱”，只描述未来路径；策略逻辑（gating / 阈值 / 仓位 / 风控 / 退出）集中在 Router。
  - **可诊断**：除了 PnL 以外，能用 rolling/conditional 指标定位是 head 退化还是 Router 假设失效。
  - **可扩展**：新增策略优先“加 Router”，不需要推翻底座；必要时只增量训练 head 或做轻量校准。

- **非目标**
  - 不在模型里输出 “reversal_head / breakout_head / trend_head” 这类**策略语义 head**（会把 policy 写死，难扩展）。
  - 不直接把 “PnL / Sharpe / win_rate” 当作 head 监督目标（强策略依赖、非平稳、难泛化）。

---

### 2. 代码组织建议（不改 `time_series_model` 大目录）

仓库现状 `src/time_series_model/` 已覆盖训练/回测/实盘/诊断全链路，重命名为 `tree_time_series_model` 的性价比很低（引用与路径连锁改动大）。

推荐按 **model family** 增量扩展：

- **`src/time_series_model/models/nn/`**：放 multi-head MLP、loss、dataset、导出/加载
- **`src/time_series_model/models/tree/`**：逐步把 LightGBM/XGB trainer 归拢（可选）
- Router/策略继续沿用现有 `src/time_series_model/strategies/*` 与 `config/strategies/*`

---

### 2.1 Qlib vs 本架构：研究框架（实验平台） vs 交易决策系统（生产系统）

一句话定性：

> **Qlib 更像“量化实验室/研究平台”（Research Framework）**  
> **本架构更像“交易操作系统/决策系统”（Trading Decision System）**

这意味着：

- Qlib **可以复用你的一部分模块**（数据管理/因子实验/模型训练/IC 指标）
- 但 Qlib **不是为 Router + Execution + RL 风控 + Shadow/Fallback 这类系统形态设计的**

#### 2.1.1 本架构的核心抽象（你现在在做的事情）

```text
Market State
  ↓
Router  (action ∈ {NO_TRADE, MEAN, TREND})
  ↓
Execution Module (可插拔：SR/Breakout/Compression/Trail 等)
  ↓
Risk / Position / Exit
  ↓
PnL / Equity / Drawdown
```

关键点：

- Router：策略原语正交化（Revert/Trend/NoTrade）
- Execution：把 SR/Breakout/Compress 下沉为“怎么执行”的工程模块
- Reward：用资金曲线语言（Sharpe/DD/Tail/Turnover）
- RL：只在 Router/Allocator 层做“目标函数调参”
- Shadow/Fallback：并行验证、防学坏、可回退

#### 2.1.2 Qlib 的默认抽象（它擅长的世界）

```text
Features (factors)
  ↓
Model (Tree/NN)
  ↓
Score / Rank
  ↓
Portfolio Construction
```

它更擅长：

- 数据管理、因子计算、模型训练
- IC / rank IC 等研究指标
- 横截面（cross-sectional）选股/组合构建

它通常不直接覆盖：

- Router（“现在该不该交易/用哪种行为模式”）
- Execution（入场/出场/失败退出/滑点成本的结构化建模）
- RL reward（Sharpe/DD/Tail 等财务目标内生化）
- Shadow/Fallback（上线闸门与回退状态机）

#### 2.1.3 逐层对比（实盘工程视角）

| 维度 | 本架构 | Qlib |
| --- | --- | --- |
| 决策对象 | 行为原语/风险模式（NO/MEAN/TREND） | 单一预测模型的 score/rank |
| 核心问题 | “现在该不该交易、押哪种风险形态、押多少” | “哪些资产更好/预测能力如何” |
| Execution | 一等公民（模块化） | 多为简化/隐式假设 |
| 目标函数 | 财务目标（Sharpe/DD/Tail/Turnover） | 研究目标（IC/Rank IC 等） |
| 上线安全 | Shadow + Fallback FSM | 非默认关注点 |

#### 2.1.4 推荐态度（务实）

把 Qlib 当作：

- ✅ 实验室：数据/因子/模型/研究指标

把本架构当作：

- ✅ 生产系统：Router/Execution/Risk/RL/Shadow/Fallback 的“能上线能回退”闭环


### 3. 总体架构（“市场建模”与“策略决策”分层）

核心分层：

```text
Market Data
  ↓
Feature Pipeline（现有）
  ↓
MLP Backbone（统一表征）
  ↓
Heads（Path Primitives：dir/mfe/mae/t_to_mfe/(persistence)）
  ↓
Routers（SR Reversal / Breakout / Compression / Trend …）
  ↓
Execution & Risk（RR/阈值/仓位/风控/退出）
```

要点：**同一个 head 输出可被多个 Router 用不同方式解释**；当某个策略失效时，可关 Router 或调 Router，而不是立刻重训底座。

---

### 4. Head 设计（最小完备集 + 可选扩展）

#### 4.1 推荐冻结的最小集合（Extended Minimal Set）

- **`dir`（方向置信度）**
  - 训练形态：`dir_logit ∈ R` + `BCEWithLogitsLoss`
  - 推理形态：`dir_score = tanh(dir_logit)`（可选）或 `p_up = sigmoid(dir_logit)`
  - 注意：**不要用 tanh + MSE 回归方向**（梯度易被极端样本主导，难校准）。

- **`mfe_atr`（未来窗口最大有利 excursion / ATR）**
- **`mae_atr`（未来窗口最大不利 excursion / ATR）**
  - 建议监督目标使用 `log1p(mfe_atr)` / `log1p(mae_atr)` 做 Huber（更稳、尾部更不敏感）

- **`t_to_mfe`（到达 MFE 的时间尺度，bars）**
  - 建议监督 `log1p(t_to_mfe)`
  - 注意：`t_to_mfe` 比 `hold_bars` 更接近“反事实原语”（更可复用）

- **`persistence`（可选，方向一致性/持续性）**
  - 作为第五个 head 的边际收益通常较高（尤其对 Breakout/Trend 的 Router）
  - label 可定义为未来窗口内“按方向涨/跌 bar 比例”

#### 4.2 不建议做 head（但可作为 Router 派生量）

- **`p_win` / `win_rate`**：策略依赖、与退出规则绑定；建议作为 Router 派生量或做成多阈值曲线再考虑。
- **`efficiency = mfe/mae`**：是派生量，且在 `mae→0` 时数值不稳；建议 Router 实时计算或用 `mfe_atr + mae_atr` 组合替代。

---

### 5. Label 构造（80H / 80 bars 的“口径一致性”是生命线）

#### 5.1 先钉死两个口径

- **horizon**
  - 80H 不是 80 bars。若 timeframe=4H，则 `horizon_bars = 80H / 4H = 20 bars`。
  - 建议代码中只出现 `horizon_bars`，并在配置里由 `horizon_hours` 和 `bar_hours` 推导。

- **entry 定义**
  - 训练 label 的基准价应与回测/实盘一致。推荐：`entry = open[t + entry_offset]`，常用 `entry_offset=1`。
  - 避免用 `close[t]` 当 entry（会造成训练口径与执行口径错位）。

#### 5.2 路径原语的“交易一致”计算方式

- 使用 **high/low** 扫描 future window，贴合 RR/执行的 intra-bar 假设。
- 使用 **ATR(t)** 做尺度归一，得到跨品种/跨策略可比的无量纲标签。
- 对存在性做 **mask**（例如没有上行 excursion 时不监督 `mfe_atr` 与 `t_to_mfe`）。

建议输出字段（示例）：

- `dir_y ∈ {0,1}`（上行占优=1，否则=0；可加入 neutral band 变 3 类）
- `mfe_atr, mae_atr ≥ 0`
- `t_to_mfe`（0..H，建议监督 `log1p`）
- `mfe_valid ∈ {0,1}`（例如 `max_up > 0`）
- 可选：`mfe_censored`（mfe 出现在窗口末端附近，提示右删失风险）

---

### 6. 训练与稳定性（防止某个 head “拖死” backbone）

#### 6.1 Loss 形态（推荐）

- `dir`: `BCEWithLogitsLoss(dir_logit, dir_y)`
- `mfe_atr`: `Huber(log1p(pred), log1p(true)) * mfe_valid`
- `t_to_mfe`: `Huber(pred, log1p(true)) * mfe_valid`
- `mae_atr`: `Huber(log1p(pred), log1p(true))`（不一定需要 mask）

#### 6.2 Loss 权重调度（实践导向）

一个可用的训练节奏：

- **前期**：更多关注 `dir`（学会基本方向感知）
- **中期**：增加 `mfe/mae`（学习路径幅度与风险）
- **后期**：提高 `t_to_mfe` / `persistence`（学习时间尺度/形态）

注意：权重调度的目标是**防止某个 head 早期 loss 太大导致 backbone 只服务它**。

---

### 7. Router 设计（策略层：gating + score + 仓位/风控/退出）

#### 7.1 Router 的统一接口（建议）

- `gating_mask(df_features) -> bool[]`（结构条件：SR 附近、压缩状态、趋势状态…）
- `score(heads, df_features) -> float[]`（把 path primitives 映射为可排序的 score）
- `position_map(score, risk_proxy) -> size[]`（把 score 映射为仓位与风险约束）

#### 7.2 四类 Router 的典型偏好（经验规则）

- **SR Reversal（均值回归）**
  - 偏好：`mae_atr` 可控、`mfe_atr` 够大、`persistence` 不要太高
  - 方向：可由规则确定（只做多/只做空），模型负责质量与仓位

- **Breakout（突破）**
  - 偏好：`t_to_mfe` 小（更快）、`persistence` 高（更一致）、`mfe_atr` 大
  - 方向：可用 `dir_score` 或规则方向；注意不要把 breakout 写成“hold 越大越好”

- **Compression Breakout（压缩→扩散）**
  - gating：压缩强度满足
  - score：`persistence`、`t_to_mfe`、`mfe_atr` 的组合；“真假突破”用风险/时间尺度过滤

- **Trend（趋势）**
  - 偏好：`persistence` 高、`mae_atr` 可控、`mfe_atr` 大；`t_to_mfe` 不必极小（慢推也可）

---

### 8. 监控与告警（生产系统三层面板 + SOP）

监控层级（强制分层）：

```text
Layer 1：Head（市场感知是否还准）
Layer 2：Router（策略解释是否还对）
Layer 3：Portfolio（资金曲线与集中度）
```

#### 8.1 Head 层（优先看误差/校准/漂移，其次看 IC）

- `dir`：AUC / SignAcc / Brier / ECE（必要时再看 IC）
- `mfe/mae/t_to_mfe`：rolling Spearman（在 Router gating 子集内）+ rolling MAE/RMSE（log1p 空间）
- 分布漂移：KS/PSI（建议对 log1p 后的连续 head 做漂移）
- **样本量下限**：在 conditional 窗口内 `n_samples >= N_min` 才允许触发告警（避免小样本乱跳）

#### 8.2 Router 层（定位“模型坏”还是“解释坏”）

判断准则：

- **Head 稳 + Router 崩**：优先调 Router / 降杠杆 / 关 Router（止血），不是立刻 retrain
- **多 Router 同时崩 + Head 同时漂**：结构性变化候选，进入 retrain SOP

#### 8.3 Retrain vs 调 Router 的 SOP（写死到流程里）

- **只调 Router（高频事件）**
  - Head 指标稳定；某 Router 的 PnL/假设验证指标偏离
  - 动作：阈值/score/仓位映射/风控，必要时关 Router

- **触发 retrain（少数但致命）**
  - 多个 head 同时退化（IC/误差/漂移同时触发）且多 Router 同时失效
  - 动作：暂停新仓（视情况）、触发 retrain pipeline、做数据口径核查

---

### 9. 什么时候从 MLP 升级到 Mamba（非拍脑袋）

触发条件（任一满足才值得考虑）：

- **路径依赖显著且稳定**：在相同静态状态下，结果强依赖最近 T 根因子轨迹（跨月稳定）。
- **低加工信号可用**：引入更低层输入（1m/订单流）带来稳定 lift。
- **残差结构可解释**：MLP head 的误差在“特定时序形态”上系统性偏差，且加轨迹特征能显著修复。

正确升级路径：

```text
MLP（静态 path primitives）
  ↓
MLP + short Mamba（8–16 bars，只影响 1–2 个 head）
  ↓
必要时再扩展
```

---

### 10. 实施路径（建议分三阶段）

- **Phase 1：底座可跑**
  - 固定 head 集合（dir/mfe/mae/t_to_mfe/(persistence)）
  - 固定 label 口径（entry、horizon_bars、ATR 归一、mask）

- **Phase 2：多 Router 复用**
  - SR/Breakout/Compression/Trend 的 gating 与 score/仓位函数落地

- **Phase 3：监控与 SOP**
  - rolling head health + conditional diagnostics + router PnL
  - retrain/调 Router 的自动化触发逻辑（分级告警）

---

### 11. NN 多头路径的多因子方案（横截面 / 资产配置 / TS+CS 融合）

本章把“多头路径原语（dir/mfe/mae/t_to_mfe/…）+ Router 解耦”的范式迁移到 **多资产横截面（Cross-Section, CS）** 与 **组合构建（Allocation）** 场景中。核心结论：

- **Head 与标签构造的范式基本可复用**：依然学习“未来路径的几何属性”，而不是直接拟合收益均值。
- **必须改变的是 Router**：从“交易条件逻辑（if/then）”迁移为“排序/配置逻辑（ranking/allocation）”。
- **在衍生品/杠杆/绝对收益目标下**：CS 不应拥有“强迫交易”的权力；TS（或结构 gating）必须保留 veto（否决权）。

#### 11.1 结构对齐（TS 单资产 → CS 多资产）

时间序列（单资产/逐资产）：

```text
x_t (features)
 → MLP
   → [dir, mfe, mae, t_to_mfe]
     → Router (SR / Trend / Breakout)
       → trade
```

横截面（多资产/同一时刻）：

```text
{x_t^asset_1, ..., x_t^asset_N}
 → shared MLP
   → [dir_i, mfe_i, mae_i, t_to_mfe_i]
 → CS Router (ranking / allocation)
 → portfolio
```

关键点：**shared MLP + heads 不变**；变化集中在 **CS Router（组合构建器）**。

#### 11.2 横截面 Router：从“是否交易”迁移为“排序/配置”

在 TS 交易中常见的 Router 是：

```text
if dir > threshold and mfe > threshold:
  trade
```

横截面 Router 的本质是：

- **score constructor**：把 head 组合成可排序的评分
- **portfolio allocator**：把评分映射为权重（约束、归一化、风险预算）

示例（Top-K 多空/或连续权重）：

- **评分函数（示意）**：
  - `edge = dir_score * clip(mfe_atr / (mae_atr + eps), 0, cap) / (1 + t_to_mfe)`
- **权重映射（示意）**：
  - `w = zscore(edge)` → `clip(w, -w_max, w_max)` → `w /= sum(abs(w))`

注意：横截面中不再强调 “tradable_mask”（是否交易）的二元概念；更常见做法是 **所有资产都参与打分，但权重可能接近 0**。若必须过滤（停牌/流动性/合规），它属于 **规则层数据过滤**，不是模型 head。

#### 11.3 `t_to_mfe` 在 CS 中的含义变化：资金周转效率惩罚项

- TS：`t_to_mfe` 更像“结构兑现的时间尺度”
- CS：`t_to_mfe` 更像“资本占用效率”（同一笔钱能否更快滚动到下一轮）

因此在 CS 评分里常作为惩罚项：`score /= (1 + t_to_mfe)`（或用 `exp(-k * t_to_mfe)`）。

#### 11.4 为什么 path primitives 比 “直接预测收益”更抗 regime shift

Tree-only CS 常直接拟合 `E[r | x]`（收益均值/条件期望），在 regime 切换时容易失效。

path primitives 更像“慢变量”（相对更稳定）：

- 上下 excursion（MFE/MAE）
- 风险不对称
- 时间尺度

这些属性仍会漂移，但通常比“收益均值”更慢、更可监控、更容易被 Router 调参吸收。

#### 11.5 不适用边界

超高频横截面（分钟级、资产数极多）里：

- `t_to_mfe` 与 excursion 的统计不稳定（路径尚未展开）
- 更适合 microstructure/订单流类建模

#### 11.6 TS vs CS vs 融合：三种系统不是一回事

为了避免误用，把三类系统写清楚：

- **TS-driven 多资产交易系统**：每个资产独立出入场（适合杠杆/绝对收益/结构策略）。
- **CS-driven 资产配置系统**：同一时刻排序与调仓（更像 smart beta / 指数增强，通常低杠杆、低频）。
- **TS + CS 融合系统（进阶形态）**：TS 决定“敢不敢打/怎么打”，CS 决定“钱给谁/给多少”。

#### 11.7 控制权原则（强烈建议写进系统规范）

在杠杆/交易型系统中，推荐的控制权分层：

- **TS 拥有 veto（否决权）**：CS 不应强迫某资产交易。
- **CS 负责 allocation**：在 TS（或结构 gating）允许的资产集合内分配风险预算与权重。

推荐融合结构（稳健起点）：

```text
TS Gate (per-asset)
  → tradable_set
CS Allocate (within tradable_set)
  → portfolio weights
```

#### 11.8 PCM（Position & Capital Management）层：CS 只是其中一种实现

严格地说，你需要的是 **资本分配原理（PCM）**，CS ranking 只是其中一种 allocator。基于 path primitives，常见且更“物理”的 PCM 有：

- **Risk Budgeting（风险预算）**：分配的是风险而非名义仓位
  - 例：`position_i ∝ risk_budget_i / ATR_i`，其中 `risk_budget_i` 由 `mfe/mae/t_to_mfe` 组合得到
- **Kelly-like / Expected Utility（带约束）**
  - 例：`w_i ∝ μ_i / σ_i^2`，其中 `μ_i` 可由 `dir*mfe` proxy，`σ_i` 由 `mae` proxy，再做 clip 与风控约束
- **Conviction-weighted（工程最常用的连续仓位）**
  - 例：`conviction = sigmoid(a*dir + b*log(mfe/mae) - c*t_to_mfe)`，`position = base * conviction`

实践建议：先实现一个 **PCM 最小版本**（risk budgeting 或 conviction-weighted），在其上再选择是否需要 CS ranking。

#### 11.9 面板数据与横截面 IC（简要落地口径）

横截面训练/评估数据建议使用 panel 结构：

```text
date | asset | features... | dir_y | mfe_atr | mae_atr | t_to_mfe | (persistence)
```

横截面 IC（更稳，因为每期样本多）：

- 对每个 date 在资产维度计算 Spearman
- 再对 IC 序列做 rolling 平滑

并强调：**Conditional IC（在 CS Router 的约束/过滤条件下）比全样本 IC 更接近实盘使用方式。**

---

### 12. RL 在本架构中的正确位置（RL-ready Router / Allocator）

本章给出强化学习（RL）在当前系统里的**唯一合理位置**与落地路径，避免“端到端 RL”常见的结构性失败。

#### 12.1 定位结论（写死到系统规范）

当前主架构为：

- **TS Signal Engine（多头 NN + path primitives）**：负责市场路径建模（dir/mfe/mae/t_to_mfe/…）
- **PCM（Position & Capital Management）**：负责风险预算与仓位映射（可含 CS allocator）
- **Router（规则 / 树模型）**：负责策略解释与决策（gating/score/启停/权重）

在该架构下：

> **RL 不做价格预测，不碰特征工程，不直接决定 entry/exit。**  
> **RL 的学习对象应当是 Router/Allocator 层的“决策管理”。**

也就是：

```text
[ TS Signal Engine ]  ← 冻结（或低频更新）
[ PCM ]              ← 冻结（或规则化）
[ Router / Allocator ] ← RL 学（最小动作空间）
```

#### 12.2 为什么不是 end-to-end RL（工程视角）

端到端 RL（从价格到动作）在实盘里常失败的核心原因是：

- **状态空间过大**：raw price/tick + 高噪声 + 非平稳
- **动作空间过大**：直接决定方向/仓位/加减仓/止损止盈
- **奖励稀疏且不稳定**：PnL 延迟、强路径依赖、探索成本高

而本系统已经把市场信息压缩为 **path primitives**（低维稳定坐标系），因此 RL 的正确用法是：在小动作空间上做**决策管理**。

#### 12.3 RL 的 state / action / reward（最小可行、可解释）

**State（建议）**：来自 Router 与 head 的聚合状态（低维、可解释）

- path primitives：`dir_score, mfe_atr, mae_atr, t_to_mfe, (persistence)`
- Router 级状态：各 Router 近期胜率/回撤/触发率/IC 健康度（rolling 指标）
- 市场级状态：波动 regime、相关性上升/下降、资金占用（可选）

**Action（建议）**：只允许“策略管理动作”，避免直接下单

- 开关类：启用/禁用某 Router（SR/Breakout/Trend/Compression）
- 倍数类：每个 Router 的 `capital_multiplier`（例如 0.0~1.5）
- 风险预算：上调/下调 `risk_budget` 或 `max_position_cap`
- 保护动作：进入“暂停交易/降频/降杠杆”模式（非常重要）

**Reward（建议）**：用 financial objective 的 proxy，带风险惩罚与稳定约束

- 核心：`portfolio_return`（或 per-step log return）
- 风险惩罚：`- λ * drawdown_increment`、`- μ * volatility`
- 交易成本：`- cost * turnover`
- 稳定性：对频繁开关/频繁调参加入惩罚项（避免抖动）

备注：reward 的目标是“可控地提高长期 risk-adjusted performance”，而不是短期 PnL 最大化。

#### 12.4 落地路线（强烈建议：先规则/树 Router，再 RL）

建议的渐进式路线：

- **Phase A：规则 Router**（先跑通系统，积累日志与监控）
- **Phase B：树/线性 Router（可选）**（学习 score/阈值映射，仍可解释）
- **Phase C：RL Router/Allocator**（只学策略管理动作）

RL 上线前必须具备：

- 可回放的事件/特征/预测日志（offline RL 或仿真训练）
- 清晰的 action 边界（不允许直接碰 entry/exit）
- 完整的监控与告警（避免 RL 在漂移期放大风险）

#### 12.5 什么时候不该上 RL（比什么时候该上更重要）

以下情况先不要上 RL：

- head/label 口径尚不稳定（基础的 path primitives 都在漂移）
- Router 还没固化成可评估的接口（没有明确 action 空间）
- 回测与实盘差距大、成本/滑点模型不可信
- 数据量不足以覆盖多 regime（RL 会“学到当期 regime 的幻觉”）

#### 12.6 一句话原则（可贴墙上）

> **RL = 管理决策，不是预测价格。**  
> **把 RL 放在 Router/Allocator，才有机会稳定 work。**

#### 12.7 Router 设计成“RL-ready，但先不用 RL”（生产级建议）

本节强调一个工程事实：

> 你现在最该做的是把 Router 做成 **RL-ready 的决策管理器（Policy over strategies）**，  
> 但在规则/树 Router 稳定跑起来之前，**不要急着上 RL**。

其核心是：**冻结接口（state/action/reward 的边界）**，先用规则/树实现同接口，积累可回放日志。

##### 12.7.1 Router State 设计（建议 15–25 维，不要更高）

State 的设计原则：

- **state = 已压缩的市场 + 系统状态**
- 不要把高维特征直接塞进 Router（否则 RL/树都会变成“噪声放大器”）

推荐分块：

- **A. 市场状态（来自 TS 多头模型输出）**
  - `dir_score ∈ [-1, +1]`
  - `mfe_atr ∈ R+`
  - `mae_atr ∈ R+`
  - `t_to_mfe ∈ R+`
  - （可选）`persistence ∈ [0,1]`

- **B. 策略适配度（Strategy Fitness，RL 成败关键）**
  - `sr_fitness`
  - `breakout_fitness`
  - `trend_fitness`
  - 形态建议：EMA/rolling window 的 hit-rate / payoff / risk-adjusted proxy（可 clip/normalize）
  - 注意：fitness **不是即时 PnL**，而是“近期稳定性指标”

- **C. 风险与系统状态**
  - `rolling_vol`（realized vol）
  - `dd_ratio`（当前回撤 / 最大容忍回撤）
  - `trade_density`（最近 N bars 交易频率/换手）
  - `leverage_util`（已用风险预算）

- **D. Regime summary（可选，低维）**
  - `trendiness`（ADX / Hurst proxy）
  - `range_ratio`（inside-bar / compression density）

一个示例 state（~12–15 维）：

```text
state = [
  dir, mfe, mae, t_to_mfe,
  sr_fitness, breakout_fitness, trend_fitness,
  rolling_vol, dd_ratio, trade_density, leverage_util,
  trendiness, range_ratio
]
```

##### 12.7.2 Router Action 设计（必须极简）

工程红线：

> 动作空间一旦过大，RL 必死；规则/树也会难以稳定。

推荐动作集（三选一，按稳健程度排序）：

- **方案 A（推荐）策略权重分配（连续）**
  - `action = [w_sr, w_breakout, w_trend]`，每个 `∈ [0,1]`，并归一化（sum=1 或 clip 后再 normalize）

- **方案 B（更保守）策略开关（离散）**
  - `enable_sr / enable_breakout / enable_trend`（可理解为 Router 层 veto/enable）

- **方案 C（最稳）只控风险倍数（连续）**
  - `risk_multiplier ∈ [0.5, 1.5]`
  - RL 只调全局风险，不参与策略选择

重要红线（写死到系统规范）：

- RL **不允许**改：entry/exit、止损、标签、特征
- RL **只允许**改：Router/Allocator 的启停、权重、风险预算

##### 12.7.3 Reward 设计（不要直接用 ΔPnL）

不要用：

```text
reward = ΔPnL
```

推荐 reward = “财务目标 + 风控约束 + 稳定性约束”，示例：

- **基础收益项（风险调整）**
  - `r_base = log(1 + Δequity)` 或 `ΔPnL / rolling_vol`
- **回撤惩罚（必须）**
  - `penalty_dd = -λ * max(0, dd_ratio - dd_limit)^2`
- **稳定性/成本惩罚（重要）**
  - `penalty_turnover = -γ * turnover`（或交易次数/换手）
  - `penalty_cost = -c * cost`
- **多样性/防塌缩约束（可选）**
  - `penalty_collapse = -η * collapse_metric(w_sr, w_breakout, w_trend)`

总 reward：

```text
reward = r_base - penalty_dd - penalty_turnover - penalty_cost - penalty_collapse
```

实践建议：让 PnL 项的权重 ≤ 50%，其余用于“保命与稳定”。

##### 12.7.4 什么时候 RL 不该学（比“该学”更重要）

- 规则/树 Router 还跑不稳（RL 只会放大噪声）
- 决策步样本不足（例如 4H * 3 年只有 ~6500 steps，远低于 RL 所需）
- state/action/reward 未冻结（会学到“当期 regime 的幻觉”）

“该学”的最小前提：

- 规则 Router 至少稳定跑 6–12 个月（或 walk-forward OOS 稳定）
- 已有 replay buffer（可回放的 trajectory 日志）
- action 空间足够小且可解释

##### 12.7.5 防止 RL 过拟合 regime 的工程技巧（可落地清单）

- **Offline RL + Walk-Forward**：不要在线学；按季度/半年 retrain，严格 OOS 评估
- **Regime masking**：训练时随机 mask 某些 state 维度，强迫 policy 不依赖单一信号
- **Conservative policy**：限制 action 变化率（Δaction penalty）、输出加 L2 penalty
- **Ensemble policy**：多个 policy 平均/投票，比单一 policy 稳
- **永久保留 rule fallback**：RL 是增强不是替代，随时可回退

##### 12.7.6 最优落地路线（你现在就该做的）

- Step 1：用规则/树 Router 实现并冻结 state/action 接口（RL-ready）
- Step 2：把 Router trajectory 按统一 schema 记录（offline replay）
- Step 3：离线评估 counterfactual reward（先验证 reward 合理）
- Step 4：上 offline RL（CQL/IQL/TD3+BC），小幅替换 rule weights

---

#### 12.8 样本量：到底需要多少 symbol 才“够步数”（工程下限）

有效样本量的关键不是 bar 数，而是：

```text
effective_steps = (#symbols) × (#decision_points_per_symbol)
```

以 **4H** 为例，3 年每个 symbol 约：

```text
steps_per_symbol ≈ 3 × 365 × 6 ≈ 6500
```

经验下限（能活的量级，不是理想值）：

- Online RL：通常需要 ≥ 1e6 steps（不建议在该系统形态上做）
- Offline RL（无 BC）：~1e5 steps
- **Offline RL + BC（推荐）**：**3e4–5e4 steps**
- Rule→RL 微调：~2e4 steps 也可能 work（但要强约束与严格 OOS）

反推 symbol 数（4H、3 年）：

- RL 从 0 学（不推荐）：`1e5 / 6500 ≈ 15–16` 个 symbol（仍偏勉强）
- **Rule Router → RL 微调（推荐路径）**：`3e4 / 6500 ≈ 5` 个 symbol（可行）
- 建议区间：**8–12 个 symbol**（覆盖不同 micro-regime，减少资产“性格记忆”）

重要认知：

> RL Router 不需要 asset-specific alpha；它学的是“在什么 state 下更该用哪种策略组合”。  
> 因此 symbol 越多通常越利于泛化（前提是 state/action 固定、日志口径一致）。

---

#### 12.9 Rule Router → RL Router 的接口（先用规则跑，未来可替换）

把 Router 抽象成统一接口：输入 state，输出 action（策略权重/开关/风险倍数）。

```python
class Router:
    def act(self, state: dict) -> dict:
        raise NotImplementedError
```

Rule Router（当前使用）：

```python
class RuleRouter(Router):
    def act(self, state):
        # state example: dir/mfe/mae/t_to_mfe + fitness + risk state ...
        dir_ = state["dir"]
        mfe = state["mfe"]
        mae = state["mae"]
        ttm = state["t_to_mfe"]

        w_sr = float(dir_ < -0.5 and mfe / max(mae, 1e-6) > 1.2 and ttm < 20)
        w_trend = float(dir_ > 0.6 and mae < 1.0 and ttm > 40)
        w_breakout = float(abs(dir_) > 0.7 and mfe > 1.3)

        w = {"sr": w_sr, "trend": w_trend, "breakout": w_breakout}
        s = sum(w.values())
        return {k: (v / s if s > 0 else 0.0) for k, v in w.items()}
```

RL Router（未来替换，TS/PCM/Execution 不改）：

```python
class RLRouter(Router):
    def __init__(self, policy):
        self.policy = policy

    def act(self, state):
        obs = self._state_to_tensor(state)
        action = self.policy(obs)
        return self._decode(action)
```

---

#### 12.10 Replay Buffer Schema（离线 RL 成败关键）

推荐 schema（建议照抄，字段尽量齐全，避免后期补日志导致不可复现）：

```python
transition = {
  "state": {
    "dir": float, "mfe": float, "mae": float, "t_to_mfe": float,
    "sr_fitness": float, "trend_fitness": float, "breakout_fitness": float,
    "rolling_vol": float, "dd_ratio": float, "trade_density": float, "leverage_util": float,
  },
  "action": {"w_sr": float, "w_trend": float, "w_breakout": float},
  "reward": float,
  "next_state": dict,
  "done": bool,
  "symbol": str,
  "timestamp": str,
}
```

三个容易漏但非常关键的字段：

- **symbol**：训练时可 randomize/mask，防止学资产 idiosyncrasy
- **timestamp**：walk-forward 切分、避免未来污染
- **done**：可定义为风控 reset/回撤触发等（让 RL 学会“别把账户打死”）

Replay 构建流程：

```text
Rule Router 回测/实盘（行为策略） → 记录 (s,a,r,s') → 合并多 symbol → Offline RL (+BC)
```

---

#### 12.11 RL vs Rule Router 的 A/B 评估流程（必须 Shadow + Walk-Forward）

上线策略：**永远不要直接替换上线**，必须 shadow + A/B。

流程建议：

- Step 1：walk-forward 切分（示例）
  - Train: 2019–2022
  - Test: 2023
- Step 2：冻结 TS + PCM（只换 Router）
  - Router_A = Rule
  - Router_B = RL
- Step 3：对比指标（不只看 PnL）
  - Annual Return / Max DD / Sharpe(or Sortino)
  - Trade Count / Turnover / Cost
  - Strategy usage entropy（是否塌缩到单一策略）
- Step 4：拒绝条件（任一触发直接否决 RL）
  - `DD_RL > DD_Rule × 1.1`
  - `Turnover_RL > Turnover_Rule × 1.5`
  - 策略塌缩（长期只用一个 Router）

成熟团队的真实约束：

> RL 只能降低调参成本、放大规则优势；不能引入新的风险形态。  
> 永久保留 rule fallback，随时可回退。

---

#### 12.12 BC（Behavior Cloning）+ Offline RL：在 TS + Router 架构中的工程落地

本节说明如何把 **BC（行为克隆）+ Offline RL** 放进当前 **TS + Router** 架构里，并给出可直接落地的工程建议。

##### 12.12.1 BC 是什么（一句话）

**Behavior Cloning（BC）= 用监督学习模仿一个已有的好策略**。

在本系统里，“专家（expert）”通常是：

- Rule Router（规则路由）
- Tree Router（树模型路由）
- 人工策略（人工决策记录）

BC 学的是 `(state → action)` 的分类/回归，不是 RL。

##### 12.12.2 为什么 BC 非常适合当前阶段

你当前常见约束（举例）：

- 4H bar
- 3 年数据：每个 symbol 约 `~6500` decision steps
- TS 信号引擎已经较强（path primitives + Router）
- Router action 通常是**低频、离散、可解释**的管理决策

这正是 BC 的甜点区：sample 少、reward 噪声大、探索不可控的问题，BC 可以直接绕过。

##### 12.12.3 BC 在本系统中的准确定位（重要）

BC **不是学交易 alpha**，而是学 **Router 的决策管理逻辑**：

> BC = 学“什么时候启用哪个策略/用多大风险预算/是否进入防守模式”。

##### 12.12.4 Action 设计：离散 action 更适合 BC（示例）

如果 Router 采用离散 action，BC 可直接做 multi-class classification。例如：

```text
action ∈ {
  0: OFF / OBSERVE
  1: TS_FAST
  2: TS_SLOW
  3: REDUCE_RISK
  4: DEFENSIVE
}
```

也可采用更“连续”的 action（如 `w_sr/w_trend/w_breakout`），但 BC 在离散 action 上更稳定、解释性更好。

##### 12.12.5 State 设计：BC/RL 共用同一 state（为平滑升级做准备）

Router state 是 meta 决策层，不建议直接包含 raw price/K 线。示例（概念）：

```text
state = {
  vol_regime, trend_strength, dispersion,
  ts_confidence_fast, ts_confidence_slow,
  ts_recent_pnl, drawdown, leverage_used,
  hour_of_day, days_since_trend_change
}
```

实践要点：

- state 尽量低维（15–25 维）
- state/action/reward 在进入 BC/RL 前应尽量冻结，避免“学到当期 regime 的幻觉”

##### 12.12.6 BC 数据怎么来：Rule Router 的历史决策（行为策略）

BC 的数据来源通常是 Rule Router 回测/实盘的日志：

```text
BC samples = (s_t, a_t, w_t?)
```

- `s_t`：state
- `a_t`：rule router 的 action
- `w_t`（可选）：规则置信度或样本权重（例如“强规则”权重大，“弱规则”权重小）

样本量估算（4H、3 年）：

- `6500 steps/symbol × 10 symbols ≈ 65000`（足够做 BC）

##### 12.12.7 BC 训练细节（不踩坑版）

- **只在 Rule 有明确决策时训练**
  - 避免把“模糊/无交易”强行学成主导类
- **加 entropy regularization（或 label smoothing）**
  - 防止 policy collapse（只输出某一个 action）
- **严格 hold-out：按 time + symbol**
  - 否则会被假泛化欺骗（训练集学到某币“性格”）

loss 示例：

```text
L = CE(policy(s), a) * w
```

##### 12.12.8 BC → Offline RL：为什么更少 steps 也可能 work

关键原因：

> Offline RL 不是从 0 学，而是在 BC policy 附近做保守微调。

推荐组合（概念）：

- BC pretrain：稳定起点
- CQL/IQL/TD3+BC：防止 Q 过估
- KL(policy, BC) 或行为约束：不允许策略“乱飞”

经验量级：

- BC：3e4–5e4 steps
- BC→RL 微调：1e4–2e4 steps
- 纯 RL：不建议在此系统形态下做

##### 12.12.9 什么时候不该用 RL（复述为工程门槛）

以下情况下优先停在 BC 或 rule/tree Router：

- Router 决策频率过高（<1H）且 reward 极噪
- TS 信号引擎尚不稳定（head/labels 频繁改动）
- state/action 尚未冻结
- replay buffer 不足且缺乏可靠的成本/滑点模型

##### 12.12.10 推荐落地路线（可执行）

```text
Rule Router（行为策略） → 记录 replay → BC policy（监督学习） → Offline RL（保守微调） → Shadow/A-B → 小流量上线
```

##### 12.12.11 去神话：RL 的核心价值不是“免调参”，而是“把逻辑设计迁移为目标/约束设计”

常见误解：

> “上了 RL，Router 就自动聪明了，不用再管了。”

这是最容易导致翻车的认知。实盘工程视角下更准确的说法是：

> **RL 的核心好处不是「不用调参数」，而是：你不再需要手工写复杂 if-else 决策逻辑，但你仍然必须设计 state/action/reward 与风险约束。**

RL 主要帮你省掉的是：

- if-else 结构设计（条件组合爆炸）
- 阈值组合方式（不同 regime 下互相打架）
- 改一条规则导致全系统 response 非局部变化（频率/仓位/回撤路径一起变）

RL 并不会帮你省掉的（仍必须工程化/可回测/可监控）：

- state 选什么（维度、尺度、稳定性）
- action 定义（离散/连续，是否允许变化率）
- reward 怎么算（财务语言 + 约束项）
- 风控约束（drawdown/cost/turnover/熔断/回退）
- 是否允许探索（多数情况下仅 offline + 保守微调）

##### 12.12.12 更工程化的一句话：RL 在优化“最终财务目标”，规则/树在优化“中间代理目标”

严格来说，Rule/Tree 与 RL 的关键差异不在“调不调参”，而在于你在优化哪一层目标：

> **RL 是直接对「财务目标函数」做优化；规则/树模型通常是在对「中间代理目标」做优化。**

用表格钉死差异：

| 维度 | Rule / Tree Router | RL Router |
| --- | --- | --- |
| 优化对象 | 中间信号（置信度、阈值、排序） | 最终财务结果（收益/回撤/成本/稳定性） |
| 调参对象 | 条件/阈值/分支 | reward 结构与约束权衡 |
| 决策依据 | 局部最优/经验 | 多步长期回报 |
| 反馈信号 | 单步或短期 | 延迟、多步 |
| 回撤感知 | 只能硬编码 | 可自然进入 reward（如 drawdown 惩罚） |
| 路径依赖 | 弱（补丁式） | 强（通过 reward + state 记忆体现） |

重要澄清（防误区）：

- **RL ≠ 不需要中间模型**。在本系统里，TS 多头路径原语模型必须存在；RL 只应做 TS 之上的 meta 决策（Router/Allocator），否则 RL 直接面对市场噪音基本必死。
- **RL 不是“零参数”**：它只是把参数空间从“阈值/分支”迁移到“财务偏好权衡”（你愿意用多少回撤换多少收益、用多少成本换稳定性）。

##### 12.12.13 机构级 Router Reward 模板：把 Rule 的“隐含动机”显式写成财务语言

Rule Router 的本质动机通常是：长期收益、回撤控制、尾部风险、稳定性、成本控制、路径质量（mfe/mae/ttm）。RL 的正确做法不是“替代这些目标”，而是把它们显式写入 reward。

推荐的 reward 结构（概念）：

```text
reward_t =
  + w_pnl   * pnl_t
  - w_dd    * drawdown_penalty_t
  - w_tail  * tail_risk_t
  - w_turn  * turnover_t
  - w_switch* switch_cost_t
  + w_cons  * signal_consistency_t
```

各项含义与落地建议：

- `pnl_t`：建议做风险调整（例如除以 rolling vol），避免 RL 只学“加杠杆更爽”
- `drawdown_penalty_t`：只惩罚回撤恶化（例如 `max(0, dd_ratio - limit)^2`），把“生存约束”直接写入目标
- `tail_risk_t`：可用 `position * predicted_mae` 或 `position * (mae/(mfe+eps))` 把尾部风险“租金化”
- `switch_cost_t`：对 `action_t != action_{t-1}` 计成本，决定 RL 是否抽风（没有这项通常会乱切）
- `turnover_t/cost_t`：鼓励低摩擦、低换手（必须与回测成本模型一致）
- `signal_consistency_t`：鼓励决策在结构未走完时不要左右横跳（可用 `dir_t * dir_{t-1}` 等简化项）

备注：本仓库已有 `src/time_series_model/rl/reward.py` 的工程实现，可对齐上述结构（含 drawdown/cost/turnover/动作变化率/策略塌缩等可选项）。

##### 12.12.14 Reward 权重（λ）不是“调参”，而是“风险偏好声明”：推荐初始化区间

不同资金规模/风险容忍度对应不同权重区间。下表给的是 **“第一次上线（或 shadow A/B）”** 的可用初始化建议（请配合 walk-forward 做微调）：

| 场景（单策略账户规模） | λ_dd | λ_tail | λ_switch | λ_turnover |
| --- | --- | --- | --- | --- |
| 小资金/试运行（≤ 50 万等值） | 0.3–0.6 | 0.2–0.4 | 0.05–0.1 | 0.02–0.05 |
| 中等规模（50万–500万） | 0.8–1.2 | 0.6–1.0 | 0.15–0.3 | 0.05–0.1 |
| 大资金/对外资金（≥ 500万） | 1.5–3.0 | 1.0–2.0 | 0.3–0.6 | 0.1–0.3 |

实操建议（首次上线更稳）：

- `λ_dd/λ_tail` 取区间中值
- `λ_switch` 取偏高（先稳住行为，再谈收益）
- `λ_turnover` 不要过高（否则 RL 容易学成“不交易”）

##### 12.12.15 Observation（state）控制：Router 只看“慢变量 + 结构变量”，避免 state explosion

经验原则：

- 在你这种 4H、样本步数有限的系统里，**obs_dim 推荐控制在 ~10–14**；超过 20 往往显著增加过拟合与不稳定风险
- Router 是 meta 决策层：尽量不直接喂 raw price/K 线（保持“信息压缩 + 可解释”）

推荐保留（信息密度最高）：

```text
dir, mfe, mae, t_to_mfe,
position/gross_exposure,
drawdown(or dd_ratio), rolling_vol,
prev_action, action_hold_time
```

可删/可合并（初期高冗余/高风险）：

- `equity_norm`（与 drawdown 强共线，Router 更需要“离死亡多近”而不是“赚了多少钱”）
- `mfe_std/mae_std`（样本少时容易导致“逃避交易”；可在 policy 稳定后再加）
- 过细的执行/保证金字段（Router 不该管 execution 级细节）

##### 12.12.16 RL vs Rule 的 A/B 与上线判断：优先看“回撤形态与稳定性”，不是只看 PnL

在 `12.11` 已给出 shadow + walk-forward 的流程与拒绝条件，这里补充“更贴近 PM 的上线判断口径”：

- RL 必须不引入新的风险形态：**回撤更深/尾部更厚/换手更高**任一触发都应否决或回退
- RL 的真实价值常在“节奏管理”：降低抽风切换、缩短 DD duration、改善 `PnL/DD`，而不是把胜率提高 10%

一个可落地的上线（或继续迭代）条件示例：

```text
DD ↓ ≥ 15%
PnL/DD ↑ ≥ 20%
Switch 次数 ↓ ≥ 30%
Tail loss 不变或下降
```

##### 12.12.17 Router 专用监控面板（最小可用清单）

目标：一眼判断 RL 是否学歪、是否需要回退到 Rule Router。

- 行为健康度：
  - action usage（各 action 使用比例）
  - switch / month（切换频率）
  - avg hold time（每个 action 平均持续时间）
- 风险形态：
  - max DD、DD duration、DD recovery
  - worst 5 trades / 95% quantile loss（尾部）
- Head → Action 因果一致性：
  - 例如 Trend action 下应满足：`E[dir]` 更强、`E[mae]` 更低、`E[t_to_mfe]` 更长（否则 reward/obs/约束可能写错）
- 账户状态条件化行为：
  - `P(action | drawdown bucket)`（DD 高应更保守，DD 低可更激进）
- Rule vs RL 差异解释：
  - action differs 的占比、这些差异步的 PnL 与 DD 贡献（帮助判断 RL 改的地方“事后是否也认可”）

##### 12.12.18 「RL 学坏了」的自动回退机制（回 Rule）：RL 永远不是“主权”，而是“可撤销的代理”

核心原则：

> **RL 只能在“表现被证明更好”时才被允许接管。**  
> **回退逻辑本身不能由 RL 决定**（否则 RL 会在要被关前疯狂 `NO_TRADE` 来“保命”）。

###### 12.12.18.1 三层防线（强烈建议全部做）

**Layer 1：Hard Safety Gate（硬门）**  
任一触发 → 立即回退到 Rule（并进入冷却期）。

- 回撤恶化（示例）：
  - `max_dd_RL > max_dd_Rule × 1.2`
- Tail 风险放大（示例）：
  - `worst_5_trades_RL < worst_5_trades_Rule × 0.8`
- 行为退化/抖动（示例）：
  - `switch_rate_RL > 2 × switch_rate_Rule`

**Layer 2：Behavior Consistency Gate（软门）**  
要求 RL 尊重 head → action 的经济一致性（不是收益指标，是“是否还像个策略”）。

例如 Trend 行为检查（示意）：

```text
E[dir | Trend]_RL < E[dir | Trend]_Rule - δ
OR
E[mae | Trend]_RL > E[mae | Trend]_Rule + δ
```

**Layer 3：Performance Drift Gate（慢变量门）**  
滚动窗口（如 3 个月）上，若：

```text
(PnL/DD)_RL < (PnL/DD)_Rule - margin
```

- `margin` 常取 `10%–15%`
- 用于防止 RL 在新 regime 中“慢慢变差”

###### 12.12.18.2 回退执行策略：Stateful Fallback FSM（生产推荐）

不建议“一触发就永久关 RL”或依赖人肉干预。推荐用状态机（FSM）实现可审计、可复位的回退机制：

```text
STATE = {RULE, RL_CANDIDATE, RL_ACTIVE, RL_SUSPENDED}
```

逻辑（示意）：

```text
if STATE == RL_ACTIVE:
  if hard_gate_triggered:
    STATE = RL_SUSPENDED

elif STATE == RL_SUSPENDED:
  use Rule
  if cooldown_passed and metrics_ok:
    STATE = RL_CANDIDATE

elif STATE == RL_CANDIDATE:
  shadow-run RL (no execution)
  if outperform Rule consistently:
    STATE = RL_ACTIVE
```

工程要点：

- `RL_CANDIDATE` 必须是 shadow（不真实执行），避免一次误判造成资金曲线“断裂”
- `cooldown` 建议按时间（如 2–4 周）+ 事件（如回撤恢复/波动回落）组合
- 回退与恢复事件必须写入日志（用于审计与复盘）

##### 12.12.19 Router 是否真的需要「多 action」：action 过细是 RL 失败最常见原因之一

重要认知：

> **action space 的目的不是“表达更多策略”，而是“最小化错误决策的维度”。**  
> 在样本步数有限（4H、几年）且执行器相近的系统里：**action 越少通常越稳**。

###### 12.12.19.1 是否该合并 action：三个硬标准（可执行）

**标准 1：Head 分布是否可分？**  
检验 `P(head | action)` 的可分性（例如 `dir/mae/t_to_mfe/mfe/mae`）：

- 若 `P(head | SR)` 与 `P(head | Breakout)` 高度重叠 → 不值得分 action

**标准 2：action 是否真的改变执行？**  
问一个残酷但很实用的问题：

> SR / Breakout 最后是不是同一套 PCM + 下单逻辑？

如果是：Router 分 action 往往只是引入噪声与不稳定（应下沉到 execution 层）。

**标准 3：RL 是否“偷偷合并”？**  
典型现象：

- RL 很少选 SR  
- 但选 Trend 时 head 特征却像 SR（例如 `mae/mfe`、`t_to_mfe` 分布更接近 SR）

这通常说明 action 切分不正交，RL 会以隐式方式合并（并带来监控困难）。

###### 12.12.19.2 推荐的简化 action 设计（实战路线）

**方案 1（最稳）：2-action（节奏与风险优先）**

```text
action ∈ {0: NO_TRADE, 1: TRADE}
```

- Router 只决定“参与 or 不参与”（节奏管理）
- 具体“怎么 trade（SR/Trend/Breakout）”由 head 连续值 + execution 模块决定

**方案 2（折中、机构常见）：3-action（结构正交）**

```text
action ∈ {0: NO_TRADE, 1: MEAN_REVERT, 2: TREND_FOLLOW}
```

- SR + Compression 可归入 `MEAN_REVERT`
- Breakout + Trend 可归入 `TREND_FOLLOW`
- 这是按“风险形态/市场结构”切分，而不是按“交易技巧/策略名”

###### 12.12.19.3 什么时候一定需要更多 action？

只有在 action 对应 **完全不同的执行器/风控/仓位模型** 时，才值得扩大 action space，例如：

- Trend：金字塔加仓 / trailing
- SR：一次性反转、快速止盈、严格失败退出

否则建议遵循路线：

```text
先用 2-action 上线 → 稳定 ≥ 6 个月 → 再考虑 3-action（MEAN/TREND） → 避免把 SR/Breakout 单独给 RL
```

##### 12.12.20 为什么要“删除策略原语”（SR/Compress/Trend 等）而保留“行为原语”（NO_TRADE/MEAN/TREND）

结论（非常明确）：

> **当 Router 的 action 已升级为正交的金融行为原语（NO_TRADE / MEAN / TREND）时，旧的策略集合（SR/Compress/Trend 的细分）就应该被“遗忘”，而不是继续被模仿或逐条映射到 reward。**

这是一次本质性升级：

```text
旧：SR_revert / SR_breakout / compression / trend / no_trade
新：MEAN / TREND / NO_TRADE
```

###### 12.12.20.1 为什么“逐条映射旧规则到 reward”反而是退化（中级陷阱）

错误做法（不推荐）：

```text
reward += if SR_reversal_success then +1
reward += if compression_breakout_win then +1
```

其本质是让 RL 学“如何更好地执行旧策略”，而不是学“当前市场下什么行为更值钱”。常见后果：

- 把历史偏见固化（把策略名当真理）
- 把 regime shift 当噪音（越变越像旧世界）
- 限制探索空间（永远困在策略集合里）

###### 12.12.20.2 正确抽象：从“形态/策略名” → “行为/风险下注”

你现在要的不是：

```text
是不是 SR？
是不是 compression？
是不是 breakout？
```

而是：

```text
在这种市场状态下，
押“回归风险”（MEAN）还是“扩散风险”（TREND）？
押多少？
还是 NO_TRADE？
```

这是一种更接近“金融决策语言”的抽象：**从 if-else 走向风险下注（risk bet）**。

###### 12.12.20.3 MEAN vs TREND 的本质差异：收益分布形态（正交空间）

`MEAN` 与 `TREND` 之所以适合做 action，是因为它们在收益分布与风险形态上更接近正交：

| 维度 | MEAN | TREND |
| --- | --- | --- |
| 收益分布 | 高胜率、左尾（尾部风险更突出） | 低胜率、右尾（大赢驱动） |
| 风险形态 | 局部可控（但要防趋势期） | 尾部不封顶（但要防震荡期） |
| 持仓时间 | 更短 | 更长 |

因此：Router 的 action 不应该是策略名，而应该是 **风险形态/市场结构**。

###### 12.12.20.4 Router 的 reward 应该锚定什么？

当 action 是行为原语时，reward 不需要知道“是不是 SR”，只需要回答四个问题（财务目标 + 风险偏好）：

- 单位风险收益：risk-adjusted return（如 Sharpe-like）
- 回撤是否可接受：drawdown penalty
- 尾部暴露是否在可控范围：CVaR / tail loss
- 行为是否一致克制：turnover / action stability（防过度抖动与抽风）

只要 reward 回答这四点，Router 会自然学会：

- 什么时候 MEAN 值钱
- 什么时候 TREND 值钱
- 什么时候 NO_TRADE 最好

###### 12.12.20.5 旧 SR/Compress/Breakout 规则的正确位置：Execution 层（工程实现细节）

旧规则并不是消失，而是下沉到执行层：

- `MEAN` execution：可以复用你原来的 SR entry/exit（或更稳的均值回归执行器）
- `TREND` execution：可以复用 compression / breakout / trailing stop 等

它们回答的是：

> **怎么执行这个行为**（execution detail）

而不是：

> **为什么要选择这个行为**（routing decision）

##### 12.12.21 上线 SOP（生产级）：Shadow → Candidate → Active → Fallback（可回退）

目标：让 RL 作为“可撤销代理”上线，而不是把系统主权交给 RL。

**角色分工（必须明确）**：

- **Rule Router**：永远可用、永远可回退（baseline）
- **BC/RL Router**：只在“被证明更好”且符合安全约束时接管
- **回退逻辑**：必须由系统/风控决定，不能由 RL 决定

**推荐状态机（FSM）**：

```text
STATE = {RULE, RL_CANDIDATE, RL_ACTIVE, RL_SUSPENDED}
```

执行要点：

- `RL_CANDIDATE`：只 shadow，不真实执行（并行评估）
- `RL_ACTIVE`：允许真实执行
- `RL_SUSPENDED`：强制回到 Rule（冷却期），冷却结束才可回到 Candidate

**门槛/回退规则（对齐 12.12.18）**：

- Hard gates：回撤恶化 / 尾部更厚 / 过度抖动 → 立即 `SUSPENDED`
- Drift gate：滚动 `PnL/DD` 明显变差 → `SUSPENDED` 或保持 `CANDIDATE`
- Promotion：Candidate 期需连续 N 个评估窗口通过（例如 10 天/10 个窗口）才允许 `ACTIVE`

工程落地建议：

- 强制输出并保存 `shadow_report.html` 与 `counterfactual report.html`
- 每次状态变更必须落日志（state/reason/触发门槛值），便于审计
- 永久保留 rule fallback，且回退应“快速、无人工干预、可复位”

##### 12.12.22 “押多少”（risk bet）由谁负责：Router vs PCM vs Execution 的边界

当 action 已是行为原语（NO_TRADE/MEAN/TREND），“押多少”不应由执行细节决定，而应拆成三层：

- **Router**：决定 *是否参与*（NO_TRADE/MEAN/TREND）以及可选的 *风险模式*（更保守/更激进）
- **PCM（Position & Capital Management）**：把 Router 的 mode 映射为风险预算与仓位上限（exposure cap、风控缩放、DD 防守）
- **Execution**：在给定预算内执行 entry/exit/stop/scale（工程细节），不改变风险偏好本身

这样做的好处：

- action 与 reward 更稳定（不被策略名/执行细节污染）
- 便于做 counterfactual（同一预算下对比不同 mode）
- 风控与回退更简单（可以只收缩 PCM，而不改执行器）

---

### 13. 跨市场复用 Router：是否需要“每个标的一套模型”？（工程决策层）

#### 13.1 结论先行（非常关键）

> **用“市场原语（NO/MEAN/TREND）做 Router”以后：**
>
> - ✅ **不需要**像旧范式那样「每个标的/每类资产」训练一套 Router（不再按 BTC/ETH/SOL 一套套拆）
> - ❌ 但也不是“一个 Router 吃所有市场”
>
> **正确答案是：一个共享 Router（shared backbone）+ 少量（2–4）结构化 specialization（按交易制度/波动结构拆）**

典型拆分维度（按“结构/制度”，不是按“币名”）：

- **HighCap**：BTC/ETH 等主流（相对低噪音、更“趋势可持续”）
- **Alt**：中小币（噪音更强、跳跃更频繁）
- **Meme**：超高波动、jump process 更明显
- **Equity**：A 股/美股（制度与波动结构不同）

> **Router 少模型 ≠ Execution 少模型**  
> Router 尽量共享；Execution/参数/风险预算允许按市场类型微调（这是正确的“差异承载层”）。

#### 13.2 为什么旧世界“必须多模型”，而新世界可以“少 Router”

- **旧范式**：模型在学“市场本身”（features → direction/return）  
  不同市场 = 不同分布 ⇒ 不同模型不可避免。
- **新范式**：Router 在学“如何做决定”（market state → action ∈ {NO, MEAN, TREND}）  
  Router 不关心 BTC/ETH/Meme 的名字，只关心“现在更像哪种结构”。

这三种原语是结构不变量（regime invariance）：

- **NO_TRADE**：噪音/震荡/缩量横盘
- **MEAN**：假突破/回踩/拉高回落
- **TREND**：单边/持续推进/主升浪

#### 13.3 是否要拆 Router？——统计诊断方法（不要凭感觉）

目标不是“看起来不一样”，而是回答：

> **不同 market / asset，在 Router 决策层是否“统计同分布”？**

建议同时看三类证据（A/B/C 必须一起看）：

##### 13.3.1 A：Action 分布一致性（结构一致性）

对每个 market group 统计：

- **P(NO), P(MEAN), P(TREND)**（平均 policy）
- **Jensen–Shannon divergence（JS）**：衡量 action 分布差异

经验阈值（可作为 SOP）：

- **JS < 0.02**：可共享 Router
- **0.02–0.08**：建议加 regime embedding
- **> 0.08**：需要拆 Router（或至少做 specialization head）

##### 13.3.2 B：Action → Reward 映射一致性（金融一致性）

关键不是绝对收益，而是“排序是否一致”：

- 在同一 action 下（NO/MEAN/TREND），不同 market 的 reward 分布是否同向
- 如果出现“排序反转”（例如 BTC: TREND 最好；Meme: MEAN 最好），说明共享 Router 的因果结构不一致 ⇒ **应拆 Router 或将差异下沉到 Execution/PCM**

##### 13.3.3 C：Policy Stability（时间稳定性/漂移）

滚动窗口（例如 1–3 个月）监控：

- action 概率曲线（rolling mean）
- policy entropy（是否坍塌/是否过度抖动）
- 关键门槛：漂移是否集中发生在某一 market group（提示结构 mismatch）

##### 13.3.4 最终拆分决策表（建议写进系统规范）

| 证据 | 结论 |
| --- | --- |
| A/B/C 全部一致 | 一个 Router |
| A 一致，B 不一致 | 差异更可能在 Execution/PCM（不要急着拆 Router） |
| A 不一致，B 一致 | 优先加 regime embedding（结构差异，但因果一致） |
| A/B 都不一致 | 拆 Router（做 2–4 个 specialization） |

#### 13.4 Embedding 怎么加？什么时候加？（按阶段，不要一开始就加）

默认顺序强烈建议：

> Shared Router → 统计诊断 → regime embedding → 最后才拆 Router

三种 embedding（推荐第 2/3 种）：

- **Level 0（不推荐）**：one-hot asset embedding  
  容易学到“币名”，过拟合标的。
- **Level 1（推荐）**：Regime embedding（离散 bucket）  
  用 volatility/liquidity/jump/trendiness 等统计量离散化成 regime_id，再 embedding。
- **Level 2（进阶）**：Soft regime embedding（小 MLP encoder）  
  直接用低维统计量 → MLP → embedding，强调“结构”而非“币名”。

embedding 是否“值得”的验收标准（务实三条）：

- action entropy 是否下降（决策更确定）
- reward variance 是否下降（更稳）
- 拆 Router 的需求是否消失（JS 降、排序一致）

不满足就删掉（embedding 必须可开关、可回滚）。

#### 13.5 一个 Router + 多 Execution：正确的分工（生产级）

核心结论：

> **Router 决定行为（NO/MEAN/TREND）**  
> **Execution 在不同 market_profile 下，把该行为变成可控下注（参数/止损拓扑/入场方式/退出几何）**

推荐工程化结构（概念目录）：

```text
router/
  model.py               # shared router / policy
  diagnostics.py         # JS / reward / drift 报告
  regime_encoder.py      # embedding（可选）
execution/
  base.py                # interface
  mean/                  # MEAN 执行器（可按 market_type 分参数）
  trend/                 # TREND 执行器
pcm/
  allocator.py           # risk budget / exposure cap
monitor/
  router_dashboard.py
  execution_dashboard.py
  shadow_ab.py
```

Router → Execution 接口（关键：Router 不知道 SR/Breakout/Compress）：

- Router 输出：`mode ∈ {NO, MEAN, TREND}` + 可选风险模式（保守/标准）
- Execution 根据 market_profile（流动性/波动/jump）选择实现细节与参数

#### 13.6 Execution 控制论：为什么“经验调参”不是老式规则算法（防打架）

必须澄清一个认知误区：

- **策略规则**：控制市场（IF price/feature → trade）
- **Execution 控制**：控制系统自身（IF execution error → adjust params）

ExecutionController 的职责是校准（calibration），不是优化（optimization）：

- 输入只允许是“条件于已成交”的执行误差量（mfe_realized/pred, mae_realized/pred, slippage/expected, hold vs t_to_mfe）
- 禁止直接用行情结构指标（MACD/RSI/趋势强度）去同时影响 trade set 与 execution 参数（否则双闭环打架）

频率分离原则（贴墙上）：

- **Router：低频（天/周）**——决定是否交易、偏向哪种行为
- **Execution：中频（小时/日）**——只做止盈/止损/仓位/执行形态的慢速校准
- **Order/Venue：高频（秒/分）**——处理滑点/拆单/撮合细节

建议写死的 invariant（宪法级约束）：

- Execution **不得改变 trade set**（不改 entry_time/方向/交易频率）
- 参数更新必须慢于 alpha 半衰期（窗口样本不足时不更新）
- 参数必须有硬边界/回弹力/一键冻结（kill-switch lite）

#### 13.7 Shadow A/B + Chaos Test：让系统“活三年”的工程方法

建议永久保留：

- **Shadow execution**：同一 Router 决策下，A 实盘执行、B/C 影子执行并行记录（不影响实盘）
- **Chaos tests**：主动注入滑点/延迟/流动性塌陷/信号翻转，验证系统不会自激振荡

> 拆 Router 往往是失败信号，不是成功信号。  
> 成功系统更常见的形态是：Router 复用，差异下沉到 Execution/PCM，并通过 shadow/回退机制持续验证。

#### 13.8 附录：Router 拆分诊断伪代码（可直接落地）

本附录给出一套“无玄学”的诊断伪代码，用于决定：

- 是否一个 Router 足够（共享）
- 是否需要加 embedding
- 是否必须拆 Router（做 2–4 个 specialization）

输入假设（最小集合）：

- `df`：包含 columns
  - `group`：market group / asset group（如 HighCap/Alt/Meme/Equity，或你自己的 bucket）
  - `timestamp`
  - `action`：0/1/2（NO/MEAN/TREND）
  - `reward`：Router 层的 step reward（建议用 counterfactual/风险调整 reward；不要直接用裸 PnL）
  - `logit_0, logit_1, logit_2`：Router logits（或你也可直接给 `p0,p1,p2`）

> 说明：如果没有 logits，也可以用 `action` 的频率作为近似（但信息更少）。

---

##### 13.8.1 Action 分布一致性（JS divergence）

```python
import numpy as np

def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def action_prob_from_logits(logits_3):
    # logits_3: [T,3]
    p = softmax(logits_3, axis=-1)
    return p.mean(axis=0)  # [3]

def js_divergence(p, q):
    # Jensen–Shannon divergence (squared) without scipy
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / (p.sum() + 1e-12); q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        a = np.clip(a, 1e-12, 1.0); b = np.clip(b, 1e-12, 1.0)
        return float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def diagnose_action_distribution(df, base_group=None):
    groups = sorted(df["group"].unique().tolist())
    base = base_group or groups[0]
    base_logits = df[df["group"] == base][["logit_0","logit_1","logit_2"]].values
    base_p = action_prob_from_logits(base_logits)

    rows = []
    for g in groups:
        logits = df[df["group"] == g][["logit_0","logit_1","logit_2"]].values
        p = action_prob_from_logits(logits)
        rows.append({
            "group": g,
            "p_no": float(p[0]),
            "p_mean": float(p[1]),
            "p_trend": float(p[2]),
            "js_vs_base": float(js_divergence(base_p, p)),
        })
    return rows
```

推荐阈值（经验，但可作为 SOP）：

- `js < 0.02`：可共享 Router
- `0.02–0.08`：加 regime embedding（优先）
- `> 0.08`：拆 Router 或做 specialization head

---

##### 13.8.2 Action → Reward 映射一致性（排序是否一致）

核心不是 reward 的绝对值，而是：在每个 group 内，**哪一个 action 更赚钱** 的排序是否一致。

```python
import numpy as np

N_ACTIONS = 3  # 0:NO,1:MEAN,2:TREND

def reward_by_action(actions, rewards, min_n=50):
    actions = np.asarray(actions, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    out = {}
    for a in range(N_ACTIONS):
        m = actions == a
        out[a] = float(np.nanmean(rewards[m])) if int(m.sum()) >= int(min_n) else np.nan
    return out

def action_rank(reward_map):
    # reward_map: {0:...,1:...,2:...}
    items = [(a, v) for a, v in reward_map.items() if v is not None and np.isfinite(v)]
    items = sorted(items, key=lambda x: x[1], reverse=True)
    return [a for a, _ in items]  # best -> worst

def diagnose_reward_consistency(df):
    rows = []
    for g, sub in df.groupby("group", sort=False):
        rm = reward_by_action(sub["action"].values, sub["reward"].values)
        rows.append({
            "group": g,
            "reward_no": rm.get(0),
            "reward_mean": rm.get(1),
            "reward_trend": rm.get(2),
            "rank_best_to_worst": action_rank(rm),
        })
    return rows
```

解释：

- 如果不同 group 的 `rank_best_to_worst` 经常“反转”，说明共享 Router 的因果结构不一致  
  ⇒ 更可能需要 **拆 Router** 或把差异下沉到 **Execution/PCM（不同市场同一 action 的下注函数不同）**。

---

##### 13.8.3 Rolling stability（漂移/坍塌/抖动）

建议对每个 group 计算 rolling window 的：

- 平均 action 概率（或 action 频率）
- policy entropy（是否坍塌成单一 action）
- switch rate（是否抖动）

```python
import numpy as np
import pandas as pd

def entropy_from_probs(p):
    p = np.asarray(p, dtype=float)
    p = p / (p.sum() + 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))

def rolling_policy_stats(df, window=200):
    # df columns: group,timestamp,action
    df = df.sort_values(["group","timestamp"]).copy()
    out_rows = []
    for g, sub in df.groupby("group", sort=False):
        a = sub["action"].astype(int)
        # rolling counts
        c0 = (a == 0).astype(int).rolling(window, min_periods=max(10, window//5)).mean()
        c1 = (a == 1).astype(int).rolling(window, min_periods=max(10, window//5)).mean()
        c2 = (a == 2).astype(int).rolling(window, min_periods=max(10, window//5)).mean()
        # entropy
        ent = []
        for i in range(len(sub)):
            p = np.array([c0.iloc[i], c1.iloc[i], c2.iloc[i]], dtype=float)
            if np.any(~np.isfinite(p)) or p.sum() <= 0:
                ent.append(np.nan)
            else:
                ent.append(entropy_from_probs(p))
        ent = pd.Series(ent, index=sub.index)

        # switch rate (simple)
        switches = (a != a.shift(1)).astype(int)
        sw = switches.rolling(window, min_periods=max(10, window//5)).mean()

        out_rows.append(pd.DataFrame({
            "group": g,
            "timestamp": sub["timestamp"].values,
            "p_no": c0.values,
            "p_mean": c1.values,
            "p_trend": c2.values,
            "entropy": ent.values,
            "switch_rate": sw.values,
        }, index=sub.index))
    return pd.concat(out_rows, axis=0)
```

解读要点（经验规则）：

- `entropy → 0` 且 `p_no → 1`：策略变怂/坍塌（可能是 reward 不对或市场环境恶化）
- `switch_rate` 异常升高：抖动（可能是 Router/Execution 指标打架或噪声过大）
- 某个 group 漂移显著，而其他 group 稳定：提示结构 mismatch（优先 embedding/拆分）

---

##### 13.8.4 自动结论生成（决策表模板）

把 A/B/C 结果合并成一张表，输出一个建议结论：

```python
def decide_router_arch(js_value, rank_consistent):
    # js_value: float (e.g. vs base)
    # rank_consistent: bool (reward ranking consistent across groups)
    if js_value < 0.02 and rank_consistent:
        return "ONE_ROUTER_OK"
    if js_value < 0.08:
        return "ADD_REGIME_EMBEDDING"
    return "SPLIT_ROUTER"
```

落地建议：每次 walk-forward / 每周定期跑一次诊断，输出：

- `router_diagnostics.csv`（可审计）
- `router_diagnostics.html`（可读）
- 把结论写入 SOP（是否允许新增 specialization / 是否强制回退共享 Router）


