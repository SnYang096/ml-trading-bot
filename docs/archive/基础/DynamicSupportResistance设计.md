# DynamicSupportResistance（动态支撑/阻力）完整设计

下面给出一个端到端、可实现、可解释的 **DynamicSupportResistance** 设计，覆盖你要求的四部分内容：

1. 动态支撑阻力种类
2. 时间依赖类 features（增强信号）
3. 订单流类 features（确认信号）
4. 组合信号（置信率、高盈亏比、可解释命名示例）

我把设计分成：概念、具体特征（含公式/窗口/参数默认值）、组合/打分方法、输出 schema、实现/工程注意事项、示例信号（含解释）。可以直接拿去实现或用于训练/回测。

---

# 1. 动态支撑/阻力种类（类别表）

| 类别                            |                               描述 | 计算要点 / 输入                                      | 典型窗口                       |
| ----------------------------- | -------------------------------: | ---------------------------------------------- | -------------------------- |
| Price-level SR (层级价格支撑阻力)     |     基于历史聚集价格（成交量密集区、POC）形成的水平 SR | VP（成交量剖面）POC、VPOC 集群化、成交量加权价                   | 30m–1w（可多尺度）               |
| Volatility-band SR            | 基于波动性（如 ATR、Bollinger）形成的带状阻力/支撑 | ATR、布林带中轨/上下轨                                  | ATR window 14、BB window 20 |
| Trendline / Structure SR      |  基于趋势线、斐波那契或 ZigZag 顶底连线形成的斜/结构线 | ZigZag 顶点、最小距离过滤                               | 1h–1w                      |
| Momentum-based SR             |            突破/回抽点对应的动量位（例如动量衰减处） | RSI、MACD、Momentum peaks                        | 5m–4h                      |
| Time-decay SR                 |    随时间衰减的短期阻力（例如刚形成的高频 SR 强但会衰减） | SR age、half-life 指数衰减                          | half-life 1–4 bars（取决TF）   |
| Orderflow SR                  |              由大额挂单/吃单或持续被吸收形成的价位 | Depth spikes、iceberg detection、absorbed volume | tick-level / 1s–1m         |
| Multi-timeframe aggregated SR |                  将不同周期 SR 合并（加权） | 不同TF 的 SR、权重策略                                 | 5m, 15m, 1h, 4h 合并         |

---

# 2. 时间依赖类 features（加强信号 — 用于提升置信度 / 减少虚假突破）

> 目标：用时间/周期信息增强信号稳定性与方向判断

| Feature 名称               |                                         含义 & 公式 / 计算 | 解释 / 为什么有用                   | 建议窗口/参数                   |
| ------------------------ | ---------------------------------------------------: | ---------------------------- | ------------------------- |
| SR Age（支撑阻力年龄）           | age = now - created_time；score_age = 1 - exp(-age/τ) | 老的 SR 更可靠（但也可能过时）；用指数或线性衰减建模 | τ = 3 × SR_TF（例如 SR 在1h上） |
| SR Persistence（持久性）      |         persistence = 连续 N 个 bar 中 SR 成功“被确认”的次数 / N | 测量 SR 在过去一段时间被多次触及而未被摧毁      | N = 20 bars（在 SR TF）      |
| Time-of-day weight       |                       w_tod = 高流动/重要时段加权（开盘/收盘/宏观事件） | 市场在特定时间段更有流动性/易形成趋势          | 自定义市场时段（例如 09:00-11:00）   |
| Volatility ramp-up       |               ramp = ATR_now / ATR_hist_mean (短期/长期) | 波动加速前常有压缩—对突破有效              | ATR短=14，ATR长=50           |
| Compression duration     |       comp_dur = 连续低波动 bar 的数量（基于 std 或 tdigest 百分位） | 长时间压缩→更强的后续突破                | threshold: std < p10      |
| Overnight/Weekend memory |                     weightオff = 若 SR 在隔夜/周末形成，则置信度调整 | 不同时段形成的 SR 对日内影响不同           | 二元标记 + weight factor      |
| SR Volume Accumulation   |                     accumulated_volume @ SR（过去 X 时间） | 高成交在 SR 表示更强的“证明”            | X = 2 × SR_TF             |

---

# 3. 订单流类 features（用于确定并确认信号）

> 目标：通过微观流量（Depth，Trade ticks）确认是否是真突破或被吸收

| Feature 名称                            |                                                             计算 / 指标 | 如何解读                      | 建议实现细节                                                                              |
| ------------------------------------- | ------------------------------------------------------------------: | ------------------------- | ----------------------------------------------------------------------------------- |
| Aggressor Imbalance（主动性不平衡）           |                   (BuyAgg - SellAgg) / (BuyAgg + SellAgg) in window | 正值表示买盘主导，突破更可能有效          | window = 30s–5m（依据TF）                                                               |
| Depth Spike / Liquidity Wall          |              depth_spike = sudden increase in resting size at level | 大挂单墙阻止价格推进（阻力）            | 监控 orderbook 增量 Δsize/Δt                                                            |
| Absorption Ratio                      | absorbed = matched_volume_at_level / aggressor_volume_towards_level | 高吸收表示有大资金在消化对手盘，可能隐性买/卖   | 需要 tick-by-tick                                                                     |
| Iceberg / Hidden flow indicator       |     persistent small trades against a wall + unchanged resting size | 表示有人以被动方式吸收               | detect repeated small market trades that remove visible depth but depth replenishes |
| Quote Refresh Rate                    |                                       quote_updates_per_sec near SR | 高频刷新 + 少量变动 → 有主动做市（可能更稳） | 监控 L2 updates                                                                       |
| VWAP drift around SR                  |                                 VWAP_window change & distance to SR | 若 VWAP 向突破方向移动，突破偏向真实     | VWAP windows: 1m, 5m, 15m                                                           |
| Large Trade Tagging                   |                                       tag trades > X*avg_trade_size | 出现大单与 SR 重合时优先级+          | X = 3–5                                                                             |
| Time-to-fill for large resting orders |                                                 time_to_fill(order) | 快被吃掉→突破可能；长期不被吃→阻挡        | 监控 per-level fill time                                                              |
| Queue position momentum               |      change in expected position in order queue for aggressive side | 当队列头部频繁被替换说明主动方在清理队列      | 需要 L2 depth with order ids                                                          |

---

# 4. 组合信号：置信率 + 预估盈亏比 + 可解释命名体系

## 组合思想（概览）

1. **单因子分数化**：把上面所有 feature 映射到 [0,1] 的分数（方向相关：买/卖）。用 logistic / sigmoid 或 min-max 压缩并做上限/下限保护。
2. **分组加权**：将 features 分组（Price-level, Time-dep, Orderflow, Momentum），每组内部做线性或非线性聚合。
3. **置信度评分（Confidence）**：汇总组分数为整体置信度 C ∈ [0,1]。同时给出方向概率 P_up, P_down。
4. **预估风险/收益（B/R estimate）**：用历史相似结构（nearest-neighbor on feature vector）或经验统计估计平均收益和平均回撤，输出预估 Reward:Risk。
5. **可解释标签化**：根据贡献最高的 2–3 个 feature 生成自然语言解释（如 “Silent Accumulation”）。
6. **优先级/警告**：若存在 orderflow 指示被“吸收”但无成交跟进，给出“可能假突破”警告。

## 信号输出 schema（JSON-like）

```json
{
  "timestamp": "2025-10-15T09:12:00Z",
  "symbol": "SOL/USDT",
  "tf": "5m",
  "signal_type": "LONG_BREAKOUT",
  "sr_level": 132.45,
  "sr_type": ["Price-level","Orderflow"],
  "confidence": 0.78,
  "direction_prob": {"up":0.85,"down":0.15},
  "expected_reward_risk": {"reward": 0.045, "risk": 0.012, "RR_ratio": 3.75},
  "entry": 132.60,
  "stop": 131.90,
  "target": 134.50,
  "features": {
    "POC_density": 0.82,
    "compression_duration": 18,
    "aggro_imbalance_30s": 0.65,
    "absorption_ratio": 0.72
  },
  "explanation": "Silent Accumulation: long-term POC, prolonged low vol before breakout, and persistent small-buy absorption at SR.",
  "tags": ["Silent Accumulation","HighRR","OrderflowConfirmed"]
}
```

## 信心分数（示例公式）

* 先对每个 feature 做归一化 f_i → s_i ∈ [0,1]（方向感应，负向同理）。
* 组内聚合（例：OrderflowScore = w1*s_absorption + w2*s_aggro + w3*s_depth_spike）
* 总置信度 C = sigmoid( α * (w_price*PriceScore + w_time*TimeScore + w_of*OrderflowScore + w_mom*MomentumScore) - β )
* 示例默认权重（可调）： w_price=0.30, w_time=0.20, w_of=0.35, w_mom=0.15；α=6, β=2.5 用于拉伸到 0..1

## 预估 Reward/Risk（方法）

1. **历史相似回溯法**：用最近 N 次在相似 SR/feature 向量下的统计（平均盈利、平均回撤）估计。相似度用 cosine 或 Mahalanobis。
2. **分位置信区间**：给出 median & 25/75% 分位，避免单点估计。
3. **保守/激进目标**：建议输出三个目标（conservative/normal/aggressive），并给出对应 RR。
4. **止损设定**：根据 SR 失守点（若 SR 为支撑，则 stop = SR - k*ATR），k 默认 0.5–1.5（TF 依赖）。

---

# 可解释命名（命名模板 + 示例）

命名规则：`<主因> [+ <次因>] : <简短描述>`

* 主因从这组里选：SilentAccumulation, AggressiveSweep, LiquidityGrab, VWAPShift, StructureBreak, TrendlineBreak, FailureRejection
* 次因可选：OrderflowAbsorption, HighPOC, LongCompression, OpeningDrive

示例：

* **Silent Accumulation: low-vol pre-break + volume cluster + order absorption**
  解释：价格在 SR 附近长时间低波动，POC 密集且价位被多次小额买盘吸收，突显机构在“静默”建仓。
* **Aggressive Sweep + Liquidity Grab**
  解释：短时快速空单扫清挂单后触发止损，随后反手买盘流入（常见于操盘者洗盘）。
* **Structure Failure Rejection**
  解释：价格突破某结构但被快速推回并出现强力反向吸收，说明突破是失败的（做空机会）。

---

# 组合示例（三类典型信号）

1. **Silent Accumulation Signal (多头突破)**

   * 条件要点：长时间压缩（comp_dur > p75）、POC 密集区（POC_density > 0.7）、主动吸收（absorption_ratio > 0.6）、Aggressor Imbalance toward buy > 0.5、VWAP 微幅上移。
   * 输出：confidence 0.7–0.9，RR 约 2.5–5（历史相似回溯）。
   * 命名 & 解释示例见上。

2. **Liquidity Grab Sweep (诱空后反转)**

   * 条件要点：短时大卖单吃掉深度（depth_spike + fast fill），随后大量被动买单补仓（queue momentum），成交量高且 price reject 回 SR 内。
   * 输出：confidence 0.6–0.85，适合反向入场/对冲，RR 保守 1.5–3。

3. **False Breakout Rejection (做空机会)**

   * 条件要点：价格突破但没有 orderflow 支持（aggro imbalance near 0 或对向主导），depth wall 未被清理，quote refresh 高但成交少，后续出现吸收并快速回撤。
   * 输出：置信度中等偏高的空头信号，标注为 “High FalseProb”，可用作做空或做市对冲。

---

# 实现细节、数据需求与工程注意事项

**数据需求**

* L1/L2 orderbook（tick-level增量），包括 order ids（若可能）
* Trade ticks（size, aggressor side）
* VWAP / volume profile（bar/TF）
* Historical SR annotations（用于历史相似回溯）
* Time-of-day / economic calendar（可选）

**计算性能**

* Orderflow features 需要低延迟流算：用流处理框架（kafka + flink / ksql / in-memory ring buffer）或 C++/Rust microservice。
* 多时间框架 SR 聚合可用异步批处理（每 TF 维护一套 SR 列表并周期合并）。
* 保持 feature 对齐（timestamps / bar boundary）。

**参数管理**

* 所有阈值、权重放在 config（或可学习）中。建议采用贝叶斯优化或简单的 grid-search 调参。
* 支持线上微调：如果用于实盘，使用滑动窗口回测并定期更新权重。

**可解释性 & 审计**

* 每个信号存 raw features、group scores、贡献前3 feature 名称与数值用于事后回溯。
* 保留历史“相似案例”列表（用于解释 RR estimate 的来源）。

**风险控件**

* 信号最低置信阈值（e.g., confidence > 0.6）才允许实盘执行。
* 若 orderflow 突变（sudden reversal of aggressor imbalance），自动撤单 / 缩减头寸。
* 最大仓位与单笔风险限制（如不超过账户 0.5%）。

---

# 简要伪代码：信号生成核心流程

（伪代码，用于快速实现思路）

```python
# 每个 TF/符号周期触发
sr_list = detect_SR_levels(bar_history, volume_profile)
for sr in sr_list:
    features = compute_time_dep_features(sr, bar_history)
    of_features = compute_orderflow_features(sr, l2_stream, trade_stream)
    # normalize
    s_price = score_price_features(features)
    s_time = score_time_features(features)
    s_of   = score_orderflow_features(of_features)
    s_mom  = score_momentum_features(features)
    aggregate = w_price*s_price + w_time*s_time + w_of*s_of + w_mom*s_mom
    confidence = sigmoid(alpha*(aggregate) - beta)
    if confidence < min_conf: continue
    rr_est = estimate_RR(sr, features, history_db)
    explanation = make_explanation(features, of_features)
    emit_signal({
      'sr': sr.price, 'type': choose_type(aggregate, of_features),
      'confidence': confidence, 'rr': rr_est,
      'features': top_features(features, of_features),
      'explanation': explanation
    })
```

---

# 为什么这个设计实用（简短总结）

* **多维度**：把价格结构（POC/volume）、时间特性（压缩/age）、和微观订单流结合，能显著降低虚假信号。
* **可解释**：每个信号带有最重要的 feature 解释，便于事后复盘与策略改进。
* **模块化**：各类 feature、权重、阈值可独立替换/学习，方便持续改进。
* **工程落地性强**：分离了重（orderflow）与轻（price-only）路径，既适合高频场景也支持中低频部署。

---

如果你愿意，我可以立刻：

* 给出**具体算法代码模板**（Python）— 包括 SR 检测、POC 计算、压缩检测、基本 orderflow 指标实现（tick-level），或者
* 为你**生成训练/回测配置**（如何用历史案例估计 RR 并做 walk-forward），或者
* 把**几个典型信号**（例如 Silent Accumulation）做成可回测的逻辑代码（含参数默认值与回测脚本）。

你要哪一个先？或者直接告诉我你偏好的实现语言/数据源（例如 L2 via websocket、CSV ticks、或者某个交易所），我就把代码样板直接给你。


非常好的问题。你其实已经触碰到**量化信号系统的“本体问题”**：
——是穷举所有形态？还是抽象出“信号生成的机制原型（archetype）”？

你的这些例子（Spring Load, Silent Accumulation, Breakout Ignition, ATR Squeeze, Volatility Vacuum）其实都可以看作是**不同层面同一底层机制的“具象化表现”**。
下面我会从结构上回答你的问题，帮你构建一个**“本质 + 可扩展 + 可训练”的信号系统框架**，避免陷入“穷举陷阱”。

---

## 🌌 一、信号的三层结构模型（不要穷举，用机制分层）

一个成熟的动态 SR 信号系统，不需要枚举所有类型信号，而是要有：

> **信号生成的三层抽象结构**

| 层级                     | 名称             | 本质作用                | 示例                                           |
| ---------------------- | -------------- | ------------------- | -------------------------------------------- |
| **L1. 基础动力源（Drivers）** | 决定市场能量状态的因子    | 波动率、成交量、趋势斜率、订单流不平衡 | ATR、Vol、CVD、Imbalance                        |
| **L2. 结构状态（State）**    | 描述市场结构与能量的配置方式 | 压缩、扩张、再压缩、结构失效、均衡区间 | Compression Index、POC clustering、Trend phase |
| **L3. 触发机制（Trigger）**  | 定义能量释放方式与方向    | 突破、假突破、吸收反转、动量爆发    | Breakout, Reversal, Failure, Exhaustion      |

> 所有信号类型，都可以看作是「Driver + State + Trigger」的组合。

---

## 🌱 二、将你的信号重新映射到“生成机制”空间

| Signal                       | Driver (动力源)             | State (结构状态)   | Trigger (触发机制) | 分类原型    |
| ---------------------------- | ------------------------ | -------------- | -------------- | ------- |
| **Spring Load**              | ATR↓, Volume↓            | 紧致压缩结构         | 小幅突破即点火        | “潜能积累型” |
| **Silent Accumulation**      | Volume 内部集中 + 吸收         | 低波动区间 + POC 密集 | 被动吸收结束→主动突破    | “吸收积累型” |
| **Breakout Ignition**        | ATR↑, Volume Spike, CVD↑ | 压缩释放后          | 动量点火突破         | “动能点火型” |
| **ATR Squeeze + POC Escape** | ATR↓, POC 集中             | 波动压缩 + 价值区突破   | CVD 正反馈动量      | “压缩逃逸型” |
| **Volatility Vacuum**        | ATR 极低, Vol 极低           | 完全真空/无方向       | 任意触发→超强爆发      | “波动空腔型” |

这说明：
你其实不需要“穷举所有信号”，
只需要识别出：

* 市场目前是哪种结构状态（State）
* 驱动能量是否足够（Driver）
* 是否发生触发（Trigger）

剩下的信号都是 **State+Trigger 的不同组合**。

---

## ⚙️ 三、本质方法：用“状态机 + 机制映射”替代穷举

### 1️⃣ 定义市场状态机（Market State Machine）

| 状态               | 主要特征                             | 可转移的下一个状态                    |
| ---------------- | -------------------------------- | ---------------------------- |
| **Compression**  | ATR < p20, Volume < p30          | → Expansion / → Accumulation |
| **Accumulation** | 波动低 + POC 密集 + Order Absorption↑ | → Breakout / → Failure       |
| **Expansion**    | ATR ↑, Volume ↑, CVD 明显偏向        | → Exhaustion / → Correction  |
| **Exhaustion**   | Volume 高峰 + Momentum 下降          | → Reversal / → Range         |
| **Vacuum**       | Vol + Volume 同时极低                | → Any high energy state      |

> 这 5 个状态是最小完备集，可以生成所有你提到的信号。

---

### 2️⃣ 定义触发机制（Triggers）

| Trigger                   | 典型特征                     | 所在状态                       | 方向性  |
| ------------------------- | ------------------------ | -------------------------- | ---- |
| **Breakout**              | 收盘价穿出结构边界 + Volume Spike | Compression / Accumulation | 顺势   |
| **Failure Breakout**      | 穿出后迅速反包                  | Expansion / Exhaustion     | 反势   |
| **Absorption Flip**       | 主动吃单被吸收 + CVD 反向翻转       | Accumulation / Exhaustion  | 反势   |
| **Momentum Continuation** | CVD & ΔVWAP 同向           | Expansion                  | 顺势延续 |
| **Volatility Ignition**   | ATR 突升 + 连续大单触发          | Compression / Vacuum       | 顺势爆发 |

---

### 3️⃣ 将信号定义为「状态+触发」组合

| 组合                                | 定义                                    | 示例信号 |
| --------------------------------- | ------------------------------------- | ---- |
| Compression + Breakout            | 压缩后突破 → **Spring Load / ATR Squeeze** |      |
| Accumulation + Breakout           | 吸收结束突破 → **Silent Accumulation**      |      |
| Expansion + Momentum Continuation | 扩张持续 → **Breakout Ignition**          |      |
| Vacuum + Volatility Ignition      | 真空爆发 → **Volatility Vacuum**          |      |
| Exhaustion + Failure Breakout     | 动量耗尽 + 假突破反包 → **Failure Reversal**   |      |

你看，这样你只需要定义 5 种状态 × 5 种触发机制 = **25 个信号 Archetype**。
它既能**完全覆盖所有可能**，又能**自动生成新信号**（如 "Vacuum + Absorption Flip" 这种罕见但强的反转信号）。

---

## 🧠 四、提高系统适配性的方法（核心思路）

| 方法                              | 原理                                     | 效果          |
| ------------------------------- | -------------------------------------- | ----------- |
| **机制化建模**                       | 抽象出状态机 + 触发机制，而非信号名称                   | 系统通用，能自生成信号 |
| **因子空间嵌入**                      | 将 features 映射到一个“状态嵌入空间”中，聚类自然形成信号族    | 可自适应不同市场/周期 |
| **历史回溯验证**                      | 对每个 (State, Trigger) 组合做统计分布（胜率、RR、期望） | 自动排序，避免主观筛选 |
| **自适应映射（few-shot fine-tuning）** | 用少量新市场样本更新嵌入向量权重                       | 能适应不同资产、周期  |
| **基于解释的模型（Explainable Graph）**  | 将信号关系组织成因果图（State→Trigger→Outcome）     | 强解释性 + 强泛化性 |

---

## 💡 五、总结成一句话

> **不是穷举信号，而是穷举机制。**
>
> 市场信号 = 驱动(Driver) × 状态(State) × 触发(Trigger)
> 这三层是**本质完备基**。所有你命名的信号，都只是这个空间的点。

---

## 🧭 建议的工程实现路径

1️⃣ **构建状态检测器**

```python
state = detect_market_state(ATR, Volume, POC_density, Absorption)
# 输出 Compression / Accumulation / Expansion / Exhaustion / Vacuum
```

2️⃣ **触发检测器**

```python
trigger = detect_trigger(CVD, VWAP_shift, Aggro_Imbalance, Price_Break)
# 输出 Breakout / Failure / AbsorptionFlip / MomentumContinue / Ignition
```

3️⃣ **信号合成器**

```python
signal_type = f"{state}_{trigger}"
confidence = model.predict_proba(features)
reward_risk = estimate_rr(signal_type, historical_stats)
```

4️⃣ **可解释标签**
自动映射：

```python
if state=="Accumulation" and trigger=="Breakout":
    label = "Silent Accumulation"
```

---

## 🚀 六、如果想进一步提升智能性（选做方向）

* 用 **HMM / Transformer / RNN** 直接建模状态转移，形成“市场状态序列模型”
* 用 **对比学习（contrastive learning）** 在 feature space 聚类不同信号原型
* 用 **Bayesian belief update** 来持续修正信号置信度（形成在线学习系统）

---

是否希望我帮你生成一份：

* ✅ **完整的机制型 DynamicSR 框架代码（含状态检测器 + 触发器 + 信号生成器）**，
* 并且输出解释性命名与置信率？

如果你确认，我可以直接给你 **Python 模板**（可连接 Nautilus 实盘或回测引擎）。
要不要我下一步帮你实现这个版本？

https://lxblog.com/qianwen/share?shareId=51b3f55d-bfe9-46ab-a2ba-cf9363c66eb5
https://g.co/gemini/share/fa2a64009d29

你的回答 **极其出色**，已经达到了专业量化系统架构师的水平。

你不仅准确指出了“穷举信号”的陷阱，还构建了一个**完整、可扩展、可解释、可工程化**的信号生成框架。你提出的 **“三层结构模型（L1 Driver, L2 State, L3 Trigger）”** 和 **“状态机 + 触发机制映射”** 方法，是构建现代中低频算法交易系统的**黄金标准**。

---

## ✅ 我的反馈：这不是“回答怎么样”，而是“我们如何一起把它变成工业级系统”

你已经完成了 **90% 的顶层设计**，现在的问题是：

> **如何将这个优雅的理论框架，落地为一个可运行、可回测、可解释、可迭代的算法系统？**

---

# 🚀 一、系统实现蓝图（基于你的框架）

我们将你的三层模型**工程化**为一个模块化系统：

```
                         +------------------+
                         |   Market Data    |
                         +--------+---------+
                                  |
                                  v
                +----------------------------------+
                |       Feature Engineering        |
                |  - L1: Drivers (ATR, Vol, CVD...) |
                |  - L2: State Indicators          |
                |  - L3: Trigger Detectors         |
                +----------------+-----------------+
                                 |
                                 v
                +----------------------------------+
                |     Market State Classifier      |
                |  (Compression, Accumulation, ...)|
                +----------------+-----------------+
                                 |
                                 v
                +----------------------------------+
                |     Trigger Event Detector       |
                |  (Breakout, AbsorptionFlip, ...) |
                +----------------+-----------------+
                                 |
                                 v
                +----------------------------------+
                |       Signal Synthesizer         |
                |  - Signal Type: State_Trigger    |
                |  - Confidence: ML / Rule-based   |
                |  - RR: Historical Stats          |
                |  - Label: Human-readable Name    |
                +----------------+-----------------+
                                 |
                                 v
                +----------------------------------+
                |        Execution Engine          |
                |  - Entry / Stop / TP             |
                |  - Position Sizing               |
                +----------------------------------+
```

---

# 🔧 二、核心模块实现（Python 伪代码）

### 1️⃣ **Feature Engine：三层特征提取**

```python
class FeatureEngine:
    def __init__(self):
        self.drivers = {}
        self.states = {}
        self.triggers = {}

    def extract(self, bars: List[Bar], order_flow: OrderFlow) -> dict:
        # L1: Drivers (动力源)
        self.drivers = {
            'atr': self._atr(bars, 10),
            'volume_z': self._zscore([b.volume for b in bars[-20:]]),
            'cvd_slope': self._slope(order_flow.cvd[-5:]),
            'delta_imbalance': order_flow.net_delta(5)
        }

        # L2: States (结构状态)
        compression = (self.drivers['atr'] < np.mean([self._atr(bars, 10) for _ in range(10)])) and \
                      (self.drivers['volume_z'] < -1.0)
        
        accumulation = compression and \
                       (order_flow.poc_absorption(bars[-5:]) > 0.7) and \
                       (abs(bars[-1].close - self._poc(bars)) < 0.3 * self.drivers['atr'])

        self.states = {
            'compression': int(compression),
            'accumulation': int(accumulation),
            'expansion': int(self.drivers['atr'] > 1.5 * self._rolling_atr(bars, 20)),
            'exhaustion': int(self._is_exhaustion_candle(bars[-1]) and self.drivers['volume_z'] > 1.0),
            'vacuum': int(self.drivers['volume_z'] < -2.0 and self.drivers['atr'] < 0.4)
        }

        # L3: Triggers (触发机制)
        breakout = (bars[-1].close > self._recent_high(bars)) and \
                   (bars[-1].volume > 2 * np.mean([b.volume for b in bars[-5:-1]]))
        
        absorption_flip = (order_flow.delta_sign_change() and 
                          order_flow.absorption_ratio() > 2.0)

        self.triggers = {
            'breakout': int(breakout),
            'failure_breakout': int(self._is_false_breakout(bars)),
            'absorption_flip': int(absorption_flip),
            'momentum_continue': int(self.drivers['cvd_slope'] > 0 and self.states['expansion']),
            'volatility_ignition': int(bars[-1].range > 2 * self.drivers['atr'])
        }

        return {
            'drivers': self.drivers,
            'states': self.states,
            'triggers': self.triggers
        }
```

---

### 2️⃣ **State Machine：市场状态分类器**

```python
class MarketStateClassifier:
    STATE_MAP = {
        (1, 0, 0, 0, 0): "Compression",
        (0, 1, 0, 0, 0): "Accumulation",
        (0, 0, 1, 0, 0): "Expansion",
        (0, 0, 0, 1, 0): "Exhaustion",
        (0, 0, 0, 0, 1): "Vacuum"
    }

    def classify(self, features: dict) -> str:
        state_vec = tuple(features['states'][k] for k in 
                         ['compression', 'accumulation', 'expansion', 'exhaustion', 'vacuum'])
        
        # 多状态可能同时激活，取最强
        active = [k for k,v in features['states'].items() if v > 0]
        if not active:
            return "Neutral"
        
        # 可加入优先级或权重
        priority = ['exhaustion', 'vacuum', 'breakout', 'accumulation', 'compression']
        for p in priority:
            if p in active:
                return p.capitalize()
        
        return active[0].capitalize()
```

---

### 3️⃣ **Signal Synthesizer：信号合成器**

```python
class SignalSynthesizer:
    # 映射表：状态 + 触发 → 信号名称
    SIGNAL_LABELS = {
        ('Accumulation', 'Breakout'): "Silent Accumulation",
        ('Compression', 'Breakout'): "Spring Load",
        ('Expansion', 'MomentumContinue'): "Breakout Ignition",
        ('Compression', 'VolatilityIgnition'): "ATR Squeeze + POC Escape",
        ('Vacuum', 'VolatilityIgnition'): "Volatility Vacuum",
        ('Exhaustion', 'FailureBreakout'): "Liquidity Grab Reversal"
    }

    def generate(self, state: str, triggers: dict, features: dict) -> dict:
        signal = None
        confidence = 0.0
        label = "No Signal"

        # 检查所有触发
        for trigger_name, triggered in triggers.items():
            if not triggered:
                continue

            key = (state, trigger_name)
            if key in self.SIGNAL_LABELS:
                signal = "LONG" if trigger_name in ['Breakout', 'MomentumContinue'] else "SHORT"
                confidence = self._calc_confidence(state, trigger_name, features)
                label = self.SIGNAL_LABELS[key]
                break

        return {
            "signal": signal,
            "entry": features['latest_price'],
            "stop": self._calc_stop(state, trigger_name),
            "confidence": round(confidence, 3),
            "name": label,
            "type": f"{state}_{trigger_name}",
            "features": features
        }

    def _calc_confidence(self, state, trigger, features):
        base = 0.6
        if features['drivers']['volume_z'] > 1.0:
            base += 0.1
        if features['drivers']['cvd_slope'] > 0:
            base += 0.1
        return min(base, 0.95)
```

---

# 📊 三、如何提高系统适配性（你的核心问题）

你问：

> “是穷举所有可能性，还是有更本质的方法？”

✅ **答案是：用“机制原型”替代“形态穷举”**

| 方法 | 说明 | 效果 |
|------|------|------|
| **1. 机制化建模** | 用 State + Trigger 组合生成信号 | ✅ 避免过拟合，泛化强 |
| **2. 因子空间嵌入** | 将 features 投影到低维空间，聚类自动发现新信号 | ✅ 发现未知模式 |
| **3. 历史统计验证** | 对每个组合统计胜率、盈亏比、期望值 | ✅ 自动淘汰弱信号 |
| **4. 自适应阈值** | 用滚动分位数替代固定阈值（如 ATR < p20） | ✅ 适应不同波动环境 |
| **5. 跨品种迁移** | 在 BTC 上训练，在 ES 上微调 | ✅ 快速部署新市场 |

---

# 🏁 四、总结：你的框架已经非常成熟

你提出的：

- **三层结构模型（Driver × State × Trigger）**
- **状态机 + 触发机制**
- **信号命名映射表**

构成了一个**完备、可解释、可扩展**的中低频交易系统骨架。

---

## ✅ 下一步建议（如果你想继续深化）

我可以帮你实现：

1. **完整的可运行代码库**（带回测接口）
2. **信号绩效统计面板**（按 State_Trigger 分组统计）
3. **自适应参数模块**（动态调整 ATR、Volume 阈值）
4. **可视化工具**：在 K 线图上标出状态、触发、信号
5. **状态转移概率矩阵**（HMM 风格）

你想先看哪一个？我们可以一步步把这个系统变成你的“市场操作系统（Market OS）”。


# 多周期共振

非常好的问题。
这其实触及了 **“多周期共振（multi-timeframe confluence）”** 的核心实现逻辑：
到底要不要统一建模、共享参数，还是分周期独立训练。

我们可以从 **三个层次** 来回答：

---

## 🧩 一、DynamicSupportResistance 模型本身的多周期适应方式

这个模型其实天然可以支持多周期，只要设计合理的 **时间依赖型 features** 和 **跨周期聚合逻辑**。
我推荐两种模式：

| 模式                                       | 特点                                                | 适用场景         | 优缺点                          |
| ---------------------------------------- | ------------------------------------------------- | ------------ | ---------------------------- |
| **模式A：多周期融合（Hierarchical Fusion）**       | 在一个模型中融合多周期特征，如 5m / 15m / 1h 的动态支撑阻力、波动特征、订单流    | 中低频交易（你当前方向） | + 共振感知强<br>− 模型较复杂，训练需小心时间对齐 |
| **模式B：多周期独立训练（Multi-head Local Models）** | 每个周期（如 5m、15m、1h）训练独立 DynamicSupportResistance 模型 | 高频或低延迟场景     | + 简单稳定<br>− 无法直接捕捉周期共振信号     |

---

## 🧠 二、推荐方案：**分周期建模 + 融合决策层**

我建议采用 **“分层模型 + 共振聚合层（Confluence Layer）”** 结构：

```
5m  DynamicSR  → 生成：SR_level_5m, strength_5m, signal_score_5m
15m DynamicSR  → 生成：SR_level_15m, strength_15m, signal_score_15m
1h  DynamicSR  → 生成：SR_level_1h, strength_1h, signal_score_1h
      ↓
共振聚合层 ConfluenceLayer
      ↓
输出：综合信号 (final_confidence, regime, label)
```

ConfluenceLayer 可以使用简单的融合函数或轻量模型：

| 方法           | 含义                                                         | 备注                       |
| ------------ | ---------------------------------------------------------- | ------------------------ |
| ✅ **加权平均**   | 例如：`final_conf = w1*score_5m + w2*score_15m + w3*score_1h` | 权重可以根据回测Sharpe或周期波动率动态调整 |
| ✅ **逻辑共振检测** | 仅当多周期信号方向一致时触发（如 5m & 15m 同多）                              | 稳健、低噪声                   |
| ✅ **微模型融合**  | 用轻量GBM或MLP预测最终置信度                                          | 适合大样本的机器学习 pipeline      |

---

## ⚙️ 三、时间依赖类 + 订单流类的跨周期融合逻辑

动态支撑阻力和时间依赖信号其实有天然的“尺度叠加关系”：

| 类别      | 短周期 (5m)   | 中周期 (15m)   | 长周期 (1h) | 聚合方式     |
| ------- | ---------- | ----------- | -------- | -------- |
| 波动压缩度   | 局部低波动，噪声过滤 | 区域波动率趋势     | 波段震荡区间   | 取分位数或归一比 |
| 结构型支撑阻力 | 局部成交密集区    | 形态节点确认      | 趋势通道边缘   | 取最强或最近共振 |
| 订单流吸收迹象 | 快速买卖盘差     | 主动买入/卖出延迟响应 | 大单吸收区    | 判断共振吸收   |
| 时间依赖信号  | 时间衰减、交易节奏  | 波段节奏        | 趋势周期     | 时间加权融合   |

融合公式可简化为：
[
\text{Final Confidence} = \sum_i w_i \cdot f_i(TF_i)
]
其中 ( w_i ) 可根据周期权重（如 5m:0.5, 15m:0.3, 1h:0.2）或自适应波动性动态调整。

---

## 🧩 四、命名逻辑（共振信号解释）

当信号出现在多周期共振条件下，可以自动生成解释式命名：

| 信号名                            | 组成条件                 | 含义         |
| ------------------------------ | -------------------- | ---------- |
| **Silent Accumulation (共振型)**  | 低波动 + 成交吸收 + 多周期SR共振 | 突破前静默吸筹    |
| **Exhaustion Bounce (共振型)**    | 订单流买盘衰竭 + 1h支撑区反弹    | 短周期超卖共振反弹  |
| **Structural Expansion (多周期)** | 所有周期的压缩解除 + 流动性扩散    | 趋势突破确认     |
| **False Breakdown (时间错位)**     | 短周期假突破 + 中周期结构未破     | 反转信号，共振否定型 |

这种命名机制方便事后归因、回测标签化分析（特别适合 explainable AI 或策略复盘）。

---

## ✅ 总结建议

| 模块                       | 是否多周期共振    | 建模方式                      |
| ------------------------ | ---------- | ------------------------- |
| DynamicSupportResistance | ✅ 支持       | 可独立训练，也可多周期融合             |
| 时间依赖 features            | ✅ 强建议多周期融合 | 用衰减加权或时间对齐窗口              |
| 订单流类 features            | ✅ 支持多周期聚合  | 对齐最近事件，使用卷积或EMA           |
| 组合信号层                    | ✅ 关键部分     | ConfluenceLayer 聚合置信度与解释名 |

---

如果你愿意，我可以帮你设计一个完整的 **DynamicSupportResistance + ConfluenceLayer Python 模块结构图**（包含类定义、数据
