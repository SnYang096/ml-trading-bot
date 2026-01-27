# 策略子模型（Sub-model）开发手册：规则 vs 树模型策略 vs NN 原语+Router+Execution

这份文档回答一个非常工程化的问题：

> 当你不断发现新的“市场模式 / 交易形态”（pattern），**哪种开发方式更容易持续产出更多子模型**，并且**上线维护成本不会爆炸**？

这里的“子模型”不只指 ML 模型（LGBM/NN），也包含“可上线、可监控、可回退”的**可执行策略单元**。

---

### 1. 术语对齐：pattern / 子模型 / mode / Router / Execution

- **pattern（模式/形态）**：你观察到的结构性现象（例如“插针爆空→大反转”）。
- **子模型（sub-model）**：可部署的策略单元，至少包含：
  - **gating**：何时允许交易（结构过滤）
  - **score**：机会质量（排序/阈值/信号强度）
  - **execution**：怎么进出（止损止盈、持仓上限、分批、滑点/费用假设）
  - **monitoring**：如何监控失效并回退
- **mode（行为模式）**：系统级的少量动作空间，例如 `{NO_TRADE, MEAN, TREND}`。
- **Router**：把“输入状态/预测”映射成 mode/仓位/风控的模块。
- **Execution**：在给定 mode 下执行并产出一致口径回报（对齐回测/实盘）。

**核心工程结论**：

> 要“不断增加子模型”且不把系统拖死，关键不是“能不能再训练一个模型”，而是：  
> **新增的东西到底是在“策略语义层”膨胀，还是能被压缩进少量 mode + execution 模板里。**

---

### 2. 三种开发方式的“可扩展性”本质差异

### 2.0 为什么需要 gating？能不能只用 score？新特征是不是直接增强 NN 原语模型即可？

你提出的是最关键的工程取舍：**Router 里到底要不要显式 gating（硬过滤）**，以及 **pattern 新特征应该放到哪里**。

#### 2.0.1 Score-only（只靠打分/阈值）什么时候足够？

如果满足以下条件，score-only 往往就能跑得很好：

- 你的 score **可校准**（同一分数在不同时间/币种含义接近），并且你愿意用阈值控制交易频率
- 交易成本/滑点对你来说 **不是“必须先 veto 才能安全”的级别**
- 你的模式不是极端稀有事件（不是“每年几次”的插针清算那种）

这种情况下，**score + 阈值** 本质上等价于一个“软 gating”，你可以不写显式 if/else。

#### 2.0.2 为什么先进做法仍然保留 gating（至少保留一层硬 veto）？

在生产系统里，gating 通常承担 **“否决权（veto）”**，它解决的是 score 很难优雅解决的四类问题：

- **成本与风险硬约束**：例如波动/流动性/价差异常时直接 NO_TRADE（否则再高分也不该开）
- **分布外/漂移保护**：当特征缺失、数据质量异常、或 heads 明显漂移时先 veto（避免 score 在 OOD 下瞎输出）
- **稀有事件的工程化**：极端事件样本极少，纯 score 很容易过拟合或校准崩；用 gating 把事件检测与执行口径分离更稳
- **行动空间稳定**：你希望 action 分布长期稳定（不要出现“长期全 NO_TRADE/全 TREND”），gating 可以把“结构允许交易”与“质量评分”解耦

> 最先进的实践里，gating 往往不是为了“更聪明”，而是为了 **更可控、更可审计、更容易回退**。

#### 2.0.3 硬 gating vs 软 gating（推荐折中）

- **硬 gating**：`if not allowed: NO_TRADE`（用于风险/数据质量/结构性 veto）
- **软 gating**：把 detector 输出当作权重/先验，例如 `score_trend *= (1 + w * detector)` 或者只在 detector=True 的子集内重新校准阈值

实际建议：**保留一层硬 veto + 其余尽量用软 gating/score**，这样既不规则爆炸，也不会失去安全闸门。

#### 2.0.4 pattern 新特征要放到哪里：Router detector 还是增强 NN 原语？

**能直接增强 NN 原语模型（更“底座化”）的特征**通常满足：

- **通用性强**：对多类模式都有帮助（不是只为某个 pattern 服务）
- **连续可学习**：不是“触发/不触发”的极端离散事件，或者离散事件也能稳定编码
- **跨时间/跨币种稳定**：分布可控（配合 Feature Contract/归一化），不容易 drift
- **不会引入泄露**：严格因果、口径明确、缺失策略清晰

**更适合放在 Router detector/gating 的特征**通常是：

- **强语义、强事件驱动**：插针/清算/极端量能/关键位结构这种“发生了就很重要”的模式
- **样本稀少**：你更希望用显式结构检测把样本切片出来，而不是让底座去“暗中学会”
- **上线迭代快**：你想不重训底座就能快速增加/修改模式逻辑

结论是：

> **新增特征并不总是要喂给 NN 原语模型。**  
> 优先把“强语义/稀有事件/快速迭代”的东西做成 detector（Router 输入）；  
> 把“通用、连续、稳定”的东西逐步沉淀进 NN 原语底座（Feature Contract）。

#### 2.0.5 gating 和 Router detector 的区别（最容易混淆，但必须分清）

- **Router detector（检测器）**：把原始特征/结构信息压缩成“事件/结构是否发生”的信号（或强度），例如：
  - `wick_event_strength`（插针强度）
  - `absorption_score`（吸收/反复测试阻力的结构强度）
  - `failed_breakout_flag`（假突破/跟风不足）
  - 这些通常是 **输入侧的 feature engineering**，目标是：让 Router 更容易利用“结构信息”。

- **gating（闸门/否决权）**：Router 内部（或执行前）对“是否允许交易/允许哪类 action”做硬约束：
  - 风险/成本/数据质量 veto：`if spread_too_wide: NO_TRADE`
  - 结构 veto：`if not detector: NO_TRADE`（只在特定结构下才允许某类交易）
  - 关键点：gating 是 **决策层的安全边界**，不是为了更聪明，而是为了更可控、可审计、可回退。

一句话记忆：

> **detector 负责“看见结构”，gating 负责“不给犯错机会”。**

#### 2.0.5.1 你现在卡住的关键：detector 在这里“是什么”？

你说得对：**detector 的计算用到的东西，本质上也是特征（feature）**。区别不在“数学形态”，而在“它被哪一层消费、承担什么职责”：

- **model features（模型输入特征）**：喂给 NN 多头（MLP）去学路径原语 heads（`dir/mfe/mae/ttm/...`）。  
  - 一旦改动，通常需要 **重训**，并受 Feature Contract/归一化/缺失策略的强约束。
- **router detectors（Router 输入的显式结构变量）**：从同一份 features 表里派生出来的 **结构/事件信号（flag 或 strength）**，直接给 Router 做 gating/加权/阈值切片。  
  - 优点是：你可以 **不重训底座** 就快速迭代，而且更可解释、更可审计。

因此在本文语境里：

> **detector = 显式命名的结构变量 \(d = f(X)\)**，它可以由任何原始特征集合 \(X\) 派生；  
> 但它的“身份”是 Router 输入，而不是 NN 多头的训练输入。

#### 2.0.6 代码走读：detector → gating/score → action（NO/MEAN/TREND）→ execution 模板

仓库里已经有一条非常贴近“先进做法”的最小闭环：

- **固定 action（NO/MEAN/TREND）**：见 `src/time_series_model/rule/router_3action.py`
  - 它用 heads（`pred_*`）算出可交易条件（tradable）并进一步分 MEAN/TREND
  - 这段逻辑里同时包含了：
    - **gating**：`tradable`（不满足就 NO_TRADE）
    - **score/条件**：对 trend/mean 的分流条件（效率/时间/方向置信）

- **execution 模板（RR exit）**：见 `src/time_series_model/rl/execution_returns_rr.py`
  - 它把 mode 映射为方向（TREND=顺势，MEAN=逆势），并用 ATR R/R 规则做出入场与退出仿真
  - 同样包含一个 entry gate（基于 `head_mfe_atr/head_mae_atr`）来保证“没 edge 就不进场”

如果你要把 “插针爆空→大反转” 加进系统，但又不想新增 action，推荐做法是：

- 写一个 detector（输入特征→结构强度/flag）
- 在 Router 里把 detector 作为：
  - **硬 gating**（只在 detector=True 时允许 MEAN）
  - 或 **软 gating**（提升/压制 MEAN/TREND 的 score，再由阈值决定）
- Execution 仍然复用 MEAN/TREND 的模板（RR、时间止损、追踪等），保证口径一致

（示意代码，非仓库现有实现）：

```python
def wick_reversal_detector(df) -> pd.Series:
    wick_ratio = (df["open"] - df["low"]) / (df["high"] - df["low"] + 1e-9)
    vol_z = (df["volume"] - df["volume"].rolling(240).mean()) / (df["volume"].rolling(240).std() + 1e-9)
    return (wick_ratio > 0.65) & (vol_z > 2.0)

def router_score_fixed_actions(heads, detector_flag):
    # action 空间固定：NO/MEAN/TREND
    score_trend = heads["dir_conf"] * heads["mfe_atr"] / (heads["mae_atr"] + 1e-9)
    score_mean = (1.0 - heads["dir_conf"]) * heads["mfe_atr"] / (heads["mae_atr"] + 1e-9)

    # 软 gating：结构事件发生时更偏向 MEAN
    score_mean = score_mean * (1.0 + 0.5 * detector_flag.astype(float))

    # 硬 veto：风险/数据质量层（示意）
    if heads["mfe_atr"] < 0.4:
        return "NO_TRADE"
    return "MEAN" if score_mean > score_trend else "TREND"
```

#### 2.0.7 怎么决定“哪些特征进 NN 底座”，哪些更适合当 detector？

实战里推荐用“默认分流”原则，避免一开始把系统复杂度推爆：

- **优先放进 NN 底座（model features）的特征**（更“原语化/底座化”）：
  - **跨策略通用**：描述市场状态/路径几何，跟具体 entry/exit 语义弱绑定
  - **连续、稳定、可归一化**：跨币种/跨时期尺度可控（最好 ATR/收益率无量纲化）
  - **缺失策略清晰**：实盘/回放中稳定可算，不依赖稀有数据源
  - **不会引入泄露**：严格因果对齐（只用 \(t\) 时刻及以前信息）

- **优先当 detector（Router features）的特征/信号**（更“语义化/事件化”）：
  - **强语义/强事件**：插针、清算、关键位结构、假突破等“发生了就很重要”
  - **稀有且分布尖锐**：样本太少，放进 NN 容易学不稳、校准漂
  - **你希望快速迭代**：改阈值/定义不想重训底座
  - **你需要否决权**：它更多是“风险/结构 veto”，而不是让模型自己自由发挥

关于你的问题：“树模型各个策略非常有效的特征，能不能都放到 NN 训练里？”

- **不建议“全塞进去”**：树模型的“有效特征”往往和某个策略标签/执行口径强绑定；直接塞进 NN 底座容易把底座变成“隐式策略模型”，破坏原语复用与可维护性。
- **更推荐的用法**：
  - 把树模型里反复出现、跨策略也有意义的那部分，**提炼成更原语化的连续特征**（例如波动/趋势/压缩强度/结构距离），再进入 NN 底座；
  - 把强策略语义的那部分，先做成 **detector**（Router 输入），用 shadow/counterfactual 验证稳定后，再决定是否下沉到底座。

一句话建议：

> **NN 底座尽量“原语化”，Router(detector/gating/score) 承载“语义化”。**  
> 当某个语义信号在多阶段、多币种、多切片稳定有效时，再考虑把它下沉为底座特征。

#### 2.0.8 Detector-first → 验证门槛 → 下沉到底座（流程图式规则）

你说得对：这不是一道“唯一正确”的数学题，但可以把它工程化成一套**默认策略 + 明确门槛**，让决策尽量客观、可重复。

**默认策略（强烈推荐）**：任何新模式/新信号，先做 **detector（Router 输入）**，不要一上来就进 NN 底座。

**流程图（可照抄执行）**：

```text
新想法/新特征
  ↓
先实现为 detector（Router 输入的显式结构变量 d=f(X)）
  ↓
在 Router 里以“软 gating/加权”为主，保留一层硬 veto（风险/数据质量）
  ↓
跑验证（OOS + 多切片 + 多 seed）
  ↓
满足下沉门槛？ —— 否 ——> 保持为 detector（继续迭代阈值/定义，零重训成本）
      |
     是
      ↓
提炼/归一化/写入 Feature Contract
  ↓
作为 model feature 下沉进 NN 底座（触发重训 + 回归 + shadow 验收）
```

**“下沉门槛”（尽量硬化的标准）**：只有满足多数（建议 3/4 以上）才考虑下沉进 NN 底座。

- **稳定性门槛（必须）**：在多个时间窗口、多个 symbol、多个 seed 下，提升是稳定的（不是一次性运气）。
- **泛化性门槛（强烈建议）**：它对不止一个 Router 子策略有帮助（不是只对某个策略语义/单标签有效）。
- **可控性门槛（必须）**：
  - 分布可控（可归一化、不会在不同币种/不同阶段尺度乱飘）
  - 缺失策略清晰（missingness policy 可写进 contract）
  - 因果对齐明确（严格不泄露）
- **工程 ROI 门槛（必须）**：下沉带来的增益，足以覆盖“重训 + 回归测试 + 上线风险”的成本；否则保持 detector 更划算。

> 经验法则：**只要你还在频繁改定义/阈值，它就更像 detector，而不是底座特征。**

#### 2.1 纯规则（手写 if/else）

**扩展方式**：新增一个 pattern，通常就是新增/修改一段 gating + 条件阈值 + 交易动作。

- **最适合**：快速验证想法、做 baseline、做生产 fallback、做硬风控闸门（veto）
- **可扩展性瓶颈**：当 pattern 多到一定程度，规则会出现：
  - 条件之间互相打架（冲突/覆盖）
  - 参数维护困难（阈值漂移、不同币种/不同波动环境难统一）
  - 规则爆炸（几十个 if/else 之后可读性和可审计性急剧下降）

**工程建议**：

- 把规则写成“模板化的子策略”，每个子策略有明确名字、输入特征清单、输出行为与回测口径。
- 规则永远保留，但尽量让它承担 **baseline/fallback/安全闸门**，而不是无限承载 alpha。

#### 2.2 树模型策略（每个策略一个模型/一套特征）

**扩展方式**：新增 pattern → 通常新增一个策略目录（或在现有策略族里新增 variant），并训练一个树模型（或若干模型）。

- **最适合**：
  - 你有很多策略语义（不同 entry/exit 逻辑）要并行试
  - 你强依赖特征选择与非线性组合（`feature-group-search`/Pool B/语义分组）
  - 你希望每个策略都有比较强的可解释性与独立评估
- **可扩展性瓶颈**：
  - “策略×币种×多空×参数”乘法增长，模型数量容易爆炸
  - 每个策略有自己的标签口径/执行口径，**一致性维护成本高**

**工程建议**：

- 用 `features_base.yaml` 把“标签/回测必需特征”固定住，避免 baseline 不能跑。
- 把新增 pattern 当作“新增策略族/策略 variant”，但要强制走统一的：
  - 数据/特征落盘（FeatureStore）
  - 评估报告模板
  - 回退/上线闸门（Shadow/Fallback/FSM）

#### 2.3 NN 原语（多头）+ Router + Execution（少量 mode + 模板化执行）

**扩展方式（推荐理解）**：

新增 pattern 时，尽量不要新增“一个完整新模型”，而是：

- **新增 Router 子模块**（新的 gating + score），仍然复用同一套原语 heads
- 少量情况下才新增/微调 head（例如多一个“结构确认/失效概率”的辅助 head）
- Execution 尽量不新增模板，而是用参数化（risk/exits）去适配

**为什么它更容易“扩展出很多子模型”？**

- 因为你把系统“降维”成少量 mode：  
  新 pattern 多数只是 **“什么时候用 MEAN/TREND”** 的新条件与新评分方式。
- 因为 “模型复用” 是结构性的：  
  你可以拥有 **N 个 Router 子策略**，但只维护 **1 个（或很少几个）原语底座模型**。

**可扩展性瓶颈**：

- 前期要把 Feature Contract、归一化、训练-推理一致性、监控与回退先打穿
- Router 设计要足够规范，否则 Router 自己会变成新的“规则爆炸”

**工程建议**：

- Router 子策略必须以“模块化配置/函数”存在，并且每个都能单独做 shadow/counterfactual。
- 强制把 Router 子策略的输入限制在：
  - 少量结构特征（趋势/压缩/水平位/波动状态）
  - 原语 heads 的输出（dir/mfe/mae/ttm/(persistence)）
- 新增 pattern 的默认动作是：**新增一个 Router 子策略**，不是新增一个模型。

---

### 3. 用同一组 pattern 举例：三种方式怎么落地

你给的 3 个例子都非常典型：它们更像是“结构事件 + 行为模式选择”，非常适合用 `{NO, MEAN, TREND}` 来承载。

下面每个 pattern 我都会给出：**规则版**、**树模型策略版**、**NN 原语+Router 版** 的落地方式。

#### 3.1 Pattern A：大趋势向上 + 插针爆空（巨大下影/清算）→ 大反转

**直觉**：上升趋势中出现极端下探（短时恐慌/清算），但结构未破坏，往往触发均值回归式反弹。

- **规则版（最快验证）**
  - **gating**：trend 上行（例如 MA/结构高低点/回撤限制）
  - **event**：下影线比例、当根振幅/ATR、成交量/成交额 spike
  - **entry**：事件发生后等待“收回”（close 回到某阈值之上）再做多
  - **exit**：MEAN 模板（RR、时间止损、最大持仓时长）

- **树模型策略版（更像一个独立策略）**
  - 新建一个策略变体，例如：`wick_liquidation_reversal_long`
  - **标签**：在事件发生后，未来 H 内的“反弹幅度/成功概率”
  - **特征**：wick_ratio、range/ATR、volume_z、trend_state、距关键位（SR/均线/前高）
  - **优点**：能学到“哪些插针是真的反转，哪些是继续下跌”
  - **代价**：模型数量增加（这是一个新策略）

- **NN 原语 + Router 版（推荐长期形态）**
  - **关键原则**：**action 空间固定**（例如 `{NO_TRADE, MEAN, TREND}`），新增 pattern **不应新增 action**，而是新增“pattern detector / gating & scoring module”。
  - 不新增底座模型；新增一个**模式检测器**（例如 `wick_reversal_detector`），作为 Router 的一个输入/模块
  - **gating（结构）**：`trend_state=UP` 且 `wick_event=True`
  - **score（用原语 heads）**（仍然只是在固定 action 上打分/选路由）：
    - 选择 **MEAN** 还是 **NO_TRADE**：看 `mfe_atr` 是否足够大、`mae_atr` 是否可控、`t_to_mfe` 是否合理
  - **execution**：复用 **MEAN** 模板（统一风控口径）
  - **好处**：你可以很快增加很多“事件/结构驱动的 detector”，但底座只维护一套原语模型；系统的执行与风控仍然通过固定模板保持一致性

#### 3.2 Pattern B：大趋势向上 + 反复测试阻力位（阻力缓慢上移）→ 放量突破

**直觉**：上行趋势中“多次触碰 + 上移”的阻力/供给吸收结构，最终放量突破更可信。

- **规则版**
  - **gating**：trend 上行
  - **structure**：阻力线（局部高点回归线/水平区）+ touch_count + resistance_slope>0
  - **trigger**：突破时放量（volume_z/成交额）+ 收盘站上阻力
  - **execution**：TREND 模板（分批、回撤止损、追踪止盈）

- **树模型策略版**
  - 新建策略变体，例如：`rising_resistance_breakout`
  - **标签**：突破后 H 内的 continuation（例如 mfe_atr、持有收益、或突破失败概率）
  - **特征**：touch_count、resistance_slope、breakout_strength、volume_z、波动状态/压缩状态
  - **优点**：能学到“什么样的触碰/上移结构更可靠”
  - **代价**：仍然是“每策略一模型”的乘法结构

- **NN 原语 + Router 版**
  - 新增一个模式检测器（例如 `absorption_breakout_detector`），作为 Router 的输入/模块（不新增 action）
  - **gating（结构）**：trend_up + absorption_structure=True（这是少量结构特征）
  - **score（用原语 heads）**：倾向 TREND：
    - `persistence` 高、`mfe_atr` 大、`t_to_mfe` 不需要极小但不能太慢
  - **execution**：复用 TREND 模板
  - **好处**：结构识别和执行模板可复用，新增 pattern 更像“加一个 detector + 一组打分/阈值”，而不是加一个新 action/新执行体系

#### 3.3 Pattern C：先突破一边 → 巨量反转 → 趋势初期反而回归（跟风不足/假突破）

**直觉**：breakout 发生但没有足够跟风/持续性，随后出现量能极端的反转，往往变成“失败突破回归区间”的均值回归机会。

- **规则版**
  - **event**：breakout_bar + follow_through 弱（1–2 根内回到区间）+ reversal_volume_climax
  - **行为**：从 TREND 切到 MEAN（或者直接 veto 不交易）
  - **execution**：MEAN 模板（目标回归到区间中枢/阻力位下方）

- **树模型策略版**
  - 新建策略变体：`failed_breakout_reversion`
  - **标签**：回归到区间的概率/幅度/时间（也可以用类似原语的目标）
  - **特征**：breakout_strength、follow_through、volume_climax、区间宽度、波动状态
  - **优点**：对“真假突破”区分通常更强

- **NN 原语 + Router 版**
  - 新增 Router 子策略：`router_failed_breakout`
  - **gating（结构）**：breakout_event=True 且 follow_through_weak=True
  - **mode 选择**：
    - 若原语显示 `persistence` 低、`mae_atr` 大、`mfe_atr` 不够：切 MEAN 或 NO
    - 若原语显示 `persistence` 仍高：保持 TREND（容错）
  - **execution**：复用 MEAN/TREND 模板（统一口径）

---

### 4. 最推荐的“策略工厂”工作流（让子模型增长但系统不爆炸）

把你不断发现的 pattern，变成**可控的产线**：

- **Step 1：规则原型**（1 天内）
  - 用最少特征实现 gating + execution，先验证“方向是否对、风险口径是否能接受”
- **Step 2：树模型做强对照**（1–3 天）
  - 把 pattern 作为一个策略 variant（或放入对应策略族），用 `feature-group-search` 验证特征有效性
- **Step 3：沉淀为 Router 子策略**（长期维护形态）
  - 把 pattern 压缩成“结构 gating + 原语 heads 解释 + 模板化 execution”
  - 这一步做完，你就能在不增加底座模型数量的情况下，持续增加“子模型（Router 子策略）”

**这就是为什么：**

- **树模型** 更像“并行研究很多独立策略”，很强，但数量会乘法增长
- **NN 原语+Router** 更像“维护一个底座 + 扩展很多 Router 子策略”，更适合长期实盘维护

---

### 5. 结论：到底哪种方式更容易持续产出更多子模型？

- **想快速堆很多想法并且马上可跑**：规则最快，但会很快进入维护地狱
- **想快速验证很多策略语义并做强特征筛选**：树模型是最强的“研究引擎”
- **想长期持续扩展子模型且上线维护成本可控**：**NN 原语 + Router + Execution 最合适**

最终建议落地为一条原则：

> **规则负责“能跑与能回退”，树模型负责“研究与对照”，NN 原语+Router 负责“生产主链路的可持续扩展”。**


