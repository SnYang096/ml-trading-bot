# ml_trading_bot 工作流整体架构与管线改进计划

> 本文档把已有的 `ABC三层收益结构`、`A/B/C 系统*.md`、`regime_layer.md`、`树模型方法论演进*.md`、
> `B系统运维心智梳理.md` 串成**一张图 + 一套时间轴 + 一份落地清单**，
> 并与 ML4T（Stefan Jansen, _Machine Learning for Algorithmic Trading_, 2020/2021）
> 给出的"标准工作流"做对照，明确我们**有什么 / 缺什么 / 哪些是 crypto 特有不该照搬**。
>
> 阅读对象：策略架构师 / 工程师 / 自己未来的复盘。
>
> 与已有文档的关系：本文是 **统揽**，每个具体决策仍以专文为准。
> - 战略动机：[`ABC三层收益结构_战略框架_CN.md`](ABC三层收益结构_战略框架_CN.md)
> - B 系统运维心智：[`B系统运维心智梳理.md`](B系统运维心智梳理.md)
> - 树/ML 方法论：[`树模型方法论演进与短期树重建指南_CN.md`](树模型方法论演进与短期树重建指南_CN.md)
> - regime 层：[`regime_layer.md`](regime_layer.md)
> - **R&D 执行手册（quick_layer_scan → event_backtest → watchdog）**：[`方法论_R_and_D流程_CN.md`](方法论_R_and_D流程_CN.md) ← 把本文档的 §4 / §5 压成一条可重复流程

---

## 0. 摘要 — 一张图

```text
              ┌───────────────────────────────────────────────────────────┐
              │  时间常数分轴（这是整套架构最重要的一根脊柱）              │
              └───────────────────────────────────────────────────────────┘
                       Q-级（季度，按需）        M-级（月度，自动）
                  ┌──────────────────┐       ┌──────────────────┐
                  │   R&D 假设生成    │ ───►  │  纯 cross-validation │ ◄── W-级
                  │   ──────────────  │       │  ──────────────────  │   regime
                  │ • offline scan    │ 改 yaml │ • 跑现 config        │   watchdog
                  │ • event_backtest  │       │ • drift 报告          │   ┌────────┐
                  │ • 人工 review     │ ◄──告警 │ • 不改 yaml          │   │ 阈值监控 │
                  │ • commit yaml     │       │                      │   │ 不改 yaml │
                  └──────────────────┘       └──────────────────────┘   └────────┘
                          │                                                  │
                          ▼                                                  │
                  ┌──────────────────────────────────────────────────────────┘
                  │  上线门禁（任何 yaml change → validate_static.full_study）
                  │  require_human_confirm → deploy_config_to_live
                  └──────────────────────────────────────────────────────────


              ┌───────────────────────────────────────────────────────────┐
              │  ABC × 五层职责矩阵                                        │
              └───────────────────────────────────────────────────────────┘
                       Regime    Prefilter   Direction   Gate      Entry     Execution
   A (spot accum)      ✅ 周EMA   —            买/卖     —          —        慢出场
   B (BPC/TPC/ME/SRB)  ✅ EMA1200 ✅ 形态语义  ✅ 公式   ⚠ tail veto ⚠ 择时   规则 SL/TP
   C (chop/trend_scalp) ✅ chop   ✅ 分流       ✅ regime →   ⚠ 软协调   —    多腿 fee-aware
                                            子策略
```

- **白盒**（规则 YAML，locked，季度看一次）：Prefilter、Direction、Execution、PCM 仲裁。
- **灰盒**（统计标定 / 小帽子树，月度看 drift，季度才重训）：Regime、Gate、Entry。
- **完全静态**（年度才动）：A 的周线 EMA200 死区、宪法 failure 硬约束。

---

## 1. 现状诊断 — 你之前管线为什么又慢又飘

`docs/strategy/B 系统运维心智梳理.md` 已经讲了 **正确的运维哲学**：

> 入场层定稿后不再用树模型优化。Gate / entry filter 统计做**一次**，固定。
> 唯一建议定期更新：regime（EMA1200、chop/box 等慢变量）。

但 **代码里实际跑的 `config/strategies/<策略>/research/calibrate_roll.default.yaml`** 与这个哲学相反：

```yaml
threshold_calibration:
  prefilter:        { optimize: true }
  gate:             { optimize: true }   # ← doctrine 说"统计做一次"
  entry_filter:     { optimize: true }   # ← doctrine 说"统计做一次"
  direction_tuning: { enabled: true }    # ← doctrine 说"几乎不动"
  execution_opt:    { enabled: true }    # ← doctrine 说"几乎不动"
```

`research_roll.features_on.yaml` 再加 `shap_feature_selection.enabled: true` + `enable_model_training: true`，
等于每季度做一次 SHAP 特征发现并自动 promote 回 features.yaml。

**5 个结构性问题：**

| # | 问题 | 现象 | 根因 |
|---|------|------|------|
| 1 | doctrine 与代码不一致 | locked 规则旁阈值月月动，长期被"挤"出 plateau 中心 | turbo 默认所有层 optimize |
| 2 | R&D 与运维塞同一管子 | 改任何东西都要跑 1-2h，迭代慢 | 没有"快速 ablation"入口 |
| 3 | walk-forward 同时调参 + 验证 | 每月调出"最优"，下月 OOS 一般 | 经典 walk-forward overfit |
| 4 | SHAP 当特征选择器用 | 季度推一批新特征 → features 自己抖 | importance ≠ causal contribution |
| 5 | 层间隐性耦合 | 同段历史先后训 prefilter/gate/entry，互相 over-fit 到同一噪声 | 多层级联未做正交化 |

**所有这些都不是 bug，是"管线目标定义"的错位**：当前管线被设计成"自动 R&D 装置"，但 doctrine 要求的是"自动**验证**装置"。

---

## 2. ABC 系统 × 五层职责矩阵

> 五层定义来自 `B系统不变的层.md` + `regime_layer.md`，与代码 `config/strategies/<slug>/archetypes/*.yaml` 一一对应。

### 2.1 五层职能（统一接口）

```text
原始 bar / 订单流 / 衍生品流 / 链上
        │
        ▼  特征工程（features.yaml + features_prefilter.yaml + features_gate.yaml）
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ① Regime    "今天能不能做"  — 慢变量门控，宏观语义                    │
│   档案: archetypes/regime.yaml                                       │
│   慢变量: EMA1200_position、EMA1200_slope_10、chop、box_pos_1200      │
│   2025 现状: TPC 用 |ema_1200_position|>0.10；BPC/ME/SRB chop≤0.40    │
├──────────────────────────────────────────────────────────────────────┤
│ ② Prefilter "今天该做哪一仗" — archetype 语义闸                       │
│   档案: archetypes/prefilter.yaml                                    │
│   规则: BPC=突破后回踩、TPC=趋势内深回调、ME=动量延续、SRB=结构反扑     │
│   原则: locked，季度才审，不让 SHAP 改                                │
├──────────────────────────────────────────────────────────────────────┤
│ ③ Direction "这一笔做多还是做空" — 符号公式                           │
│   档案: archetypes/direction.yaml                                    │
│   原则: 与叙事绑定，几乎不变（TPC=macd_atr 符号；BPC=突破方向）         │
├──────────────────────────────────────────────────────────────────────┤
│ ④ Gate      "尾部风险否决" — 该不该 veto                              │
│   档案: archetypes/gate.yaml                                         │
│   原则: 仅留经离线 t-test + plateau 双重验证的硬 deny；               │
│         统计 lift 高但 trade R 实测负效应的规则要去掉                 │
├──────────────────────────────────────────────────────────────────────┤
│ ⑤ Entry    "这一根 bar 入不入" — 择时（订单流 / 结构）                │
│   档案: archetypes/entry_filters.yaml                                │
│   原则: OR 关系；少而精；统计做一次后固定                              │
├──────────────────────────────────────────────────────────────────────┤
│ ⑥ Execution "怎么进 / 怎么出" — SL/TP/trail/网格/加仓                 │
│   档案: archetypes/execution.yaml + multi_leg/*.yaml                 │
│   原则: 与 payoff hypothesis 绑定，几乎不动                            │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼  TradeIntent → PCM 仲裁（多策略共账户预算） → multi-leg 执行
```

### 2.2 各系统的"哪些层活、哪些层冻"

| 层 | A (spot_accum) | B (BPC/TPC/ME/SRB) | C (chop_grid + trend_scalp) |
|---|---|---|---|
| **Regime** | 周 EMA200 死区（10 年一更）| EMA1200 dead zone + chop（季度看）| `semantic_chop` 0.50/0.32 双阈（半年看）|
| **Prefilter** | n/a（A 没有形态层）| 4 archetype 形态（locked）| 路由：chop → chop_grid，expansion → trend_scalp |
| **Direction** | 周 EMA200 上/下（粗）| 公式（locked）| regime 强势侧 |
| **Gate** | n/a | tail veto（chop）| n/a |
| **Entry** | n/a | OR rules（locked，本轮我们重选）| n/a |
| **Execution** | 越涨越快卖出 | 紧 SL + 较快兑现 | 网格 spacing / fee-aware TP |
| **维护频率** | **几乎不动** | **季度审 regime + gate** | **半年审 regime 阈值** |
| **复杂度来源** | 单参数：卖出 5× 阈值 | 4 策略 × 6 层 | 多腿引擎 |
| **能否用 ML** | 否（哲学不允许）| Gate / Entry 可用 small tree（"小帽子"，§3.3）| 否（多腿无 ML 可言）|

**关键认知**：A、C **本质上不该有"特征发现"工作流**。只有 B 才需要 R&D 管线，且 R&D 仅作用于 **regime + gate + entry** 三层。

---

## 3. 每层"特征发现 + 阈值优化"的算法分工

> 算法 != 自动 promote。下面每条算法的产物都是 **markdown 候选报告**，由人审后写入 archetypes。

### 3.1 Regime 层（慢变量，单变量 plateau 法）

**输入**：`features_labeled.parquet` 中的慢变量列（`ema_1200_*`, `chop`, `box_*`, `atr_percentile` 等）。

**算法**：
1. **单变量阈值 plateau 扫描**（`scripts/regime_threshold_calibrate.py --dry-run`）
   - 在阈值候选网格上算 effect / p / pass_rate / Spearman 单调性
   - 找 `_identify_plateau()` 返回的 `PlateauRange(start, end, mid)`，要求 plateau 宽度 ≥ 5 个候选点、内部 effect σ < 0.5%
2. **桶诊断**（`scripts/regime_ablation_report.py`）
   - 按 `ema_1200_position` 切 bull/bear/neutral 三桶
   - 每桶单独算 chop / box / atr 的 effect → 找**桶间分歧**的特征（如本次 TPC 发现 chop 在 ema_bull 是 +3%，ema_bear 是 -2%）
3. **多变量 AND 验证**（不做新组合搜索，仅验证候选）
   - 比如 `|ema_1200_pos|>0.10 AND |slope_10|>0.002` 是候选时，
     评估比单独 0.10 阈值更强：是否 effect ↑ ≥ +1% AND p < 0.01 AND pass_rate ≥ 25%

**不要做**：树模型自动发现 regime（regime 是宏观语义，不是统计构造）。

**频率**：季度运行 / 漂移触发。

### 3.2 Prefilter 层（archetype 语义，几乎不动）

**输入**：archetype 命名的特征（`tpc_pullback_depth`, `bpc_pre_breakout_score`, `me_score_continuation`）。

**算法**：仅做 **plateau 复核**（不发现新特征）：
1. 在最新数据上验证现有阈值仍在 plateau 中心
2. 漂移 > 阈值时**告警** → 转到 §3.1 由人决定是否调

**不要做**：SHAP 推荐新 prefilter 特征。Prefilter 是 archetype 叙事，**不接受统计驱动的特征替换**。

**频率**：半年 / 漂移触发。

### 3.3 Gate 层（尾部风险否决，小帽子树 + 单 τ）

**输入**：full feature set，包括 `vol_persistence`, `vol_leverage_asymmetry`, `evt_var_99`, `tpc_semantic_chop` 等。

**算法**（按 `树模型方法论演进*.md` §1.2 的"小帽子树"路径）：
1. **单规则 ablation**（first）
   - 对每条现有 gate 规则，在 holdout 上看 effect on `pnl_r`（不是 label）→ 负 effect 的规则**删掉**
   - 本次 TPC 验证：`vol_persistence` deny、`vol_lev_asymmetry` deny 都是负 effect → 删
2. **新候选搜索**（second，仅离线 t-test）
   - 在 post-prefilter+regime 子样本上扫所有数值特征 → top-20 by effect+p
   - 候选必须能被 1 个阈值表达（如 `evt_var_99 >= 0.869`），不接受复杂组合
3. **浅树 gate（可选，§3.3.2）**
   - 训 1 棵深度 ≤ 3 的 tree → 输出 `gate_score`
   - 在 holdout 上对 `gate_score` 做**单维 plateau 扫描** → 得 `τ`
   - 与现有"硬 deny 规则"并存运行 shadow 期（不立即替换）

**KPI**（必须同时满足才上）：
- holdout effect ≥ +1.5% on `pnl_r`
- p < 0.01
- pass rate ≥ 30%
- bull / bear / neutral 三桶分别都不为负

**频率**：季度。

### 3.4 Entry 层（择时，order flow / 结构）

**输入**：订单流特征（`vpin_*`, `cvd_*`, `wpt_*`, `vp_absorption_score`）+ 结构特征（`box_compression_score`）。

**算法**：
1. **post-gate 子样本上 t-test 扫描**（`scripts/quick_layer_scan.py`，**下一步要建**）
   - 子样本：pullback ∧ regime ∧ chop_pass（n ≈ 7500 on TPC）
   - 输出每特征的 effect / p / pass_rate / 桶分布
2. **候选 OR-rule**（少而精）
   - 选 top-3 by effect AND p < 0.01 AND pass ≥ 30%
   - 写成 OR 规则放到 `entry_filters.yaml`
3. **回测验证**：在 event_backtest 1y 上比较新旧 OR 集

**不要做**：树模型 entry（除非按 §3.3.2 当"score 头"），保留 yaml 可读性。

**频率**：季度。

### 3.5 Execution 层（不动）

**只在**以下情况动：
- payoff hypothesis 改变（如从 swing 改成 scalp）
- 费率结构变化（交易所改 fee）
- 极端市场结构变化（流动性塌方）

**算法**：execution_opt grid（保留管线里），但**不自动 promote**。

**频率**：年度 / 触发式。

### 3.6 算法总表

| 层 | 主算法 | 辅助算法 | promote 方式 | 频率 |
|---|---|---|---|---|
| Regime | 单变量 plateau 扫描 | 桶诊断 (ablation_report) | 人审 → yaml | 季度 |
| Prefilter | plateau 复核 | drift detection | **不变 / 人审** | 半年 |
| Direction | — | — | **完全冻结** | 年度 |
| Gate | 单规则 ablation + t-test top-k | 浅树 score + τ plateau | 人审 → yaml | 季度 |
| Entry | 子样本 t-test + OR rule | 浅树 score（可选）| 人审 → yaml | 季度 |
| Execution | — | execution_opt grid（手工触发）| **完全冻结** | 年度 |

---

## 4. 时间常数分轴 — D/W/M/Q 级工作流

> 这是 §0 那张图的展开。**任何 R&D / 运维操作都必须能挂到一个时间轴上**，否则它就是"管线杂质"。

### 4.1 D 级（日 / 异常触发）

**任务**：单笔事后复盘 / 异常诊断

**触发**：
- live 出现意外亏损
- 月度报告里某 symbol 命中率突跌
- 用户怀疑某层逻辑

**工具**：
- `scripts/event_backtest.py --start-date X --end-date Y` 局部窗口
- `scripts/quick_layer_scan.py`（待建）对该窗口做单层 ablation
- 人工读 `event_trades_*.csv`，按 feature 切片

**不允许的事**：碰主 yaml；动 features.yaml。

### 4.2 W 级（周）

**任务**：regime 阈值监控（唯一定期"看阈值"的口子）

**工具**：`scripts/regime_watchdog.py`（待建）— 每周一次，做：
1. 拉最新 `features_labeled.parquet`
2. 对每个慢变量阈值，重做 plateau 扫描
3. 若 `|new_plateau.mid - current_yaml_value| / current_yaml_value > 5%` → 写 `docs/decisions/regime_drift_<日期>.md` + Slack alert
4. **不改 yaml**

**为何只 watchdog 不 auto-adjust**：
- regime 是策略"开关"，错一次全盘皆输（`B系统不变的层.md`）
- doctrine 明确要求人工 confirm
- 5% 漂移可能是季节性，自动调反而引入噪声

### 4.3 M 级（月）

**任务**：纯 cross-validation — **用现有 config 跑过去 12 个月**

**改造**：把 `bpc/research/calibrate_roll.default.yaml`（及 TPC/ME/SRB 对应版本）改成：

```yaml
threshold_calibration:
  prefilter:        { optimize: false }   # ← 改
  gate:             { optimize: false }   # ← 改
  entry_filter:     { optimize: false }   # ← 改
  direction_tuning: { enabled: false }    # ← 改
  execution_opt:    { enabled: false }    # ← 改
  # 保留 locked_threshold_tuning，但 search_mode 改成 dry_run 模式（只算 plateau drift，不写回）
```

**产物**（写到 `results/<strat>/monthly_validation/<日期>/`）：
1. 月度 sharpe / dd / win / pnl_r 分布
2. 每月 trade 漏斗（regime/gate/entry reject 比例）
3. **drift 报告**：plateau 是否已偏移现有阈值
4. **不**改 yaml；只 alert

**为何要这层**：
- 跑得快（不 optimize）→ 月度成本低
- 对**已上线**配置做 holdout 监控
- 提供"配置是否还有效"的客观证据

### 4.4 Q 级（季）

**任务**：R&D 假设生成 + 验证

**工具链**：
1. `scripts/quick_layer_scan.py`（待建）
   - 输入：strategy + features_labeled.parquet
   - 输出：每层 top-20 候选 + 桶分布 + plateau 候选阈值
   - 跑 1-2 分钟，纯离线 t-test
2. **手工** cp config 到 `config_experiments/<variant>/strategies/`
3. `scripts/event_backtest.py --strategies-root ...` 跑 1y 验证
4. 比较 totR / win / maxDD / Ret-DD
5. **人审**赢家 → 写入 `config/strategies/<strat>/archetypes/*.yaml`
6. → 触发上线门禁层

**这层产物示例**（即本次 TPC 实验）：
- variant B (gate_only_chop) → +59.83R / -5.23% DD / 4.17 ret/dd
- 离线 t-test 提前预测了"vol gates 在 bear 子样本上负效应"

### 4.5 上线门禁层

**触发**：任何 `config/strategies/*/archetypes/*.yaml` 改动。

**工具**：`config/strategies/<strat>/research/validate_static.full_study.yaml`（已有）

**产物**：`results/<strat>/validate_static.full_study/<日期>/report.json`
- 整段历史（2022-01 → now）评分
- `deploy_gate` 字段：`require_adopt`, `trigger_sharpe_improve`, `trigger_drift_level`
- `require_human_confirm: true`

**`scripts/deploy_config_to_live.py` 必须加 hard check**：
- 上一次 `validate_static` 必须 < 7 天
- `deploy_gate.adopt == true`
- 否则拒绝 deploy

### 4.6 时间常数总表

| 时间轴 | 任务 | 触发 | 工具 | 输出 | 改 yaml？ |
|---|---|---|---|---|---|
| D | 单笔/异常复盘 | 异常 | event_backtest 局部 | 内部 markdown | ❌ |
| W | regime 漂移监控 | 周 cron | regime_watchdog | alert + decisions/*.md | ❌ |
| M | cross-validation | 月 cron | turbo (改为纯验证) | drift report | ❌ |
| Q | R&D 假设生成+验证 | 季度 / 漂移 | quick_layer_scan + event_backtest | 候选 archetypes | ✅（人审后）|
| 触发 | 上线门禁 | yaml change | validate_static + deploy | report + live | live deploy |

---

## 5. 管线职责重定义 — turbo / slow / non_rolling

| 管线 | 当前职责 | **改成** | 原因 |
|---|---|---|---|
| `calibrate_roll.default.yaml` (turbo) | 月度全层 optimize | **纯 cross-validation**：不 optimize 任何层；只产 monthly drift report | doctrine 要求 locked rules 不动 |
| `research_roll.features_on.yaml` (slow) | 季度 SHAP + 全层 retrain | **季度"特征体检"**：SHAP 仅写到 `results/shap_audit/<日期>.md`，**不 auto-promote** | SHAP importance ≠ causal contribution |
| `validate_static.full_study.yaml` (non_rolling) | 全周期评分 + deploy_gate | **保留不变** — 这是上线门禁，已经是正确职责 | 唯一应"严守"的管线 |
| **新增** `regime_watchdog.py` | — | 周度 regime 阈值监控；只产 alert，不改 yaml | doctrine: regime 定期检查 |
| **新增** `quick_layer_scan.py` | — | R&D 第一步：1-2 分钟离线扫所有层 top-k 候选 | 加速 R&D 迭代 |

**禁止做的事**：

1. **任何管线 auto-promote 进 archetypes/*.yaml** — 都必须经人审
2. **rolling 时同时调参 + 验证** — walk-forward overfit 陷阱
3. **SHAP 推荐直接进 features.yaml** — 留作"诊断证据"由人审
4. **turbo 月度跑 prefilter optimize** — locked 规则旁阈值漂移
5. **execution_opt grid 自动 promote** — execution 应几乎不动

---

## 6. 漂移检测与告警

> 漂移检测不是"自动修复"，是"早期警报"。任何漂移信号都应 **alert → 人审 → 决定是否 R&D**。

### 6.1 三层漂移指标

| 层级 | 漂移指标 | 算法 | 阈值 | 触发动作 |
|---|---|---|---|---|
| 数据 | feature distribution shift | PSI (Population Stability Index) 月对月 | PSI > 0.25 | 数据团队检查 source |
| 模型/规则 | 单特征 IC@H 衰减 | rolling 3-month IC vs 全周期 IC | abs(rolling - all) > 0.05 | feature health alert |
| 配置 | plateau drift | 现有阈值与新 plateau 中心距离 | rel diff > 5% | regime alert (写 decisions/*.md) |
| 业绩 | strategy sharpe drift | 滚动 3-month sharpe vs 12-month | drop > 0.5 σ | 触发 R&D 季度复盘 |

### 6.2 已有的 / 待建的检测

| 检测 | 现状 | 待做 |
|---|---|---|
| PSI / feature drift | ❌ 没有 | 加到 `scripts/regime_watchdog.py` 月度模块 |
| IC@H 时序追踪 | ✅ `mlbot ts-factor-eval` 已有 IC | 加 cron + alert |
| plateau drift | ⚠ 部分 — `prefilter_drift_guard` 在 slow 里 | 抽出做独立 watchdog；改成"alert only"模式 |
| strategy sharpe drift | ✅ `rolling_dashboard` 有 dashboard | 加 alert 阈值 |

### 6.3 不要做的检测

- **每周用 SHAP 重排 feature importance** — importance 不稳定不代表 feature 失效
- **每天看 monthly sharpe** — 噪声主导，会引发过度反应
- **gate hit rate 单层监控** — 必须配合下游 trade R 一起看才有意义

---

## 7. 与 ML4T (Stefan Jansen) 工作流对照

> ML4T 第 1 章给的"标准工作流"如附图（Data Sources → P-I-T → Feature Engineering → ML Models → Predictions → Asset Selection → Portfolio Optimizer → Orders → Execution → Live → Monitoring）。
> 我们逐项对照，找出 **你已有的、缺的、和不该照搬的**。

### 7.1 你已有且较强的部分

| ML4T 模块 | 你的实现 | 比 ML4T 强在哪 |
|---|---|---|
| Data Sources | parquet 化的 OHLCV + 衍生品（funding/OI）+ 订单流（aggTrades）| 数据粒度细到 1m，订单流原生 |
| Factor & Feature Engineering | 357 特征 + 语义分组（features_*.yaml）| crypto 微观结构特征（VPIN, CVD, wavelet, EVT）远超标准 ML4T 因子库 |
| Predictions | LightGBM `predict_proba` + 多 horizon | 已与 archetype 解耦，可单独运行 |
| Execution | 1m-bar 重放 + 多腿引擎 | ML4T 主要讲 daily portfolio，多腿是你独有 |
| Live → Backtest 同源 | `live/highcap/config/strategies/` 与研究版同 schema | walk-forward 与 live 强一致 |

### 7.2 你与 ML4T 有但**实现弱**的部分

| ML4T 模块 | 你的现状 | 改进建议 |
|---|---|---|
| **Point-in-Time** | ✅ 隐式（parquet 月度切片）| 加 PIT contract checker：`assert df.index.max() < snapshot_ts` 在每个 stage 入口 |
| **Cross-Validation** | ⚠ 简单 rolling | 升级到 **Combinatorial Purged CV**（ML4T Ch.7 / Lopez de Prado）— 处理时间序列样本重叠 |
| **Bet Sizing** | ⚠ 固定 `risk_per_r = 1%` | 加 **Kelly fraction**（capped at 25%）+ vol-target 缩放 |
| **Portfolio Optimizer** | ⚠ PCM 是"仲裁器"而非"优化器"| 在 PCM 层加 **risk parity / max-sharpe** 选 sub-strategy 配比；当前 ABC 配比是手工设定 |
| **Performance Attribution** | ⚠ 按策略/symbol 分账，无 regime/feature 归因 | 加 **regime × strategy 二维归因表**（季度复盘看哪类 regime 哪个策略赚钱）|
| **Risk Management** | ✅ 宪法 failure 是 circuit breaker，但不是 risk model | 加 **portfolio VaR / ES** 估算（虽然单笔有 evt_var_99 但组合层没有）|

### 7.3 你**缺**的 ML4T / Lopez de Prado 工具（按价值排序）

| 工具 | ML4T 章节 | 你为什么需要 | 建议优先级 |
|---|---|---|---|
| **Triple-Barrier Labeling** | Ch.17 (López de Prado) | 当前 `success_no_rr_extreme` 是固定 horizon；triple-barrier (TP/SL/Time 三选一最先触) 更贴近实盘退出语义 | ⭐⭐⭐ |
| **Meta-Labeling** | Ch.17 | 在主信号之上加一个二级模型决定"要不要拿"。可作为 entry filter 的小帽子树 | ⭐⭐⭐ |
| **Sample Weights by Uniqueness** | Ch.4 | 你的 bar 标签会重叠（一个 trade 跨多个 bar），不加 unique weight 会让模型 over-fit 高重叠区间 | ⭐⭐ |
| **Fractional Differentiation** | Ch.5 | 价格类特征要么 stationary（diff，丢记忆）要么 non-stationary（raw，bad for tree）；frac-diff 兼顾 | ⭐⭐ |
| **Combinatorial Purged CV** | Ch.7 | 替代当前 rolling，缓解 walk-forward overfit | ⭐⭐ |
| **Bet Sizing (Kelly)** | Ch.10 | 当前固定 risk_per_r 是"硬上限"，没有按信号置信度缩放 | ⭐⭐ |
| **Trend-Scanning Labels** | Ch.5 | 用 t-value 自动找 trend 段而不是手工设 horizon | ⭐ |
| **Hierarchical Risk Parity** | Ch.13 | 当 ABC 子策略数 > 6 时做组合权重 | ⭐ |

### 7.4 你**不应该照搬** ML4T 的部分（crypto 与 equity 差异）

| ML4T 模块 | ML4T 假设 | 为什么 crypto 不该照搬 |
|---|---|---|
| Quantopian 风格 cross-sectional | 1000+ 股票池，按市值/行业切片 | crypto 高 cap 实质只有 5-10 个标的；横截面 alpha 弱 |
| Sector / Style Factors | Fama-French 因子库 | crypto 没有等价 sector，且情绪/链上数据才是主因 |
| Alphalens / pyfolio 套件 | 假设日频 / cross-sectional return | 你是 4H / 2H 单标的方向策略，alphalens 不太用得上 |
| Mean-Variance Portfolio Opt | 假设资产收益正态、可分散 | crypto 高度相关（同涨同跌），MV 给的权重退化 |
| News / 情感分析 | 标准 NLP pipeline | crypto 链上数据 / OI / funding 比 news 更直接 |

### 7.5 一句话总结对照

- **数据 + 特征**：你比 ML4T 标准玩家**强**（订单流 + 衍生品 + 链上 + 微观结构）
- **建模 + CV**：你比 ML4T **弱**（CV 没 purged，bet sizing 简化）
- **执行 + 多腿**：你**独有**（ML4T 不讲）
- **风险管理 + 归因**：你比 ML4T **弱**（宪法是 hard caps，没有 portfolio VaR / 归因）

---

## 8. 落地路线图（5 个里程碑）

按优先级排，每个 ≤ 1 周工作量：

### M1：改 `calibrate_roll.default.yaml` 为纯验证模式（最高优先级）

**做什么**：4 个策略（BPC/TPC/ME/SRB）的 `calibrate_roll.default.yaml` 都改成：
- 所有 layer optimize = false
- locked_threshold_tuning.search_mode = `dry_run`
- 保留 event_backtest 输出，加 drift report stage

**预期效果**：
- 月度跑时间从 1-2h 降到 ≤ 30min
- locked rules 周边阈值不再漂移
- 月度产物清晰：sharpe/dd 趋势 + drift alert

**风险**：可能错过某些"该自动调"的微调；但 doctrine 与本次 TPC 实验都说明这些"自动调"通常是 noise。

### M2：建 `scripts/regime_watchdog.py`（已落地，20260526）

**已实现**（`scripts/regime_watchdog.py`）：周度 cron，针对 TPC bull-conditional gate（variant H）健康度做三件事：

1. 当前窗口 `ema_1200_position` 分布（p25/p50/p75/p90/bull_share/bear_share）；
2. 每条 `ema_1200_position>0.10` 条件 gate（vp_bull_only / vla_bull_only）的实际触发率；
3. 与 `config/monitoring/regime_watchdog_baseline.json` 比较，bull_share 漂移 > 10pp 或 trigger_rate 相对漂移 > 50% 即 ALERT。

**已落地的 baseline**（来自 `train_final_20260523_122438_rr_extreme`）：

```json
{"tpc": {
  "bull_share": 0.169,
  "trigger_rates": {
    "gate_vol_persistence_vol_persistence_bull_only": 0.0345,
    "gate_tpc_vol_leverage_asymmetry_mid_bull_only": 0.0537
  }
}}
```

**示例用法**：

```bash
python scripts/regime_watchdog.py \
  --strategies tpc \
  --window-parquet results/<recent>/features.parquet \
  --baseline-json config/monitoring/regime_watchdog_baseline.json
# exit=1 表示 ALERT；写入 results/regime_watchdog/<ts>/report.json + summary.txt
```

**搭配 `regime_drift_monitor.py`**：后者监控所有策略 regime.yaml 的 plateau 漂移（D-层），前者专门盯 TPC variant H 的 bull-conditional gate（W-层）。两者形成"慢变量分布漂移 + gate 触发率漂移"双重监控。

**预期效果**：唯一定期"看阈值"的口子；其他层全部冻结。

### M3：建 `scripts/quick_layer_scan.py`

**做什么**：封装本次 TPC 实验用的 label scan + 桶诊断：

```bash
python scripts/quick_layer_scan.py \
  --strategy tpc \
  --labels success_no_rr_extreme \
  --layers regime,prefilter,gate,entry \
  --output results/<strat>/quick_scan/<日期>.md
```

输出每层 top-20 候选特征的 effect / p / pass_rate / 桶分布 + plateau 候选阈值。**markdown 报告，不写 yaml**。

**预期效果**：R&D 第一步从"跑一次 turbo (1-2h)"变成"扫一次 quick_scan (1-2min)"。

### M4：把 `research_roll.features_on.yaml` 的 SHAP 改成 audit-only

**做什么**：
- SHAP 结果只写到 `results/shap_audit/<日期>.md`
- 不再自动 promote 进 `features.yaml`
- 季度跑，与 quick_layer_scan 对照看

**预期效果**：features.yaml 稳定，特征发现回到人审节奏。

### M5：实施一个"小帽子树" pilot（TPC gate 或 entry，二选一）

**做什么**：按 `树模型方法论演进*.md` §2 路径：
1. 固定特征池（用 quick_layer_scan top-20）
2. 训 LightGBM depth ≤ 3 输出 `gate_score`
3. 在 holdout 上对 score 做单维 plateau → 得 τ
4. shadow 期与现有规则并行

**预期效果**：验证 §1.2 "决策层 ML + 执行层规则" 是否可行；不可行就回到纯规则栈。

### 顺序与依赖

```
M1 (改 turbo 验证模式) ──► M2 (regime_watchdog)
       │                          │
       ├─► M3 (quick_layer_scan) ─┤
       │                          ▼
       ▼                    M5 (小帽子树 pilot)
   M4 (SHAP audit)
```

**M1 + M3 是最关键的两件事**：能直接让你 R&D 速度提 10×、运维稳定性提一档。

---

## 附录 A：当前管线代码层面的具体问题清单

| # | 文件 | 行为 | 问题 | 修复 |
|---|---|---|---|---|
| 1 | `bpc/research/calibrate_roll.default.yaml:39-49` | `optimize: true × 5` | 月月动 locked rules 旁阈值 | 改 false |
| 2 | `research_roll.features_on.yaml:127-138` | shap_feature_selection auto-promote | 特征自漂移 | 改 audit-only |
| 3 | `auto_research_pipeline.py:_run_pre_deploy_contract_checks_if_configured` | adopt 触发自动 cp 实验 archetypes 回生产 | 跳过人审 | 加 `require_human_confirm` gate |
| 4 | `pipeline/multileg_feature_selection.py` (929 行) | 多层级联 SHAP+plateau | 层间过拟合到同一段噪声 | 分层独立 holdout，不共享 |
| 5 | `regime_threshold_calibrate.py --dry-run` | 已存在但默认行为是 commit | 默认就该是 dry | 改默认值 |
| 6 | `prefilter_drift_guard` (in slow yaml) | 漂移阈值 0.20 warn / 0.35 max | 太宽 | 改 0.05 warn / 0.10 max + alert |
| 7 | `validate_static.full_study.yaml deploy_gate` | `require_human_confirm: true` 已对，但没有强制 | `deploy_config_to_live.py` 应 hard-check | 加 7-day 时效 + adopt=true 双门 |
| 8 | 缺失 | 没有 portfolio-level VaR/ES | 单笔有 evt_var_99，组合层无 | 加 portfolio risk module |
| 9 | 缺失 | 没有 regime × strategy 归因表 | 季度复盘没有数据基础 | 加 attribution dashboard stage |
| 10 | 缺失 | 没有 meta-labeling 框架 | entry filter 是手工 OR rules，没有"在主信号上加二级判断"的入口 | M5 pilot |

---

## 附录 B：与 docs/strategy 已有文档的关系

| 已有文档 | 本文如何引用 / 不冲突 |
|---|---|
| `ABC三层收益结构_战略框架_CN.md` | 本文 §2 直接采纳其 ABC 分工，扩展为五层矩阵 |
| `A 系统不变的层.md` | 本文 §2.2 A 列"几乎不动" = 复述其结论 |
| `B 系统运维心智梳理.md` | 本文 §1（诊断）+ §3（算法分工）= 把它的运维结论落到代码层 |
| `B 系统不变的层.md` | §2 五层维护频率表 = 它的优先级表的代码映射 |
| `C 系统运维心智梳理.md` | 本文 §2.2 C 列"半年审 chop 阈值" = 它的结论 |
| `regime_layer.md` | §3.1 regime 算法 = 它的工具列表 |
| `树模型方法论演进与短期树重建指南_CN.md` | §3.3.2、§5 小帽子树 = 它 §1.2 的具体落地 |
| `B 系统 pcm 如何防止超过宪法预算.md` | §2 PCM 仲裁 / 附录 A #8 portfolio VaR = 它的延伸 |

**本文与上述文档的分工**：
- 上述文档讲**单点**（A 系统怎么动 / B 系统不变的层 / 树往哪放）
- 本文讲**整体**（怎么把它们串成一套时间常数分明的工作流 + 与 ML4T 对照）
- 任何冲突时，**单点文档优先**（它们经过实战验证；本文是 v1 综合）

---

## 附录 C：术语表（与 ML4T / Lopez de Prado 对照）

| 中文 | 本仓库术语 | ML4T 等价 | Lopez de Prado |
|---|---|---|---|
| 死区过滤 | regime dead zone | filter / signal generation | meta-rule |
| 形态闸 | prefilter | factor screen | primary model |
| 风险否决 | gate | risk filter | secondary filter |
| 择时滤波 | entry filter | timing filter | meta-labeling candidate |
| 平坦高原 | plateau | threshold stability | parameter robustness region |
| 漂移 | drift | concept drift / regime shift | non-stationarity |
| 仲裁 | PCM | portfolio constraints | exposure cap |
| 多腿 | multi-leg | order book / iceberg | execution algo |

---

## 结语

**doctrine → 时间常数 → 算法分工 → 管线职责 → drift 检测 → ML4T 缺口**
这条链子的目的：让"维护 4 策略 × 6 层"这件事，从"每月跑一遍管线、不知道哪里又飘了"变成"每周自动看 1 个数、每季度做 1 次手工 R&D、每年看 1 次结构"。

最关键的一句：**管线只做验证，不做研究**。研究是人脑 + 离线 scan 在 Q-级做的事。把这两件事强行融在一起，是过去两年最大的运维负担。

