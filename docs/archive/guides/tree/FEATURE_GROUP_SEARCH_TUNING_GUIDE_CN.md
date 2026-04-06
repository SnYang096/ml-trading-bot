# 本文的有一些过时的结论
1. feature-group-search给了特征极大的自由度，全量搜非常慢，实际上最终结果和启发式的接近，有效的还是那些由于一的
2. 这是一个反向验证的过程，证明了启发式的特征，比如sqs，比数学类的特征更加有效，我们应该一层层往上叠加，而不是全局搜索

## Feature-group-search / pipeline 调参指南（CN）

这份文档给出“**怎么把指标（尤其 Sharpe）做上去**”的实操路线，覆盖 tree 策略模型与 nnmultihead（路径原语→Router→Execution）两条体系。

> 目标：把“训练指标变好”升级为“**OOS + 多 symbol 的可交易 Sharpe**”。

---

### 1) 先区分三层指标：Primitives vs Router vs Execution（尤其是 nnmultihead）

#### nnmultihead 三层闭环
- **Primitives（多头）**：`dir_auc / mfe_atr_spearman / roll_icir__*`
  - 作用：判断“模型有没有学到可泛化的结构信息”
- **Router（模式分类/门控）**：mode 分布、switch rate、mode entropy、NO_TRADE 比例
  - 作用：决定“什么时候交易、交易哪种模板”
- **Execution（回测假设）**：`ret_mean/ret_trend` 生成方式、交易成本、滑点、延迟
  - 作用：决定“同样的 mode/信号，真实能赚多少”

**常见误区**：只盯 primitives 指标，忽略 Router 阈值与 Execution 假设，最终 Sharpe 仍然很差。

---

### 2) 推荐调参顺序（从“稳健”到“可交易”）

#### ⭐ 推荐执行顺序（nnmultihead：先 primitives，再 Router，再 BC/RL）

> 这段是“可执行”的最小闭环顺序，用于避免你遇到的典型问题：primitives 近随机时去调 Router/做 BC/RL，只会在噪声上过拟合或学到 NO_TRADE/亏钱策略。

1) **highcap 子集（如 highcap6）先跑 feature search**
   - `mlbot nnmultihead feature-group-search ...`
   - 目标：让 primitives 输入特征更“primitives-friendly”（先把 head 指标拉到非随机）

2) **用 search 的 best config 训练一次 primitives（固定 train window）**
   - `mlbot nnmultihead train ...`
   - 目标：`dir_auc/roll_icir__dir` 等稳定非随机（不是追 Sharpe）

3) **固定 OOS 窗口做 predict（生成 heads/preds）**
   - `mlbot nnmultihead predict ...`

4) **固定 returns 假设（returns_source + 成本/延迟）生成 logs**
   - `mlbot rl build-logs-3action ...`
   - 注意：调参期间不要频繁切换 returns_source，否则 Sharpe 口径不稳定

5) **只对 rule-based Router 做阈值调参（grid/Optuna），目标用 OOS Sharpe**
   - `mlbot rule mode-3action ...`（阈值是调参对象）
   - `mlbot rl run-e2e-3action ...`（Sharpe/score 是验收指标）

6) **Router 过关后，再做 BC/RL（否则只会学到“亏钱/不交易”）**

#### Step A：固定数据口径（不要在调参期间改窗口）
- 训练窗：建议至少覆盖 1.5~2 年（4H）
- OOS：留出最近 4~6 个月（严格 out-of-sample）
- 多 symbol：先 2~3 个主币种（BTC/ETH/SOL），稳定后再扩大

#### Step B：先把 primitives 做到“明显非随机”
参考阈值（经验）：
- `dir_auc`：> 0.56 才值得继续
- `roll_icir__dir`：稳定 > 0.1（越大越好）

如果 primitives 仍然随机：
- 回到特征侧：扩大 PoolB、做更长窗 factor-eval、提升 feature-group-search budget（epochs）

#### Step C：Router 阈值调优（防止 NO_TRADE collapse / 过度交易）
你可以把 `mode-3action` 的阈值当成一个“可调参数组”：
- **偏保守**：交易少、回撤小，但容易错过机会
- **偏激进**：交易多、噪声大，Sharpe 容易变负

推荐做法：
- 在训练窗内做阈值网格搜索（或贝叶斯优化），用 OOS 做验收
- 记录 mode 分布（NO/MEAN/TREND 比例）和 switch rate，避免崩塌到 0% 或 100%

#### Step D：Execution returns 假设（rr_execution / 成本 / 延迟）
Sharpe 很差时，常见根因是 “执行假设不对”：
- `returns_source` 是 `momentum_proxy` 还是 `rr_execution`
- `entry_delay`、`cost_per_turnover`、`slippage_bps` 是否合理
- 是否需要按 symbol/profile 做 rr 参数覆盖（不同币种流动性差异）

**建议**：先在 `cost=0` 下跑出“正的结构性优势”，再逐步加真实成本。

---

### 3) 是否需要 detector 和 gating？（回答：长期一定需要）

nnmultihead 的 Router 要同时满足：
- **泛化**：跨 symbol、跨时间
- **安全**：遇到坏行情能少亏/不交易
- **可扩展**：你不断发现新模式（小样本事件）

这时 **detector/gating 是必须的工程结构**：
- **detector**：把稀有/语义事件“结构识别”出来（例如 wick/absorption/failed_breakout），让 Router 有条件输入
- **gating**：硬约束/风控门槛（低流动性、极端 drawdown、尾部风险、标签缺失等），可直接 veto 交易

它们的价值在于：对“小样本策略”友好 —— 新模式先做 detector + gating 验证，通过后再决定要不要下沉进模型特征。

---

### 4) 具体到命令：建议的最小闭环（nnmultihead）

1) 训练 primitives（train window）
- `mlbot nnmultihead train ...`

2) OOS 推理
- `mlbot nnmultihead predict ...`

3) Router（阈值可调）
- `mlbot rule mode-3action ...`

4) 组装 logs（returns_source + 成本/延迟）
- `mlbot rl build-logs-3action ...`

5) E2E 验收（Sharpe）
- `mlbot rl run-e2e-3action ...`

---

### 5) 何时扩大 symbols / 时间窗？

推荐规则：
- primitives 指标稳定非随机、Router 不 collapse → 扩大 symbols
- OOS Sharpe 在 2~3 个主币种上均不为负 → 扩大到更多币种
- 扩大后如崩溃，回到 detector/gating 与 returns 假设做分层修正

# Feature Search 工具调参指南（Pool‑B + Semantic + Pipeline）

本指南专门面向以下命令的“调参”：

- `mlbot diagnose poolb-semantic-search`（一键：Pool‑B 生成 → feature-group-search → 写回 YAML → 报告）
- `mlbot diagnose feature-group-search`（单策略的组合搜索）

目标：在**可控算力**下，避免“候选过少/误杀协同/结果不稳定”，并能逐步逼近更优的特征组合。

---

## 一、先明确：为什么你会看到“选出来很少”？

你用的是 `--search-algo pipeline`：

1) **Successive Halving（SH）预筛**：先用 **single-add** 的方式快速给候选打分并裁剪
2) **Beam Search**：在 survivors 上找协同组合
3) **SFFS prune**：对最终组合做去冗余删除

因此如果：
- SH 预筛太狠（survivors 太少）
- 或者你启用了 `--expand-semantic-singletons` 导致候选空间暴涨但预算不变

就会出现“最终组合很短”的现象（这不是 bug，而是超参数与候选空间不匹配）。

---

## 二、关键参数速查（按影响顺序）

### 1) 候选空间大小（决定你需要多大预算）

- **`--expand-semantic-singletons`**
  - **作用**：把语义块（多输出列）拆成“单列候选组”
  - **代价**：候选数显著上升（更细，但更慢）
  - **推荐**：
    - 先在 node/group 层跑通 baseline（不开 singletons）
    - 只有在“语义块内部确实存在正负打架”时再开

### 2) SH（预筛）强度：决定 survivors 有多少、是否误杀协同

- **`--pipeline-survivors`**（pipeline 专用）
  - **作用**：最终进入 Beam 的 survivors 目标数量
  - **经验值**：
    - 候选 <= 30：`pipeline-survivors=20~30`
    - 候选 50~150：`pipeline-survivors=40~80`

- **`--halving-top-fraction`**
  - **作用**：每个 halving stage 保留比例
  - **更大**：更不容易误杀协同（但更慢）
  - **经验值**：0.25（快）→ 0.5（稳）→ 0.6（更稳）

- **`--halving-min-survivors`**
  - **作用**：每个 stage 至少保留多少个候选（防止被裁到太小）
  - **经验值**：候选暴涨时建议 >= 30

- **`--halving-stages`**
  - **作用**：预算分级（按 seeds 数量）
  - **推荐**：`1,3,5`（便于快速筛 + 最后用全 seeds 复核）

### 3) Beam：决定能否捕捉“协同组合”

- **`--beam-width`**
  - **作用**：每一步保留 top‑K 条路径（K 越大越能保留“次优但潜在协同”的路径）
  - **经验值**：3（默认）→ 5（更稳）→ 8（更强但更慢）

- **`--max-steps`**
  - **作用**：最多添加多少步（组合长度上限）
  - **经验值**：4~6（通常足够；更大更慢且更易过拟合）

### 4) SFFS prune：去冗余（在“组合已经不错”的前提下）

- **`--sffs-max-backward-per-step`**
  - **作用**：每步最多尝试多少次“删一个组”
  - **经验值**：1~2（先用小值）

---

## 三、三档推荐配置（直接复制）

### A) 快速迭代（小预算）

适合：先摸清大方向、快速看趋势。

```bash
mlbot diagnose poolb-semantic-search \
  --strategies <strategy> \
  --tag <TAG> \
  --symbol BTCUSDT --timeframe 240T \
  --start-date 2024-01-01 --end-date 2025-04-30 \
  --search-algo pipeline \
  --halving-stages 1,3,5 \
  --halving-top-fraction 0.25 \
  --halving-min-survivors 10 \
  --pipeline-survivors 20 \
  --beam-width 3 \
  --max-steps 5 \
  --sffs-max-backward-per-step 1 \
  --regen-poolb --rerun-search \
  --no-docker
```

### B) 推荐默认（中预算，兼顾协同）

适合：候选较多、你不想误杀协同。

```bash
mlbot diagnose poolb-semantic-search \
  --strategies <strategy> \
  --tag <TAG> \
  --symbol BTCUSDT --timeframe 240T \
  --start-date 2024-01-01 --end-date 2025-04-30 \
  --search-algo pipeline \
  --halving-stages 1,3,5 \
  --halving-top-fraction 0.5 \
  --halving-min-survivors 30 \
  --pipeline-survivors 50 \
  --beam-width 5 \
  --max-steps 6 \
  --sffs-max-backward-per-step 2 \
  --regen-poolb --rerun-search \
  --no-docker
```

### C) 更强探索（大预算，适合 singletons）

适合：你明确要做列级别精挑细选（`--expand-semantic-singletons`）。

```bash
mlbot diagnose poolb-semantic-search \
  --strategies <strategy> \
  --tag <TAG> \
  --symbol BTCUSDT --timeframe 240T \
  --start-date 2024-01-01 --end-date 2025-04-30 \
  --search-algo pipeline \
  --expand-semantic-singletons \
  --halving-stages 1,3,5 \
  --halving-top-fraction 0.6 \
  --halving-min-survivors 40 \
  --pipeline-survivors 60 \
  --beam-width 5 \
  --max-steps 6 \
  --sffs-max-backward-per-step 2 \
  --regen-poolb --rerun-search \
  --no-docker
```

---

## 四、常见症状 → 对应调参

### 1) “最终选出来太少 / 不像以前 greedy 那样丰富”

优先调：
- 提高 `--pipeline-survivors`（例如 30→60）
- 提高 `--halving-top-fraction`（0.25→0.5/0.6）
- 提高 `--halving-min-survivors`（10→30/40）
- 提高 `--beam-width`（3→5）

### 2) “Sharpe 不稳定/波动很大”

优先调：
- 增加 seeds（例如 `1,2,3,4,5`）
- 固定 `--deterministic`（默认已开）
- 保持时间段一致（严格使用同一 `start-date/end-date`）

### 3) “跑得太慢”

优先调：
- 先关掉 `--expand-semantic-singletons`
- 降低 `--pipeline-survivors` / `--beam-width`
- 减小 `--max-steps`



