## EXP_007: NN Multihead（Path Primitives）统一评估协议（Head / System / Execution）

目标：把你现在“看回归指标没感觉、看 Router Sharpe 又受 execution 假设影响”的困惑，变成一套**可复盘、可对齐口径**的评估流程。以后每次训练/调参，都按同一套口径出报告与结论。

这份协议把评估拆成三层（**必须同时做，但职责不同**）：
- **A. Head 评估（解耦标签）**：尽量不依赖 execution，回答“head 有没有信息量/排序能力/稳定性”
- **B. System 评估（固定 execution）**：固定一版 execution 假设，回答“Router+策略系统在这个执行假设下能否赚钱/稳健”
- **C. Execution 条件化评估（上游固定）**：固定 head+router，改 execution 规则，回答“execution 用 head 做条件化是否真的提升”

> 关键原则：**避免上下游互相依赖造成的“归因混乱”**。做法是：每次只动一个层的自由度，其他层冻结，并把冻结内容写入产物 meta。

---

## 0) 每次实验必须冻结/声明的 6 个轴（口径一致的前提）

不管你在做 A/B/C 哪一层，都必须记录：
- **Universe**：symbols 列表（TopN）与分组（如 HighCap/Alt/Meme）
- **时间窗**：train start/end；OOS start/end
- **粒度**：timeframe（例如 240T=4H）
- **Labels**：horizon / entry_offset / 定义（例如 dir_y=...）
- **Router thresholds**：完整阈值集合（别只记 3 个；best 常常是 7 个）
- **Execution 假设**：returns_source + entry_delay + cost + slippage + (execution profile/version)

---

## A) Head 评估（解耦标签）——回答“NN 多头到底有没有信息量”

### A1. 用什么命令跑？

#### A1.1 标准评估（计算 labels + 预测 + 写出 report）

用 `mlbot nnmultihead eval`（会从 raw 数据生成特征与 labels，速度较慢但“自洽”）：

```bash
mlbot nnmultihead eval --no-docker \
  --config <CONFIG_DIR> \
  --symbols <SYMS> \
  --timeframe 240T \
  --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD> \
  --model <MODEL_PT> \
  --horizon-hours 80 --bar-hours 4 \
  --output-dir <OUT_DIR>
```

产物：
- `<OUT_DIR>/meta.json`
- `<OUT_DIR>/metrics.json`
- `<OUT_DIR>/metrics_summary.md`
- `<OUT_DIR>/report.html`（主要看这个）

#### A1.2 从 model.pt + FeatureStore 重新评估（推荐用于“快速复盘 / 补产物”）

当训练 run 只剩 `model.pt`、或者你更新了 report 模板想补产物，用：

```bash
CUDA_VISIBLE_DEVICES="" /usr/bin/python3 scripts/eval_path_primitives_from_model.py \
  --model <MODEL_PT> \
  --config <CONFIG_DIR> \
  --symbols <SYMS> \
  --timeframe 240T \
  --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD> \
  --features-store-root feature_store \
  --features-store-layer <LAYER> \
  --out-dir <OUT_DIR> \
  --max-rows-per-symbol 6000
```

> 这个脚本支持把 Router tuned thresholds 传进来，让 report 里出现“阈值一致评估”表：

```bash
... scripts/eval_path_primitives_from_model.py \
  --router-mfe-min <MFE_MIN> \
  --router-eff-min <EFF_MIN> \
  --router-dir-conf-trend-min <DIR_CONF_MIN>
```

#### A1.3 只重渲染 report（不重新评估）

当你只改了 HTML/summary 模板，想对旧 run 重渲染：

```bash
mlbot nnmultihead render-report --no-docker --run-dir <RUN_DIR>
```

---

### A2. 看哪些指标？怎么解读？

`report.html` 里你要重点看 4 块：

- **(1) 全局指标（Global）**：用于 sanity check
  - `dir_auc/dir_acc`：方向 head 的弱信息量（0.5≈随机；>0.55 才算“有一点”）
  - `mfe_atr/mae_atr/t_to_mfe_spearman`：回归头的 rank-IC（0≈无信息；正负都可能）
  - `mask_rate`：**有效标签比例**（越低代表标签稀疏，回归类指标会“看起来很差”）

- **(2) Rolling IC/ICIR（滚动稳定性）**：用于检查 drift / 不稳定
  - `roll_icir__*`：>0 才说明“有一点稳定相关”，否则极可能是噪声/不稳

- **(3) Router-like / trade slice（按 Router 参与样本切片）**
  - 这是为了回答“在你真正要交易的区域，head 是否更有信息”
  - 注意：trade slice 是条件评估（会有 selection bias），所以它不能代替全局评估，但对落地更有意义

- **(4) Threshold-consistent evaluation（阈值一致评估 / Router-aligned）**
  - 这块是为了把“回归拟合不高，但阈值决策仍可用”变成可量化证据
  - 常见输出（在 `metrics.json` 里也能看到）：
    - `th_eval__mfe_atr_gt__auc/ap`
    - `th_eval__eff_gt__auc`
    - `th_eval__dir_y__auc`

**A 层结论写法模板（建议固定）**
- “全局信息量”：dir_auc≈x；mfe/mae rank-IC≈y；mask_rate≈z（标签稀疏/不稀疏）
- “稳定性”：roll_icir 是否 >0；是否存在明显漂移
- “Router 对齐”：trade slice 内是否更强；th_eval 的 AUC/AP 是否显著 >0.5 / >base rate

---

## B) System 评估（固定 execution）——回答“这套系统在某个执行假设下能否赚钱”

### B1. 推荐两种 system eval（用于解耦上下游）

#### B1.1 Router-only（不让 execution 使用 head）

用 `returns_source=momentum_proxy`（只用价格过去动量构造对照 returns），用于回答：
> Router 的 action 切分本身（不靠 head-conditioned execution）是否有 edge？

```bash
mlbot nnmultihead predict --no-docker \
  --config <CONFIG_DIR> \
  --symbols <SYMS> --timeframe 240T \
  --start-date <OOS_START> --end-date <OOS_END> \
  --model <MODEL_PT> \
  --output <PREDS_DIR> \
  --feature-store-root feature_store --feature-store-layer <LAYER>

mlbot rule mode-3action --no-docker \
  --preds <PREDS_DIR> --model <MODEL_PT> \
  <THRESHOLDS...> \
  --output <MODE_PARQUET>

mlbot rl build-logs-3action --no-docker \
  --preds <PREDS_DIR> --mode <MODE_PARQUET> --model <MODEL_PT> \
  --symbols <SYMS> --timeframe 240T --start-date <OOS_START> --end-date <OOS_END> \
  --data-path data/parquet_data \
  --returns-source momentum_proxy \
  --output <LOGS_PARQUET>

mlbot rl run-e2e-3action --no-docker \
  --logs <LOGS_PARQUET> --out <E2E_DIR> \
  --entry-delay 0 --cost-per-turnover 0 --slippage-bps 0
```

你主要看：
- `<E2E_DIR>/counterfactual/metrics.json`（rule_sharpe_mean / dd / trade_rate）

#### B1.2 Full system（允许 execution 使用 head）

用 `returns_source=rr_execution`（你当前实验用的），用于回答：
> “head + router + execution”作为整体，系统级能否赚钱？

只需把 build-logs 那步改成：

```bash
... --returns-source rr_execution ...
```

---

### B2. 怎么读 E2E 报告？（统一口径）

`mlbot rl run-e2e-3action` 会输出：
- `shadow/`：BC 复现 rule 行为的稳定性门槛（不是 PnL）
- `counterfactual/`：PnL 指标（核心）
- `fsm_decision.json`：根据 gate 的自动决策（可忽略，除非你在走自动化上线）

**你主要看 `counterfactual/report.html` + `metrics.json`。**

此外，我们把 Router-aligned 两块诊断也写进了 counterfactual 报告（用于统一“tuning/正式评估”口径）：
- **阈值一致评估（trade slice 上的 AUC/AP）**
- **trade slice rolling drift tail**

跑 e2e 时建议显式带上 tuned thresholds（写入 meta，便于复盘）：

```bash
mlbot rl run-e2e-3action --no-docker \
  --logs <LOGS_PARQUET> --out <E2E_DIR> \
  --preds-in-log1p \
  --router-mfe-min <MFE_MIN> \
  --router-eff-min <EFF_MIN> \
  --router-dir-conf-trend-min <DIR_CONF_MIN>
```

---

## C) Execution 条件化评估（上游固定）——回答“execution 用 head 条件化是否真的提升”

这一步的“归因策略”是：
- 固定：同一份 `preds` / 同一份 `mode`（Router thresholds 不变）
- 只改：execution 规则与参数（ExecutionSpec v1/v2/v3）
- 在同一段 OOS 上对比：Sharpe/DD/turnover/cost sensitivity

当前 repo 已支持在 build-logs 阶段切换 execution 假设（`--returns-source`），但**execution profile 的版本化/对照实验协议**需要你明确一个“ExecutionSpec”并把它写进 artifacts meta（宪法化/版本化）。

> 这一步建议作为后续 TODO：先把 A/B 的评估口径完全固定，否则 execution 一改，所有结论都会“漂”。

---

## 建议的迭代顺序（把互相依赖变成可控）

1) **先做 A（Head 解耦评估）**：确认 head 不是纯随机，并且在 trade slice / threshold-consistent 上有“可用证据”
2) **再做 B1（momentum_proxy）**：隔离 execution，确认 Router action 切分本身不是幻觉
3) **再做 B2（rr_execution）**：看全系统表现
4) **最后做 C（execution 条件化）**：在上游固定的前提下迭代 execution

---

## 本协议对应的“缺口/改进点”（待办）

建议优先级（高→低）：
- **(P0)** 统一把 A/B 的关键冻结轴（symbols/window/thresholds/returns_source/entry_delay/cost/slippage）写入所有产物 meta（训练/评估/e2e）
- **(P0)** 增加一个一键脚本/命令：给定 `MODEL + PREDS + (optional) LOGS`，自动跑完 A + B1 + B2 并汇总到一个 `summary.html`
- **(P1)** ExecutionSpec 版本化（YAML）+ 在 build-logs 里可选 profile，并写入 meta（用于 C 层对照）

