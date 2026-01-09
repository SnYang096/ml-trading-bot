## NN 多头基座 → Rule(3-action) → RL/BC e2e（长文档）

本文件收录了从 `README_CN.md` 抽离出来的“NN 多头 + 3-action + RL/BC”端到端命令解释与长段落说明，避免 README 过长。

建议先读：
- `docs/architecture/EXPERIMENT_LOOP_ARCHITECTURE.md`
- `docs/时序模型/架构：NN多头路径原语（Path Primitives）+Router解耦升级.md`

---

## 配置文件说明：`features.yaml`（统一管理特征计算和契约）

在 `config/nnmultihead/path_primitives_4h_80h_min/` 目录下，**只需要一个配置文件**：

### `features.yaml`（特征计算 + 使用契约，统一管理）

这个文件使用**新的结构化格式**，在 `feature_pipeline.requested_features` 中直接分类特征：

#### 新格式（推荐）：结构化分类

```yaml
feature_pipeline:
  requested_features:
    required:
      - atr_f
      - trend_r2_20_f
      # ... 所有必需特征
    optional_blocks:
      compression_blocks:
        - compression_duration_f
        - compression_energy_f
      ticks_orderflow_blocks:
        - vpin_f
  missingness_policy:
    append_block_mask: true
    block_dropout_p: 0.05
```

**关键点**：
- **`required`**：必需特征，训练和推理时都必须存在
- **`optional_blocks`**：可选特征块，训练时会随机 mask（block-dropout），让模型对特征缺失鲁棒
- **所有特征都会计算**：`required` 和 `optional_blocks` 中的所有特征都会被 `FeatureComputer` 计算
- **自动推导契约**：从 `required` 和 `optional_blocks` 自动推导出 `minimal_required_cols` 和 `optional_blocks`（用于 block-dropout/mask）

#### 旧格式（向后兼容）：扁平列表

```yaml
feature_pipeline:
  requested_features:
    - atr_f
    - trend_r2_20_f
    # ... 所有特征都被视为 required
```

如果 `requested_features` 是列表，所有特征都被视为 `required`。

### 工作流程

```
原始数据 (OHLCV)
  ↓
[features.yaml: feature_pipeline.requested_features] → 定义要计算哪些特征
  ↓
FeatureComputer 计算特征 → 生成特征列（例如 atr, trend_r2_20, compression_score, ...）
  ↓
[features.yaml: feature_contract] → 定义如何使用这些特征
  ↓
- 校验 minimal_required_cols 是否存在
- 解析 optional_blocks（例如 *vpin*, *trade_cluster*）
- 应用 missingness_policy（block-dropout/mask）
  ↓
模型训练/推理
```

### 关键点

1. **所有特征都会计算**：`feature_pipeline.requested_features` 中列出的所有特征都会被计算
2. **训练时会 mask**：`feature_contract.optional_blocks` 中的特征块在训练时会随机 mask（block-dropout），让模型对特征缺失鲁棒
3. **一个文件维护**：不再需要单独的 `feature_contract.yaml`，减少维护负担
4. **向后兼容**：如果存在 `feature_contract.yaml`，代码仍会读取（legacy 支持）

### 示例

```yaml
feature_pipeline:
  requested_features:
    - atr_f
    - trend_r2_20_f
    - compression_score_f
    # ... 所有特征都会被计算

feature_contract:
  minimal_required_cols:
    - atr
    - trend_r2_20
    - compression_score
    # 训练时会校验这些列是否存在

  optional_blocks:
    ticks_orderflow_semantic_blocks:
      - "*vpin*"
    # 这些特征块在训练时会随机 mask（block-dropout）
    # 让模型对特征缺失鲁棒

  missingness_policy:
    append_block_mask: true
    block_dropout_p: 0.05
    # 训练时 5% 的概率随机 mask 一个 optional block
```

### 为什么合并？

- **减少维护负担**：只需要维护一个文件，而不是两个
- **逻辑更清晰**：特征计算和使用契约在同一个文件中，更容易理解
- **避免不一致**：不会出现 `features.yaml` 和 `feature_contract.yaml` 不一致的情况

---

## 目标与产物

- **目标**：训练/推理 NN 多头（path primitives）→ 生成 `mode`（NO_TRADE/MEAN/TREND）→ 组装 RL/BC logs（含 `ret_mean/ret_trend`）→ 一键跑 shadow/counterfactual/fsm。

### 核心产物说明（都是数据/中间产物，不是模型）

这些文件是 pipeline 的**中间数据/评估报告**，用于连接各个步骤：

1. **`preds_*.parquet`（NN 推理输出，数据）**
   - **内容**：每行是一个时间戳，包含 NN 模型对路径原语的预测值
   - **列**：`pred_dir_prob`（方向概率）、`pred_mfe_atr`（最大有利偏移）、`pred_mae_atr`（最大不利偏移）、`pred_t_to_mfe`（到达 MFE 的时间）
   - **用途**：作为 Rule Router 的输入，用于生成交易决策（`mode`）
   - **生成命令**：`mlbot nnmultihead predict`

2. **`mode_3action.parquet`（Router 决策输出，数据）**
   - **内容**：每行是一个时间戳，包含 Rule Router 根据 heads 生成的交易模式
   - **列**：`mode`（NO_TRADE/MEAN/TREND）、可选 `confidence`/`reason`
   - **用途**：作为 Execution 层的输入，决定“要不要交易、用什么策略执行”
   - **生成命令**：`mlbot rule mode-3action`

3. **`logs_3action.parquet`（RL/BC 训练数据，数据）**
   - **内容**：每行是一个“状态-动作-奖励”三元组（transition），用于 RL/BC 训练
   - **列**：`symbol`、`timestamp`、`mode`（action）、`head_*`（state 的一部分）、`drawdown`、`ret_mean`、`ret_trend`（rewards）
   - **用途**：喂给 BC（Behavior Cloning）或 Offline RL（如 IQL）训练 Router 策略
   - **生成命令**：`mlbot rl build-logs-3action`

4. **`results/rl/e2e/*`（评估报告，HTML/JSON）**
   - **内容**：shadow evaluation、counterfactual evaluation、FSM decision 的评估报告
   - **文件**：`shadow_report.html`、`counterfactual_metrics.json`、`fsm_decision.json` 等
   - **用途**：用于判断 BC/RL Router 是否比 Rule Router 更好，是否满足上线安全约束
   - **生成命令**：`mlbot rl run-e2e-3action`

### Pipeline 数据流示意

```
原始数据 (OHLCV)
  ↓
[NN 模型推理]
  ↓
preds_*.parquet (路径原语预测)
  ↓
[Rule Router 决策]
  ↓
mode_3action.parquet (交易模式)
  ↓
[组装 logs + 计算 ret_mean/ret_trend]
  ↓
logs_3action.parquet (统一日志：评估/复盘/可选训练)
  ↓
【主链路（推荐，上线/回测主力）】
  - Rule Router + Execution（先固定）+ Gate（树模型导出的 allow/deny）
  - 输出：counterfactual/report.html（PnL 口径）+ Router-aligned diagnostics（AUC/AP/漂移）
  ↓
results/rl/e2e/* (评估报告：以 counterfactual 为主)

【可选链路（非必跑）】
  A) BC shadow（行为一致性门禁，不是为了赚钱）
     logs_3action.parquet → BC(3-action) → shadow_report.html（看行为分布/切换率/是否塌缩）
  B) Offline RL（研究/探索上限，上线前必须配合宪法/门禁）
     logs_3action.parquet → RL policy → counterfactual/report.html（对照 Rule）
```

### 主链路 vs 可选链路（流程图式文字，推荐用法）

```
                    ┌──────────────────────────────────────────────────────┐
                    │ 共同前置（必须）                                      │
                    │ OHLCV → nnmultihead predict → preds_*.parquet         │
                    │ preds → rule mode-3action → mode_3action.parquet      │
                    │ preds+mode+raw → build-logs-3action → logs_3action.parquet │
                    └──────────────────────────────────────────────────────┘

主链路（推荐：先把规则 Router + execution + tree gate 做硬、可控、低维护）
  logs_3action.parquet
    → (固定 execution 假设 / 版本化) + (Gate: tree 导出规则，返回 allow/deny + reason)
    → Counterfactual 评估（PnL）+ Router-aligned 诊断（trade slice / AUC/AP / rolling drift）
    → 产物：results/rl/e2e/*/counterfactual/report.html

可选链路 A（BC shadow：行为一致性门禁）
  logs_3action.parquet
    → BC(3-action) 复现 rule 行为
    → 产物：results/rl/e2e/*/shadow/shadow_report.html
    → 用途：判断“行为是否稳定/是否塌缩”，不是 PnL 优化

可选链路 B（Offline RL：研究/探索）
  logs_3action.parquet
    → RL policy（更高自由度）
    → counterfactual 对照 Rule
    → 上线前必须：宪法（position/execution）+ gate/detector 约束自由度
```

### 为什么需要这些中间产物？（每个 bar 都输出）

**是的，每个 bar（每个时间戳）都输出一行数据。** 这些中间产物不是必须的，但强烈推荐保存，原因如下：

#### 1. **可追溯性与调试**
- 当 Router 决策异常（例如某天突然大量 NO_TRADE）时，你可以回溯：
  - `preds_*.parquet`：看 NN 预测是否正常
  - `mode_3action.parquet`：看 Rule Router 的阈值是否合理
  - `logs_3action.parquet`：看 `ret_mean/ret_trend` 计算是否正确
- **没有这些文件，你只能重新跑整个 pipeline**，耗时且可能因为数据/代码版本不一致导致无法复现

#### 2. **可选：离线训练 BC/RL（不是必跑）**
- `logs_3action.parquet` 的核心价值首先是：**统一评估/复盘口径**（counterfactual + Router-aligned diagnostics）
- BC（Behavior Cloning）和 Offline RL 的训练只是它的**可选用途**：
  - **BC shadow**：用于“行为一致性/稳定性门禁”（验证能否稳定复现 rule 的 mode 分布/切换率）
  - **Offline RL**：用于研究探索（自由度更大、维护成本更高，不建议作为 v1 主链路）

#### 3. **A/B 对比与实验迭代**
- 当你修改 Rule Router 阈值或尝试新的 Router 策略时，可以：
  - 复用同一个 `preds_*.parquet`（NN 预测不变）
  - 只重新跑 `mode_3action` 和 `logs_3action`
  - 对比不同 Router 策略的效果
- **没有中间产物，每次实验都要从 NN 推理开始**，浪费计算资源

#### 4. **数据量估算**
- 4H 框架：每天 6 个 bar，一年约 2190 个 bar
- 多 symbol（例如 30 个）：一年约 65,700 行
- `preds_*.parquet`：每行约 4 个 float（pred_dir_prob, pred_mfe_atr, pred_mae_atr, pred_t_to_mfe）≈ 16 bytes/行 ≈ 1MB/年/symbol
- `mode_3action.parquet`：每行约 1 个 string + 几个 float ≈ 50 bytes/行 ≈ 3MB/年/symbol
- `logs_3action.parquet`：每行约 10+ 列（symbol, timestamp, mode, head_*, ret_*, drawdown）≈ 100 bytes/行 ≈ 6MB/年/symbol
- **30 个 symbol 一年总计约 300MB**，完全可以接受

#### 5. **可选：不保存中间产物（不推荐）**
如果你确定不需要回溯/调试/离线训练，可以：
- 在 `mlbot nnmultihead predict` 后直接 pipe 到 `mlbot rule mode-3action`
- 在 `mlbot rule mode-3action` 后直接 pipe 到 `mlbot rl build-logs-3action`
- 但这样**无法做 A/B 对比**，也无法积累历史 logs 用于 RL 训练

**结论**：这些中间产物是**可选的，但强烈推荐保存**，因为存储成本低（MB 级），但带来的可追溯性、调试能力、实验灵活性价值很大。

### 为什么树模型不需要这些中间产物？

**树模型（LightGBM/XGBoost/CatBoost）不需要这些中间产物**，原因如下：

#### 1. **树模型输出的是直接的策略信号，不是原语**
- **树模型**：输出 `signal`（例如 `sr_reversal_long` 的 entry/exit 信号），可以直接喂给 Execution/Backtest
- **NN 多头**：输出的是**路径原语**（dir/mfe/mae/ttm），需要 Router 把原语映射成 action（NO/MEAN/TREND）

#### 2. **树模型不需要 Router 层**
- **树模型**：训练时已经学了“什么时候 entry/exit”，推理时直接输出信号，**不需要 Router 决策**
- **NN 多头**：训练时只学了“路径原语”，需要 Router 决定“用哪个原语、怎么执行”，所以需要 `mode_3action.parquet` 和 `logs_3action.parquet`

#### 3. **树模型的输出可以直接用于回测**
- **树模型**：`mlbot train sr-reversal-long` → 直接输出 `backtest_df_test.parquet`（包含 entry/exit 信号），可以直接用于 VectorBT 回测
- **NN 多头**：`mlbot nnmultihead predict` → 输出 `preds_*.parquet`（原语），还需要 Router → Execution → 回测，所以需要中间产物来连接各个步骤

#### 4. **树模型的调试方式不同**
- **树模型**：如果回测结果异常，可以直接看：
  - 模型的特征重要性（`feature_importance`）
  - 预测值的分布（`pred` 列）
  - 回测的 entry/exit 点（`backtest_df_test.parquet`）
- **NN 多头**：如果 Router 决策异常，需要回溯：
  - NN 预测是否正常（`preds_*.parquet`）
  - Router 阈值是否合理（`mode_3action.parquet`）
  - 奖励计算是否正确（`logs_3action.parquet`）

#### 5. **树模型不需要 RL/BC 训练数据**
- **树模型**：训练数据就是特征 + 标签（例如 `sr_reversal_long` 的 return/Sharpe），不需要“状态-动作-奖励”三元组
- **NN 多头**：如果要训练 BC/RL Router，需要 `logs_3action.parquet`（包含 state/action/reward）

**总结**：
- **树模型**：策略模型 → 直接输出信号 → 回测（不需要中间产物）
- **NN 多头**：原语模型 → Router 决策 → Execution → 回测（需要中间产物连接各个步骤）

---

## 最小可跑命令（示例）

### 0)（推荐）先构建 FeatureStore 宽表库

```bash
mlbot feature-store build \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2021-01-01
  --end-date 2025-8-30
  --no-docker

mlbot feature-store build \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-set starter_a \
  --timeframe 240T \
  --start-date 2021-01-01  
```

**参数说明**：
- `--config`：必需，配置目录（包含 `features.yaml`）
- `--symbols`：必需，逗号分隔的交易对
- `--timeframe`：必需，时间框架（例如 `240T` 表示 4H）
- `--data-path`：可选，默认 `data/parquet_data`（原始数据目录）
- `--root`：可选，默认 `feature_store`（FeatureStore 根目录）
- `--layer`：可选，默认 `AUTO`（自动从配置内容生成 layer 名称）
- `--start-date` / `--end-date`：可选，默认加载所有可用数据（不是配置文件里，而是 `data/parquet_data` 里存在的所有月份）
- `--warmup-months`：可选，默认 `1`（用于 stateful 特征的 warmup）
- `--no-docker`：可选，默认在 Docker 中运行

**时间范围说明**：
- 如果不指定 `--start-date` 和 `--end-date`，会加载 `data/parquet_data` 中**所有可用的数据**（按月份自动发现）
- 如果需要限制时间范围，可以指定：
  ```bash
  mlbot feature-store build \
    --config config/nnmultihead/path_primitives_4h_80h_min \
    --symbols BTCUSDT,ETHUSDT \
    --timeframe 240T \
    --start-date 2023-01-01 \
    --end-date 2024-12-31 \
    --no-docker
  ```

**关于 OOS（Out-of-Sample）时间窗口的建议**：

⚠️ **重要**：训练脚本会在数据集的**最后 20%** 作为验证集（time-ordered），但这不是真正的 OOS。真正的 OOS 需要**完全独立的时间窗口**，模型在训练时完全不知道这些数据的存在。

**推荐做法**：
- **预留最近 4-6 个月作为 OOS 测试集**（不要包含在 FeatureStore 的 `--end-date` 里）
- 例如：如果数据到 2025-10-31，训练时用 `--end-date 2025-04-30`，留 2025-05-01 ~ 2025-10-31 作为 OOS
- **3 个月也可以，但可能偏紧**（4H 框架下约 540 个 bar，路径原语 horizon=80 bars 约 13.3 天）

**为什么需要 OOS**：
- 训练时的验证集（最后 20%）仍然在**同一个时间窗口内**，模型可能"记住"了时间模式
- 真正的 OOS 可以验证模型的**泛化能力**和**时间稳定性**
- 对于生产系统，OOS 表现比训练集表现更重要

**示例（推荐）**：
```bash
# 构建 FeatureStore：只到 2025-04-30，留 5-10 月做 OOS
mlbot feature-store build \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2023-01-01 \
  --end-date 2025-10-30 \
  --no-docker

# 训练时也用同样的时间范围
mlbot nnmultihead train \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --start-date 2021-01-01 \
  --end-date 2025-04-30 \
  --epochs 10 \
  --output-dir results/nnmultihead \
  --no-docker

# OOS 评估：用 2025-05-01 之后的数据
mlbot nnmultihead predict \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --model results/nnmultihead/.../model.pt \
  --start-date 2025-05-01 \
  --end-date 2025-10-31 \
  --output results/nnmultihead/preds_oos \
  --no-docker
```

**OOS 评估通过后：重新训练实盘模型（包含 OOS 数据）**

✅ **是的，OOS 评估通过后，应该把最近月份的数据也加入训练，重新训练一个"实盘模型"**。

**为什么需要这样做**：
- **适应最新市场**：OOS 数据（2025-05-01 ~ 2025-10-31）包含了最新的市场模式，加入训练可以让模型更适应当前市场
- **更多样本**：增加训练数据量，提高模型的稳定性和泛化能力
- **时间一致性**：实盘时模型应该基于"到当前时间为止的所有可用数据"训练

**注意事项**：
- ⚠️ **不要过拟合 OOS 数据**：虽然加入了 OOS 数据，但训练时仍然会做 train/val 切分（最后 20% 作为验证集），避免过拟合
- ⚠️ **保持时间切分原则**：训练时仍然按时间顺序切分，不要打乱时间顺序
- ⚠️ **记录模型版本**：实盘模型应该明确标记为"production"版本，与"research"版本区分

**推荐流程**：

```bash
# 1) 扩展 FeatureStore：加入 OOS 月份（2025-05-01 ~ 2025-10-31）
mlbot feature-store build \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2021-01-01 \
  --end-date 2025-10-31 \
  --no-docker

# 2) 重新训练实盘模型（包含 OOS 数据）
mlbot nnmultihead train \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --start-date 2021-01-01 \
  --end-date 2025-10-31 \
  --epochs 10 \
  --output-dir results/nnmultihead/production \
  --no-docker

# 3) 实盘模型推理（用于后续 Router/Execution）
mlbot nnmultihead predict \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --model results/nnmultihead/production/.../model.pt \
  --output results/nnmultihead/production/preds \
  --no-docker
```

**关于滚动训练（Rolling Training）**：

目前 `mlbot train rolling` 是针对**树模型**的，**nnmultihead 还没有 rolling 训练命令**。但你可以用以下方式实现类似效果：

- **方式 1（简单）**：定期（例如每月）重新运行上面的流程，用"到当前时间为止的所有数据"训练新模型
- **方式 2（未来）**：如果需要更自动化的滚动训练，可以基于 `mlbot train rolling` 的实现，为 nnmultihead 添加类似的 rolling 训练命令

**实盘模型 vs 研究模型**：
- **研究模型**：用于 OOS 评估，时间范围到 2025-04-30
- **实盘模型**：用于生产部署，时间范围到 2025-10-31（包含 OOS 数据）
- **建议**：保留两个版本的模型，便于对比和回退

### 1) 训练 NN 多头

```bash
mlbot nnmultihead train \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --data-path data/parquet_data \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --feature-store-root feature_store \
  --feature-store-layer AUTO \
  --epochs 10 \
  --output-dir results/nnmultihead \
  --no-docker
```

### 2) NN 多头推理（多 symbol 输出目录）

```bash
mlbot nnmultihead predict \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --data-path data/parquet_data \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --feature-store-root feature_store \
  --feature-store-layer AUTO \
  --model results/nnmultihead/.../model.pt \
  --output results/nnmultihead/preds_multi \
  --no-docker
```

### 3) 纯规则 Router：从 heads 生成 3-action `mode`

```bash
mlbot rule mode-3action \
  --preds results/nnmultihead/preds_multi \
  --model results/nnmultihead/.../model.pt \
  --output results/rule/mode_3action.parquet \
  --no-docker
```

### 4) 组装 RL/BC logs（把 close 转成 ret_mean/ret_trend，并合并 heads + mode）

```bash
mlbot rl build-logs-3action \
  --preds results/nnmultihead/preds_multi \
  --mode results/rule/mode_3action.parquet \
  --model results/nnmultihead/.../model.pt \
  --data-path data/parquet_data \
  --timeframe 240T \
  --output results/rl/logs_3action.parquet \
  --no-docker
```

### 5) 一键跑 RL(e2e)：shadow → counterfactual → fsm

```bash
mlbot rl run-e2e-3action \
  --logs results/rl/logs_3action.parquet \
  --out results/rl/e2e \
  --no-docker
```


