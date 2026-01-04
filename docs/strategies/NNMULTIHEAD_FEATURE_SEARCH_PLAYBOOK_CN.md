## nnmultihead（Path Primitives）特征选择 Playbook（CN）

> 参考：`docs/strategies/FEATURE_SEARCH_PLAYBOOK_CN.md`（树模型/策略特征搜索）。  
> 本文把 “特征搜索” 迁移到 **nn 多头路径原语** 的语境：目标不是某个策略的 label，而是 primitives heads（`dir/mfe/mae/t_to_mfe`）。

---

## 0. 核心差异：树模型 vs nnmultihead

- **树模型**：目标通常是 “策略 label / 回测收益”，特征可以强语义绑定（例如 SR 反转专用形态）。
- **nnmultihead**：目标是 **通用路径原语**（Router/Execution 的输入），更偏向：
  - 可泛化（跨币、跨阶段）
  - 更结构化（方向/潜在收益/潜在风险/到达时间）
  - 更稳定（rolling ICIR/分组 IR）

因此：树模型里“很好用”的强语义特征，放进 nnmultihead **不一定更好**；很多更适合作为 Router detector/gating。

---

## 1) 候选集（Candidates）怎么来？

你需要先定义一个“候选宇宙”，否则任何筛选都会变成拍脑袋。

推荐两种来源：

- **A. 从某个 strategy 的 `features_all.yaml` 借候选集**（快速，但要注意一致性）
  - 示例：`config/strategies/sr_reversal_rr_reg_long/features_all.yaml`
  - 注意：候选必须 **在你的 FeatureStore layer 里真实存在**，否则会被跳过（summary 里会给 missing 列样本）。

- **B. 为 nnmultihead 单独维护一个候选清单**（长期更稳）
  - 例如：`config/nnmultihead/path_primitives_4h_80h_min/features_candidates.yaml`
  - 目标：候选集与 FeatureStore layer 完全对齐（避免 “574 候选，只有 91 存在” 这种浪费）

---

## 2) Step-1：单因子筛选（Primitives Pool B）

命令：`mlbot nnmultihead factor-eval`

它会做什么？
- 从 FeatureStore 读取特征
- 计算 primitives labels：`dir_y / mfe_atr / mae_atr / t_to_mfe`
- 以 `(symbol, month)` 为分组，计算稳定性统计（IR/t-stat）
- 导出 `features_pool_b_primitives.yaml`（Pool B：下一步搜索的候选池）

### 示例（BTC+ETH，2023-2024，4H）

```bash
mlbot nnmultihead factor-eval --no-docker \
  --config-dir config/nnmultihead/path_primitives_4h_80h_min \
  --candidates-yaml config/strategies/sr_reversal_rr_reg_long/features_all.yaml \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --features-store-root feature_store \
  --features-store-layer features_83f12ecc5e \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --min-samples-per-group 120
```

输出（约定目录）：
- `results/pools/path_primitives_4h_80h_min/pool_b_primitives/primitives_factor_eval_metrics.csv`
- `results/pools/path_primitives_4h_80h_min/pool_b_primitives/features_pool_b_primitives.yaml`

### 如何解读 `primitives_factor_eval_metrics.csv`？

每个 `factor_col` 会针对不同 target 产出一套统计：
- **Spearman**：对 `mfe_atr/mae_atr/t_to_mfe/dir_y` 的 rank 相关（越大越好，越稳越好）
- **dir_y AUC**：这里用的是 **AUC-0.5（居中）**，无信息量≈0；避免常数列 “AUC=0.5” 被误判为强信号
- **IR**：分组均值 / 分组标准差（越大越稳）
- **tstat**：分组均值的 t-stat（越大越稳）
- **nan_rate**：缺失比例（高缺失会被过滤）

> 重要：4H 一个月大约 ~180 bars；还要扣掉 horizon 末端不可用，所以 `--min-samples-per-group` 建议 80–150，默认 200 往往会过严。

---

## 3) Step-2：把 Pool B 变成 nnmultihead 的 `features.yaml`

nnmultihead 的特征配置是结构化的（`required` + `optional_blocks`），见：
- `config/nnmultihead/path_primitives_4h_80h_min/features.yaml`

### 推荐的“落地策略”

- **required（底座）**：少而稳（波动率/范围/动量/趋势/成交量等结构性变量）
- **optional_blocks（可选块）**：更重、更不稳定、更可能缺失的块（order-flow、vpvr、dl_seq 等），训练时 mask，推理时可缺失

把 Pool B 合并到这里时，建议按块归类，而不是机械地把所有 Pool B 都塞进 required：
- 例如 Pool B 选到了 `vpin_features_f`，但你当前 nn 配置把 order-flow 拆成了更细粒度的块（`vpin_base_aligned_features_f`/`vpin_derived_features_f` 等），这时可以：
  - 继续沿用你当前更细粒度的 block（更可控）
  - 或者用 `vpin_features_f` 作为一个“宏特征入口”（更省事但更粗）

---

## 4) Step-3：多变量对比（Wrapper / Ablation）

Pool B 只是 “单因子筛选”。最终还需要多变量验证（因为 NN 存在协同效应、冗余、共线）。

推荐两个层级：

### A. 快速环（Primitives 指标）
- 固定训练设置（epochs 少一些、固定 seeds、固定时间窗）
- 对比 `dir_auc`、各 head 的 spearman、rolling ICIR、mask_rate

### B. 慢速环（E2E 指标）
- `nnmultihead predict` → `rule router` → `execution_returns_rr` → `eval`
- 目标：Sharpe / 回撤 / 交易数 / 稳定性（跨币一致性）

> 工程上：先用 A 快速缩小搜索空间，再用 B 做真实复核。

---

## 4.1) nnmultihead 组合搜索（跟进树模型的升级算法）

树模型侧的 `feature-group-search` 已经支持 `halving/beam/sffs/pipeline`，nnmultihead 也已跟进同样的搜索算法：

- 命令：`mlbot nnmultihead feature-group-search`
- 候选池：建议直接使用 `mlbot nnmultihead factor-eval` 导出的 `features_pool_b_primitives.yaml`
- objective：默认 `dir_auc`（也可以换 `roll_icir__dir`、`mfe_atr_spearman` 等）

示例（推荐从 pipeline 开始）：

```bash
mlbot nnmultihead feature-group-search --no-docker \
  --base-config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --features-store-root feature_store \
  --features-store-layer features_83f12ecc5e \
  --pool-b-yaml results/pools/path_primitives_4h_80h_min/pool_b_primitives/features_pool_b_primitives.yaml \
  --objective dir_auc \
  --search-algo pipeline \
  --halving-stages 3,6,10 \
  --beam-width 3 \
  --max-steps 6 \
  --epochs 10 \
  --output-dir results/nn_feature_group_search/pipeline_poolb_dir_auc
```

输出：
- `results/.../nn_feature_group_search_result.json`

说明：
- nn 侧的 “budget 维度” 用的是 **epochs**（对应 tree 侧用 seeds 的思路）
- 单次评估会跑一个小训练（因此比 tree 慢）；建议先用较小 epochs + pipeline 缩小候选

---

## 5) 推荐的“工业化升级”（与树模型 playbook 对齐）

你在 `FEATURE_SEARCH_PLAYBOOK_CN.md` 里列的 Successive Halving / Beam / SFFS 也适用于 nnmultihead：
- **Successive Halving（强烈推荐）**：先用小预算筛掉噪声组合，再用大预算复核
- **Beam Search**：保留 top-K 路径，专治 “A+B 才有效”
- **SFFS**：允许回删，修正 early wrong pick

区别仅在于 objective：
- 树模型 objective：Sharpe / 收益
- nnmultihead objective（快环）：primitives 指标（AUC/spearman/roll_icir）
- nnmultihead objective（慢环）：E2E Sharpe（最终）

---

## 6) 产物如何接入后续流程？

当前 `mlbot nnmultihead factor-eval` 导出的 PoolB 文件：
- `results/pools/path_primitives_4h_80h_min/pool_b_primitives/features_pool_b_primitives.yaml`

用途：
- 作为 nnmultihead 的“候选池”输入（人为合并到 `config/nnmultihead/.../features.yaml`）
- 或作为后续自动化搜索（Successive Halving/Beam）候选输入（未来可以封装成 `nnmultihead feature-group-search`）

---

## 7) 常见坑（强烈建议你优先避开）

- **候选不对齐**：候选列不在 FeatureStore layer 里 → 评估会跳过，浪费算力
- **AUC 没居中**：AUC=0.5（无信息）会被误判为 “稳定且高”
- **min_samples_per_group 过大**：4H 月样本太少 → 组被跳过 → 结果失真/过少
- **把强语义特征强行塞进 nn 底座**：会导致泛化下降；更适合做 detector/gating


