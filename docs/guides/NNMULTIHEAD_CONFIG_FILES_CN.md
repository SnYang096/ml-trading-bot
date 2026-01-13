# nnmultihead：配置文件职责图（TaskSpec / FeaturePlan / features.yaml / labels.yaml / model.yaml）

本文回答一个问题：**每个配置文件到底负责什么，为什么要拆开。**

## 1) 三层职责（从“稳定”到“可变”）

### A) Feature Registry（全局注册表，最稳定）
- `config/feature_dependencies.yaml`
  - **职责**：定义每个 *feature node*（如 `vpin_base_aligned_features_f`）的计算方式、依赖、输出列 `output_columns`、归一化元信息等。
  - **特点**：这是“字典/注册表”，不是“本次 run 选哪些特征”。

### B) FeaturePlan（nnmultihead 特征域配置，稳定但可版本化）
- `config/nnmultihead/path_primitives_4h_80h_min/feature_plan_v1.yaml`
  - **职责**：把“特征选择与计算开关”集中到一个文件里：
    - `tiers_enabled` + `tier_feature_files`：本计划默认启用哪些 Tier（Tier0/1/2…）
    - `optional_blocks_library`：把一组 feature nodes 聚合成一个 block（例如 `vpin_block`）
    - `optional_blocks_enabled`：默认启用哪些 blocks（通常为空，保持 baseline 干净）
      - **启用 block 会发生什么**：
        - 这些 block 对应的 feature nodes 会被写入派生 `features.yaml.requested_features.optional_blocks`
        - 同时派生 `features.yaml.feature_contract.optional_blocks` 会被自动生成（列级映射：node → output_columns）
        - 训练时可使用 `missingness_policy.block_dropout_p` 做 **block 级 dropout**（让模型对“某块可用/不可用”更鲁棒）
      - **不启用 block 会怎么样**：
        - 这组 nodes **不会被计算/不会被喂给模型**（派生 config 里不出现）
        - 对应的列级 block mask/缺失策略也不参与（因为 block 根本不在本次任务中）
      - **列 mask 功能还在吗？**
        - **在**。只是我们把“列级 block 定义”从独立文件里移走，改为在 materialize 时从
          `feature_dependencies.yaml.output_columns` 自动推导写入派生 `features.yaml.feature_contract.optional_blocks`。
        - 也就是说：**你最终运行用的派生 config 里仍然有列级 block mask 语义**（只是来源更单一、更不容易漂移）。
    - `feature_store`：FeatureStore 的默认 root/layer/并行/fast_features
    - `exclude_columns`：哪些列“计算但不喂给 MLP”（例如 `atr`）
    - `feature_contract`：稳定的合同语义（baseline `minimal_required_cols` + `missingness_policy`）

### C) TaskSpec（任务合同，本次 run 的“非特征”定义，可变）
- `config/tasks/task_spec_v1.yaml`
  - **职责**：定义这次任务的目标与边界：
    - 数据窗口（train/holdout/oos）、universe、验收门槛、enforcement（constitution/kpi gate）等
    - 通过 `feature_plan_ref` 引用 FeaturePlan（让特征域配置不散落在 TaskSpec）
    - 可选 `feature_plan_overrides`：做 ablation 时，只写“少量覆盖”（例如临时开 Tier2、开 vpin_block）
  - **为什么 TaskSpec 放在 `config/tasks/`（而不是 nnmultihead 目录下）？**
    - 因为 TaskSpec 是**跨模块的“任务合同”**：它同时约束 `nnmultihead`（模型训练/推理）、
      `rule`（router）、`rl`（build-logs/e2e）、`enforcement`（constitution/kpi gate）等。
    - `config/nnmultihead/...` 目录下我们尽量只放“模型域/特征域”相关配置（FeaturePlan、Tier lists、base templates）。
    - 这样好处是：你以后同一个 TaskSpec 可以切换不同 model_kind（nn/tree/…）时仍保持“任务层”一致。

## 2) 为什么还保留 `features.yaml / labels.yaml / model.yaml`？

nnmultihead 的脚本/loader 仍然以“目录里存在这些文件”为运行契约（`StrategyConfigLoader.REQUIRED_FILES`）。

### `config/nnmultihead/path_primitives_4h_80h_min/features.yaml`
- **职责**：模板壳（schema carrier）。
- **事实**：真实可执行的 `features.yaml` 由命令在运行前 materialize 到派生目录：
  - `results/derived_configs/<task_id>/.../features.yaml`

### `config/nnmultihead/path_primitives_4h_80h_min/labels.yaml`
- **职责**：Path Primitives 标签配置（例如 horizon、ATR 归一化等相关参数）。
- **它和 TaskSpec 重复吗？**
  - **不重复**：TaskSpec 的 `model_plan` 只声明“用哪个 base_config_dir + 少量训练超参覆盖”，
    但 label 的细节（比如标签生成参数、字段约束）仍然由 `labels.yaml` 承载。

### `config/nnmultihead/path_primitives_4h_80h_min/model.yaml`
- **职责**：模型/训练超参数的配置载体（trainer params 等）。
- **它和 TaskSpec 重复吗？**
  - **不重复**：TaskSpec 的 `model_plan.training` 只放“本次任务的少量覆盖/关键开关”（epochs/lr/seed 等），
    `model.yaml` 才是该模型的“默认训练参数/结构参数”的承载体。
  - 经验上：TaskSpec 负责“本次 run 变的部分”，base config 负责“默认不变的部分”。

## 3) 最终“落地配置”长什么样（派生 config）

当你运行：
- `mlbot nnmultihead train --task-spec ...`
- `mlbot nnmultihead pipeline-3action-e2e --task-spec ...`

系统会先 materialize 一个派生 config（目录里会出现 `derived_from_task_spec.json`），并写出一份“自包含”的 `features.yaml`：
- `requested_features.required`：来自启用的 tiers（feature nodes）
- `requested_features.optional_blocks`：来自启用的 blocks（node 列表）
- `feature_contract`：来自 FeaturePlan.feature_contract（missingness_policy + baseline minimal cols）
  - 并且 `minimal_required_cols` 会被自动更新为：
    - baseline minimal_required_cols
    - ∪（tier nodes 的 output_columns union）
  - `optional_blocks`（列级）会从启用 blocks 的 nodes 的 `output_columns` 推导出来（用于 block mask/dropout）

## 4) 常见困惑：Tier vs Block vs Contract

- **Tier（node 级）**：控制“这次要不要计算/喂哪些 feature nodes”（预算/自由度管理）
- **Block（node 级）**：控制“把哪些 nodes 当作一整块一起开关”（orderflow/cluster 等）
- **Contract（列级语义）**：控制“这整块在列层面如何被当作可缺失块；缺失/关闭时怎么处理”

### 为什么需要 Block 这个开关？（是不是只是为了消融？）
- **是为了消融 + 也是为了工程可控性**：
  - Tier 文件更像“版本化的大开关”（Tier0/1/2… 是你要长期复用的 feature set 版本）
  - Block 更像“可插拔的增强包”（例如 tick-heavy orderflow 相关），你可以在不修改 Tier 文件的情况下做：
    - 同一份 Tier 主干不变，只开/关某个增强块 → 对比训练/推理/报告
    - 在 live 特征缺失时，block 级策略（dropout/mask）能明确表达“这块可能不可用”
  - 所以你的理解可以是：
    - **Tier：大批量、可版本化的消融/自由度管理**
    - **Block：更细粒度、可插拔的消融/工程开关**

## 5) ICIR 反向的列：在 MLP 里要不要取反？
- **通常不需要**。
  - 对树模型：我们常用 `invert_features` 是为了把“反向有效”的单调关系也变成候选（方便规则/树的阈值/单调性）。
  - 对 MLP：模型可以通过权重学到正/负相关（输入不需要人为取反）。
- **什么时候会想取反？**
  - 只有在“你做的是基于相关性/单调性假设的规则或阈值选择”时（例如 router/gate 的 hand-crafted rule），
    才会显式把反向列取反作为一种工程手段。

## 6) 一个命令复用：对比 Tier（TaskSpec） vs PoolB（factor-eval 输出）

新增命令（用于你反复做“Tier 主干 vs PoolB 建议”的差分对比）：

```bash
mlbot nnmultihead compare-feature-sets --no-docker \
  --task-spec config/tasks/task_spec_v1.yaml \
  --base-config config/nnmultihead/path_primitives_4h_80h_min \
  --poolb-yaml results/factor_eval/<RUN>/features_pool_b_primitives.yaml
```

输出：
- `features_compare_summary.json`
- `features_compare_summary.md`

默认输出目录：
- `results/feature_compare/<task_id>__<poolb_stem>/`


