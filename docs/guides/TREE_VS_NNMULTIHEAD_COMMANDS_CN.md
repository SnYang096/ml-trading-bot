## Tree vs nnmultihead 命令对齐清单（CN）

这份表用于对齐两条体系的“同类命令”，方便互相学习/复用。

### 共同目标
- **Tree（策略模型）**：直接优化策略目标（Sharpe/收益/回撤/交易数…）
- **nnmultihead（路径原语）**：先优化 primitives heads（`dir/mfe/mae/t_to_mfe`），再经 Router+Execution 才能看到 Sharpe

---

### 1) FeatureStore（两边都推荐用）
- **Tree**：`mlbot feature-store build`（为策略特征缓存/复用，加速搜索）
- **nnmultihead**：`mlbot feature-store build`（为 primitives 训练/搜索提供宽表）

---

### 2) Pool B（因子筛选 / 候选池）
- **Tree**：`mlbot analyze factor-eval`  
  - 目标：策略 label/收益相关指标
  - 输出：`results/pools/<strategy>/pool_b/features_pool_b.yaml`

- **nnmultihead**：`mlbot nnmultihead factor-eval`  
  - 目标：primitives labels（dir/mfe/mae/ttm）
  - 输出：`results/pools/<nn_config>/pool_b_primitives/features_pool_b_primitives.yaml`

---

### 3) 组合搜索（PoolB + Semantic groups）
- **Tree**：`mlbot diagnose feature-group-search`  
  - 输入：`features_base.yaml`（PoolA）+ PoolB + `feature_groups*.yaml`（Semantic）
  - 算法：`greedy/halving/beam/sffs/pipeline`

- **nnmultihead**：`mlbot nnmultihead feature-group-search`  
  - 输入：`<base-config>/features_base.yaml`（PoolA）+ PoolB + `config/feature_groups.yaml`（Semantic）
  - 算法：`greedy/halving/beam/sffs/pipeline`
  - 关键差异：budget 维度用 `epochs`（tree 常用 seeds/预算）
  - 关键差异：默认 `exclude-columns=atr`（atr 只用于 labels，不喂给 MLP；可改）

---

### 4) 训练 / 评估
- **Tree**：`mlbot train ...` / `mlbot diagnose holdout-eval` / `mlbot train final`
- **nnmultihead**：`mlbot nnmultihead train` / `mlbot nnmultihead eval` / `mlbot nnmultihead render-report`

---

### 5) E2E（Sharpe）
- **Tree**：策略模型直接 backtest 出 Sharpe（pipeline 内含）
- **nnmultihead**：必须走 Router + Execution：
  - `mlbot nnmultihead predict`
  - `mlbot rule mode-3action`
  - `mlbot rl build-logs-3action`
  - `mlbot rl run-e2e-3action`


