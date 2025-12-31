# Factor-Eval 与语义特征评估

## 问题发现

**关键问题**：`factor-eval` 不会评估语义节点，因为 `features_all.yaml` 中没有包含它们。

### 原因分析

1. **factor-eval 的输入来源**：
   - 默认从 `--strategy-config` 指定的 `features.yaml` 读取 `requested_features`
   - 如果使用 `features_all.yaml`，则评估其中的所有特征

2. **features_all.yaml 的内容**：
   - 包含大量原始特征节点（DTW、EVT、GARCH、Hilbert 等）
   - **不包含语义节点**（如 `vpin_scene_semantic_scores_f`、`wpt_scene_semantic_scores_f` 等）

3. **结果**：
   - `factor-eval` 只评估 `features_all.yaml` 中的特征
   - 语义节点不在评估范围内
   - Pool B 中就不会包含语义特征

### 验证

检查 `config/strategies/sr_reversal_rr_reg_long/features_all.yaml`：

```bash
python3 << 'EOF'
from pathlib import Path
import yaml

features_all_path = Path("config/strategies/sr_reversal_rr_reg_long/features_all.yaml")
with open(features_all_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
    all_features = data.get('feature_pipeline', {}).get('requested_features', [])

semantic_nodes = [
    'vpin_scene_semantic_scores_f',
    'wpt_scene_semantic_scores_f',
    'trade_cluster_scene_semantic_scores_f',
    'volume_profile_scene_semantic_scores_f',
    'wick_scene_semantic_scores_f',
    'fp_imbalance_scene_semantic_scores_f',
    'liquidity_void_scene_semantic_scores_f',
    'funding_scene_semantic_scores_f',
]

found = [n for n in semantic_nodes if n in all_features]
missing = [n for n in semantic_nodes if n not in all_features]

print(f"找到的语义节点: {len(found)}/{len(semantic_nodes)}")
print(f"缺失的语义节点: {len(missing)}/{len(semantic_nodes)}")
if missing:
    print(f"缺失: {missing}")
EOF
```

**结果**：所有语义节点都缺失。

---

## 解决方案

### 方案 1: 将语义节点添加到 features_all.yaml（推荐）

**优点**：
- 统一管理所有特征
- factor-eval 可以评估语义特征
- Pool B 会包含语义特征

**步骤**：

1. 读取语义 groups，提取所有语义节点
2. 将语义节点添加到 `features_all.yaml` 的 `requested_features` 中
3. 重新运行 `factor-eval` 生成 Pool B

**实现**：

```python
# scripts/add_semantic_to_features_all.py
from pathlib import Path
import yaml

# 加载语义 groups
semantic_file = Path("config/feature_groups_sr_reversal_semantic.yaml")
with open(semantic_file) as f:
    semantic_data = yaml.safe_load(f)
    semantic_groups = semantic_data.get('groups', {})
    semantic_nodes = [f for group in semantic_groups.values() for f in group]

# 加载 features_all.yaml
features_all_path = Path("config/strategies/sr_reversal_rr_reg_long/features_all.yaml")
with open(features_all_path) as f:
    data = yaml.safe_load(f)
    all_features = data.get('feature_pipeline', {}).get('requested_features', [])

# 添加语义节点（去重）
all_features_set = set(all_features)
for node in semantic_nodes:
    if node not in all_features_set:
        all_features.append(node)
        all_features_set.add(node)

# 保存
data['feature_pipeline']['requested_features'] = sorted(all_features)
with open(features_all_path, 'w', encoding='utf-8') as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
```

### 方案 2: 创建 features_all_with_semantic.yaml

**优点**：
- 不修改原始的 `features_all.yaml`
- 可以单独评估语义特征

**步骤**：

1. 复制 `features_all.yaml` 为 `features_all_with_semantic.yaml`
2. 添加语义节点
3. 使用 `features_all_with_semantic.yaml` 运行 `factor-eval`

### 方案 3: 显式指定语义节点

**优点**：
- 不需要修改配置文件
- 可以灵活选择要评估的语义节点

**步骤**：

```bash
mlbot analyze factor-eval \
  -c config/strategies/sr_reversal_rr_reg_long \
  --factors vpin_scene_semantic_scores_f wpt_scene_semantic_scores_f \
  --feature-mode append \
  ...
```

---

## 推荐工作流

### 步骤 1: 添加语义节点到 features_all.yaml

```bash
python scripts/add_semantic_to_features_all.py \
  --strategy sr_reversal_rr_reg_long \
  --semantic-groups config/feature_groups_sr_reversal_semantic.yaml \
  --features-all config/strategies/sr_reversal_rr_reg_long/features_all.yaml
```

### 步骤 2: 重新运行 factor-eval

```bash
mlbot analyze factor-eval \
  -c config/strategies/sr_reversal_rr_reg_long/features_all.yaml \
  -s BTCUSDT -t 240T \
  --start-date 2023-01-01 --end-date 2025-10-31 \
  --output-dir results/pools/sr_reversal_rr_reg_long/pool_b \
  --export-yaml results/pools/sr_reversal_rr_reg_long/pool_b/features_pool_b.yaml \
  --remove-correlated --filter-by-best-lag --no-docker
```

### 步骤 3: 验证 Pool B 包含语义特征

```bash
python scripts/analyze_poolb_semantic_overlap.py
```

---

## 影响分析

### 当前状态

- **Pool B 特征数**: 24 个
- **语义节点数**: 12 个
- **交叉特征数**: 1 个（`liquidity_void_f`）
- **Pool B 中语义特征数**: 0 个（因为语义节点不在 features_all.yaml 中）

### 预期状态（添加语义节点后）

- **Pool B 特征数**: 预计 24 + X 个（X 取决于语义特征的 IC/IR）
- **语义节点数**: 12 个
- **交叉特征数**: 预计 12 个（所有语义节点）
- **Pool B 中语义特征数**: 预计 12 个（如果它们的 IC/IR 足够好）

---

## 建议

1. **立即行动**：将语义节点添加到 `features_all.yaml`
2. **重新生成 Pool B**：使用包含语义节点的 `features_all.yaml` 重新运行 `factor-eval`
3. **验证交叉**：检查 Pool B 是否包含语义特征
4. **更新工作流**：确保后续的 `features_all.yaml` 都包含语义节点

---

## 工具

- `scripts/add_semantic_to_features_all.py`（待创建）：将语义节点添加到 features_all.yaml

