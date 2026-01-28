# invert_features 在 feature-group-search 中的使用分析

## 🎯 核心问题

**Pool B 的 `invert_features` 是否在 feature-group-search 中被正确使用？会不会只搜索了正向特征？**

---

## 一、两种反向验证机制

### 机制 1：`poolb_invcol__<col>` groups（输出列作为独立候选）

**代码位置**：`feature_group_search.py:2255-2268`

```python
# Add singleton groups for Pool-B inverted OUTPUT columns
for col in pool_inv:
    c = str(col).strip()
    if not c:
        continue
    key = f"poolb_invcol__{c}"
    groups[key] = [c]  # 输出列名作为 group 内容
```

**工作原理**：
1. Pool B 的每个 `invert_features` 列会被注入为 `poolb_invcol__<col>` group
2. 这些 groups 会被加入到搜索空间
3. 如果被选中，`requested_features` 会包含该输出列名
4. `StrategyFeatureLoader` 会通过 `output_col_to_feature` 映射找到父特征节点并计算

**优点**：
- ✅ 输出列作为**独立候选**，可以被单独选中
- ✅ 如果被选中，会自动计算对应的特征节点

**限制**：
- ⚠️ 如果这些 groups 在 prefilter 阶段就被淘汰，就不会进行反向验证
- ⚠️ 需要对应的特征节点存在（通过 `output_col_to_feature` 映射）

---

### 机制 2：对已选中 group 的反向验证（A/B 测试）

**代码位置**：`feature_group_search.py:757-817`

```python
# 对于每个 group，检查其内容是否在 invert_candidates 中
inv_cols_for_group = _stable_dedup(
    [
        str(x)
        for x in ([str(g)] + [str(xx) for xx in (groups.get(g) or [])])
        if str(x).strip() in inv_cand_set
    ]
)

# 如果匹配，尝试反向版本
if try_invert:
    # 先评估 raw
    score = evaluate(raw_version)
    
    # 再评估 inverted
    inv_score = evaluate(inverted_version)
    
    # 根据策略选择
    if inv_score > score:
        picked_inverted = True
```

**工作原理**：
1. 对于**每个已选中的 group**，检查 group 名称或内容是否在 `invert_candidates` 中
2. 如果匹配，会进行 A/B 测试：
   - Raw 版本：`invert_features = base_inv`
   - Inverted 版本：`invert_features = base_inv + inv_cols_for_group`
3. 根据策略（conservative/all）选择更好的版本

**优点**：
- ✅ 对已选中的 group 进行精确的反向验证
- ✅ 可以验证反向是否真的有利

**限制**：
- ❌ **只对已选中的 group 进行**，不对所有候选进行
- ❌ 如果 group 是特征节点（如 `macd_f`），而 `invert_candidates` 是输出列名（如 `macd_signal`），它们不会直接匹配
  - 因为 `groups.get('macd_f')` 返回的是 `['macd_f']`（节点名），不是 `['macd_signal']`（输出列名）

---

## 二、关键问题分析

### 问题 1：`poolb_invcol__<col>` groups 是否被创建？

**检查方法**：
```python
# 检查搜索结果中是否有 poolb_invcol__ 候选
poolb_invcol_candidates = [g for g in all_candidates if g.startswith('poolb_invcol__')]
```

**你的情况（compression_breakout）**：
- ❌ 搜索结果中**没有** `poolb_invcol__` groups
- ❌ 第一个 stage 中也没有这些候选

**可能原因**：
1. `--pool-b-yaml` 参数没有传递，或者文件不存在
2. `poolb_invcol__` groups 被创建了，但在某个阶段被过滤掉了
3. 或者这些 groups 根本没有被创建

---

### 问题 2：反向验证是否对特征节点 groups 生效？

**关键代码**（line 757-763）：
```python
inv_cols_for_group = _stable_dedup(
    [
        str(x)
        for x in ([str(g)] + [str(xx) for xx in (groups.get(g) or [])])
        if str(x).strip() in inv_cand_set
    ]
)
```

**匹配逻辑**：
- 检查 group 名称 `g`（如 `macd_f`）
- 检查 group 内容 `groups.get(g)`（如 `['macd_f']`）
- 如果这些值在 `inv_cand_set` 中，才会进行反向验证

**问题**：
- 如果 group 是 `macd_f`（特征节点），内容是 `['macd_f']`
- 而 `invert_candidates` 是 `['macd_signal']`（输出列名）
- 那么 `'macd_f'` 不在 `inv_cand_set` 中，**不会匹配**！

**结论**：
- ❌ **反向验证对特征节点 groups 不生效**（除非节点名恰好等于某个输出列名）
- ✅ 反向验证只对**输出列 singleton groups** 生效（如 `poolb_invcol__macd_signal`）

---

## 三、你的情况：compression_breakout

### 实际状态

1. **Pool B 的 `invert_features`**：
   - 17 个输出列名（如 `macd_signal`, `dtw_bull_flag_dist_w20` 等）

2. **Feature-group-search 选中的 groups**：
   - `poolb__liquidity_void_f`
   - `kline_core__volume_ratio_f`
   - `kline_core__rsi_f`

3. **反向验证结果**：
   - ❌ 没有 `poolb_invcol__` groups 被选中
   - ❌ 没有反向验证记录（`invert_by_group` 为空）

### 问题根源

**`poolb_invcol__<col>` groups 没有被选中，可能的原因**：

1. **这些 groups 没有被创建**
   - 检查：`--pool-b-yaml` 是否传递
   - 检查：Pool B YAML 是否存在

2. **这些 groups 在 prefilter 阶段就被淘汰了**
   - 如果 `poolb_invcol__macd_signal` 的分数很低，会在 successive halving 阶段被淘汰
   - 淘汰后就不会进行反向验证

3. **反向验证的限制**
   - 反向验证只对**已选中的 group** 进行
   - 如果 group 没有被选中，就不会进行反向验证

---

## 四、验证方法

### 方法 1：检查 Pool B YAML 是否被读取

```python
# 检查运行时的参数
# 查看 feature_group_search_result.json 中的 metadata
result = json.load(open('results/.../feature_group_search_result.json'))
meta = result.get('metadata', {})
print(f"pool_b_yaml: {meta.get('pool_b_yaml')}")
print(f"invert_candidates_yaml: {meta.get('invert_candidates_yaml')}")
```

### 方法 2：检查 groups 是否被创建

在 `feature_group_search.py` 的 `main()` 函数中（line 2268 之后）添加：

```python
# 验证 poolb_invcol groups 是否被创建
poolb_invcol_created = [k for k in groups.keys() if k.startswith('poolb_invcol__')]
print(f"✅ Created {len(poolb_invcol_created)} poolb_invcol groups")
if len(poolb_invcol_created) == 0 and len(pool_inv) > 0:
    print(f"⚠️  WARNING: No poolb_invcol groups created despite {len(pool_inv)} invert_features!")
```

### 方法 3：检查反向验证是否执行

```python
# 检查 prefilter 阶段的 rows
# 查看是否有 inv_score 不为 None 的记录
rows = prefilter['stage_tables'][-1]['rows']
inverted_tested = [r for r in rows if r.get('inv_score') is not None]
print(f"反向验证执行次数: {len(inverted_tested)}")
```

---

## 五、潜在问题

### ⚠️ 问题 1：反向验证可能没有对所有候选执行

**当前逻辑**：
- 反向验证只对**已选中的 group** 进行
- 如果 `poolb_invcol__macd_signal` 在 prefilter 阶段就被淘汰，就不会进行反向验证

**影响**：
- 可能丢失有用的反向特征
- 如果反向版本比正向版本好，但因为正向版本分数低被淘汰，反向版本永远不会被测试

### ⚠️ 问题 2：特征节点 groups 的反向验证不生效

**当前逻辑**：
- 如果 group 是特征节点（如 `macd_f`），内容是 `['macd_f']`
- 而 `invert_candidates` 是输出列名（如 `['macd_signal']`）
- 它们不会匹配，不会进行反向验证

**影响**：
- 即使 `macd_f` 被选中，如果 `macd_signal` 需要反向，也不会自动进行反向验证
- 需要手动在 `invert_features` 中指定

---

## 六、建议的改进

### 改进 1：对特征节点 groups 也进行反向验证

```python
# 改进 inv_cols_for_group 的匹配逻辑
# 不仅检查 group 内容，还检查该特征节点的所有输出列

def _get_invert_candidates_for_group(group_name, group_content, invert_candidates, feature_deps):
    """
    获取该 group 对应的反向候选列
    """
    inv_cols = []
    
    # 1. 检查 group 名称和内容
    for x in [group_name] + group_content:
        if x in invert_candidates:
            inv_cols.append(x)
    
    # 2. 如果 group 内容是特征节点，检查其输出列
    for feat_node in group_content:
        if feat_node in feature_deps:
            output_cols = feature_deps[feat_node].get('output_columns', [])
            for col in output_cols:
                if col in invert_candidates:
                    inv_cols.append(col)
    
    return inv_cols
```

### 改进 2：确保 `poolb_invcol__` groups 被创建

在 `main()` 函数中添加验证（line 2268 之后）：

```python
# 验证 poolb_invcol groups 是否被创建
poolb_invcol_created = [k for k in groups.keys() if k.startswith('poolb_invcol__')]
print(f"✅ Created {len(poolb_invcol_created)} poolb_invcol groups from {len(pool_inv)} invert_features")
if len(poolb_invcol_created) == 0 and len(pool_inv) > 0:
    print(f"⚠️  WARNING: No poolb_invcol groups created despite {len(pool_inv)} invert_features!")
```

---

## 七、立即验证步骤

### Step 1：检查 Pool B YAML 是否被读取

```bash
# 检查搜索结果中的 metadata
python3 -c "
import json
result = json.load(open('results/feature_group_search/compression_breakout_pipeline_poolb_semantic_20260108_best_abc_A/feature_group_search_result.json'))
meta = result.get('metadata', {})
print('pool_b_yaml:', meta.get('pool_b_yaml'))
print('invert_candidates_yaml:', meta.get('invert_candidates_yaml'))
"
```

### Step 2：重新运行并检查 groups

在 `feature_group_search.py` 的 `main()` 函数中添加临时日志，或者检查运行日志。

### Step 3：使用 `--invert-eval all` 重新运行

```bash
# 使用 --invert-eval all 确保反向验证更激进
mlbot diagnose feature-group-search \
  --base-strategy-config config/strategies/compression_breakout \
  --pool-b-yaml results/pools/compression_breakout/pool_b/20260108_best_abc/features_pool_b.yaml \
  --invert-candidates-yaml results/pools/compression_breakout/pool_b/20260108_best_abc/features_pool_b.yaml \
  --invert-eval all \
  ...
```

---

## 八、总结

### ✅ 确认：反向验证机制存在

1. **`poolb_invcol__<col>` groups** 会被注入到搜索空间（如果 Pool B YAML 被正确读取）
2. **反向验证逻辑** 会对匹配的 groups 进行 A/B 测试

### ⚠️ 潜在问题

1. **`poolb_invcol__` groups 可能没有被创建或选中**
   - 需要验证 Pool B YAML 是否被正确读取
   - 需要验证这些 groups 是否在搜索空间中

2. **反向验证只对已选中的 group 进行**
   - 如果 group 在 prefilter 阶段就被淘汰，不会进行反向验证
   - 可能丢失有用的反向特征

3. **特征节点 groups 的反向验证不生效**
   - 如果 group 是特征节点（如 `macd_f`），而 `invert_candidates` 是输出列名（如 `macd_signal`），不会匹配
   - 需要手动在 `invert_features` 中指定

### 🎯 建议

1. **立即验证**：检查搜索结果中是否有 `poolb_invcol__` groups
2. **如果缺失**：检查运行参数，确保 `--pool-b-yaml` 和 `--invert-candidates-yaml` 被传递
3. **如果存在但没被选中**：考虑使用 `--invert-eval all` 更激进地测试反向版本
4. **长期改进**：考虑改进反向验证逻辑，对特征节点 groups 也检查其输出列

---

**最后更新**: 2026-01-28
