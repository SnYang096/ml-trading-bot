# Gate 规则导出：imodels 蒸馏组合拳方案

## 问题背景

Gate 当前用 `_collect_splits` 从 LightGBM ensemble 提取分裂点生成规则，存在两个核心缺陷：

1. **跨 seed 不一致**：不同 seed → 不同树结构 → 不同分裂点 → 不同 gate 规则（BPC/ME 5/5 seed 一致性 0%）
2. **AND 爆炸**：N 条独立单条件规则各自 deny = AND 逻辑，规则越多 veto 率指数上升（ME gate allow mean 1.50 vs veto mean 5.20，效果为负）

## 方案演进与权衡

### 方案 A：旧方案 — `_collect_splits` 树分裂提取

```
LightGBM(seed=varying) → 遍历所有树 → 统计高频分裂点 → top_n=10 规则 → YAML
```

- ❌ 不同 seed → 不同树 → 不同规则
- ❌ 按频率选分裂点不科学
- ❌ 10 条 AND → 99.3% veto

### 方案 B：sklearn DecisionTree 直接替代

```
DecisionTree(max_depth=3, random_state=42) → export_text → YAML
```

- ✅ 确定性（fixed random_state）
- ✅ 天然 AND+OR（树路径=AND，多路径=OR）
- ✅ 零额外依赖，最简实现
- ⚠️ 探索能力弱于 LightGBM（单棵浅树 vs 100+棵提升树）
- ⚠️ 对于简单 gate 过滤"够用"，但不是最优

### 方案 C：✅ 最优方案 — LightGBM → SHAP → imodels 蒸馏

```
Step 1: LightGBM(seed=42 固定) → 高精度 teacher 模型
Step 2: SHAP 分析 teacher → Top N 关键特征（跨时间稳定）
Step 3: imodels.RuleFitClassifier → 从 teacher 的 predict_proba 蒸馏 → ≤5 条规则
Step 4: 验证蒸馏规则 accuracy vs teacher 差距
        - < 3%：上线规则
        - > 5%：增加特征数或放宽规则限制
Step 5: 导出为 YAML（现有格式兼容）
```

## 方案 C 详细设计

### 核心原则

1. **一个 teacher，一套规则**：Gate LightGBM 固定 seed=42，不受外层 pipeline seed 影响
2. **蒸馏而非重训**：imodels 学习 teacher 的 `predict_proba` 输出，保留 ensemble 学到的复杂模式
3. **YAML 交付物优先**：最终产出是人类可读、可 git diff、可手动调整的 YAML 规则

### 蒸馏 vs 从头训练（关键区别）

```python
# ❌ 从头训练（忽略 teacher，丢失 ensemble 知识）
y = (forward_rr < q30).astype(int)       # 原始标签
imodels_clf.fit(X, y)

# ✅ 蒸馏（保留 teacher 的判别边界）
y_teacher = lgbm_model.predict_proba(X)[:, 1]  # teacher 概率输出
y_distill = (y_teacher > 0.5).astype(int)
imodels_clf.fit(X, y_distill)
```

### 固定 seed 的合理性

- LightGBM 单次训练内部已含 `colsample_bytree` + `bagging` + 100+ 棵树的大规模特征搜索
- 外层多 seed 目的是测策略鲁棒性（entry/sizing/exit），不是帮 gate 探索特征
- Gate 要求确定性（同一数据 → 同一规则），固定 seed=42 正确

### YAML 格式兼容性

**不需要升级 YAML 格式**。RuleFit 输出的规则直接兼容现有 gate.yaml 格式：

**单条件规则**：
```
RuleFit 输出: vpin_ma10 > 0.35 (coef=0.32)
↓
YAML:
- id: gate_vpin_ma10
  when:
    vpin_ma10: { value_gt: 0.35 }
  then: { action: deny }
  comment: "imodels: vpin_ma10 > 0.35 | coef=0.3200"
```

**复合规则（RuleFit 的 `&` 规则）**：
```
RuleFit 输出: vpin_ma10 > 0.35 & evt_var_99 > 0.75 (coef=0.45)
↓
YAML:
- id: gate_vpin_evt_compound
  when:
    all_of:
      - vpin_ma10: { value_gt: 0.35 }
      - evt_var_99: { value_gt: 0.75 }
  then: { action: deny }
  comment: "imodels: vpin_ma10 > 0.35 & evt_var_99 > 0.75 | coef=0.4500"
```

gate 评估器 `_eval_when_vectorized` 已支持 `all_of` (AND) / `any_of` (OR)，零改动。

**复合规则比单条件规则更优**：条件组合在规则内部（vpin高 AND evt高 → deny），
而非规则之间（vpin高 → deny, evt高 → deny），天然避免 AND 爆炸。

### Prefilter 与 Gate 必须分层

**不可合并**。原因：
1. Prefilter 定义训练数据的分布边界（哪些 bar 属于此 archetype），是领域先验
2. Gate 是 archetype 内部的质量过滤（好的 BPC vs 坏的 BPC）
3. 同一 bar 对 BPC 是噪声、对 ME 是信号，混合训练标签互相矛盾
4. 让树模型同时学两层职责，浅树容量不够、深树过拟合

## 通用性

此管线完全通用，不限于 gate：

| 应用场景 | teacher | 蒸馏标签 | 输出 |
|----------|---------|----------|------|
| Gate | LightGBM(bad_trade, seed=42) | P(bad) | deny/allow 规则 YAML |
| Evidence | LightGBM(evidence_signal, seed=42) | predict(X) | evidence 解释规则 YAML |

## 影响范围

| 模块 | 是否受影响 |
|------|-----------|
| `export_lightgbm_rules_to_readme.py` | ✅ 改（蒸馏模式 + 复合规则解析） |
| `train_strategy_pipeline.py` | ✅ 改（Gate/Evidence seed 固定 + 传入 lgbm_model） |
| `optimize_gate_unified.py` | ❌ 不改（接收规则格式不变） |
| gate 评估器 (loader.py) | ❌ 不改（YAML 格式不变） |
| gate 评估器 (backtest_execution_layer.py) | ❌ 不改 |
| `auto_research_pipeline.py` | ❌ 不改 |
| Entry / Execution / PCM | ❌ 无关 |
