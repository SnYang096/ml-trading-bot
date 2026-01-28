# 规则重要性分析 vs 树模型 Feature Group Search

## 📋 概述

本文档对比两种方法：**规则重要性分析（Ablation Study）**和**树模型 Feature Group Search**，阐述它们的本质差异、适用场景和如何结合使用。

**相关文档**：
- [树模型规则导出与维护方法](./树模型规则导出与维护方法.md)
- [架构总览](./ARCHITECTURE.md)
- [项目 README](../../README_CN.md)

---

## 一、本质差异总览

| 维度 | 规则重要性分析 (Ablation) | 树模型 Feature Group Search |
|------|------------------------|--------------------------|
| **目标** | 评估**已有规则**的贡献 | 搜索**最优特征组合** |
| **输入** | 固定的规则集 | 候选特征池 |
| **输出** | 规则重要性排序 | 最优特征子集 |
| **搜索空间** | 2^n (规则数) | 2^m (特征组数) |
| **优化目标** | 找出冗余/负面规则 | 最大化 Sharpe/CV |
| **应用阶段** | 规则系统已建立后 | 特征选择阶段 |
| **思维模式** | **减法**：从已有系统中剔除冗余 | **加法**：主动构建最优系统 |

---

## 二、详细对比

### 2.1 规则重要性分析（Ablation Study）

#### 方法 1：单规则消融

```python
# 目标: 评估每个规则的贡献

# 已有规则集
rules = [
    'fbf_wick_exhaustion',
    'fbf_sr_distance',
    'fbf_vp_skewness',
    'fbf_reflexivity_shd',
    'fbf_jump_risk_high'
]

# Baseline: 所有规则
baseline_sharpe = run_with_rules(rules)  # 1.2

# 逐个移除
for rule in rules:
    without_rule = [r for r in rules if r != rule]
    sharpe = run_with_rules(without_rule)
    importance = baseline_sharpe - sharpe
    print(f"{rule}: {importance}")

# 输出:
# fbf_wick_exhaustion: +0.45 (重要!)
# fbf_sr_distance: +0.38
# fbf_vp_skewness: +0.22
# fbf_reflexivity_shd: +0.05
# fbf_jump_risk_high: -0.12 (负面!应移除)
```

**特点**：
- ✅ 快速（n 次运行）
- ✅ 简单直观
- ❌ 无法发现"缺失的规则"
- ❌ 不考虑规则间的协同效应

#### 方法 2：特征贡献分解

```python
# 目标: 评估每个**特征**(不是规则)的贡献

# 规则可能用到多个特征
rule_feature_map = {
    'fbf_wick_exhaustion': ['range_ratio_5bar', 'wick_exhaustion_score'],
    'fbf_sr_distance': ['sr_distance_normalized'],
    'fbf_vp_skewness': ['vp_skewness'],
}

# 对每个特征,找到用它的所有规则,然后移除这些规则
for feature in all_features:
    rules_using_feature = find_rules_using(feature)
    without_these_rules = [r for r in rules if r not in rules_using_feature]
    sharpe = run_with_rules(without_these_rules)
    contribution = baseline_sharpe - sharpe
    print(f"{feature}: {contribution}")
```

**特点**：
- ✅ 更细粒度（特征级而非规则级）
- ✅ 可以发现"高价值特征"
- ❌ 仍然是"减法"思维，不是搜索
- ❌ 假设现有规则已经覆盖了重要特征

#### 方法 3：Shapley 值（边际贡献）

```python
# 目标: 计算每个规则的**平均边际贡献**

# 考虑所有可能的规则组合
for subset in all_combinations(rules):
    for rule in rules_not_in_subset:
        # 加入 rule 前的 Sharpe
        sharpe_before = run_with_rules(subset)
        
        # 加入 rule 后的 Sharpe
        sharpe_after = run_with_rules(subset + [rule])
        
        # 边际贡献
        marginal = sharpe_after - sharpe_before
        
        # 累加到 Shapley 值
        shapley[rule] += weight * marginal

# 输出:
# fbf_wick_exhaustion: 0.38 (平均边际贡献)
# fbf_sr_distance: 0.31
# ...
```

**特点**：
- ✅ 最严谨（考虑了所有组合）
- ✅ 公平归因（协同效应被分摊）
- ❌ 计算量巨大（2^n）
- ❌ 仍然是"评估已有规则"，不是搜索新特征

**详细方法说明**：参见[树模型规则导出与维护方法](./树模型规则导出与维护方法.md#三规则类特征重要性判断)

---

### 2.2 树模型 Feature Group Search

#### 核心机制

```python
# 目标: 从候选特征池中**搜索**最优组合

# 候选特征组
groups = {
    'kline_core': ['macd_f', 'rsi_f', 'atr_f'],
    'sr_structure': ['poc_hal_f', 'sqs_f'],
    'volume_profile': ['vp_volatility_f'],
    'footprint': ['fp_imbalance_f'],
    'vpin': ['vpin_features_f'],
    # ... 20个候选组
}

# 搜索算法 (例如: Greedy Forward Selection)
selected = []
remaining = list(groups.keys())

while len(selected) < max_steps:
    best_group = None
    best_sharpe = current_sharpe
    
    # 尝试加入每个剩余组
    for group in remaining:
        candidate_features = base_features + flatten([groups[g] for g in selected + [group]])
        sharpe = train_and_evaluate(candidate_features)
        
        if sharpe > best_sharpe:
            best_group = group
            best_sharpe = sharpe
    
    if best_group is None:
        break
    
    selected.append(best_group)
    remaining.remove(best_group)
    current_sharpe = best_sharpe

# 输出:
# selected = ['sr_structure', 'volume_profile', 'vpin']
# final_sharpe = 1.8 (相比baseline 0.9提升)
```

**特点**：
- ✅ **主动搜索** - 从候选池中找最优组合
- ✅ **发现新特征** - 可能选出你没想到的组合
- ✅ **全局优化** - 最大化目标函数（Sharpe）
- ❌ 计算量大（需要多次训练）
- ❌ 可能过拟合到训练集

**实现位置**：`src/time_series_model/diagnostics/feature_group_search.py`

---

## 三、核心区别总结

### 区别 1：评估 vs 搜索

```
规则重要性分析 (Ablation):
  输入: 已有的规则集 R = {r1, r2, r3, r4, r5}
  问题: "这5个规则中,哪些重要?"
  方法: 逐个移除,观察影响
  输出: 重要性排序: r1 > r2 > r3 > r4 > r5
  
  → 这是"减法"思维: 从已有系统中剔除冗余

Feature Group Search:
  输入: 候选特征池 F = {f1, f2, f3, ..., f20}
  问题: "从这20个特征组中,选哪几个能最大化Sharpe?"
  方法: 前向搜索,逐步加入最优组
  输出: 最优子集: {f2, f5, f12}
  
  → 这是"加法"思维: 主动构建最优系统
```

### 区别 2：假设前提不同

```yaml
规则重要性分析:
  假设: "当前规则集基本合理,只需微调"
  适用: 系统已经搭建,需要优化
  目标: 移除负面规则,保留核心规则
  
  例子:
    - 当前FBF有10个规则
    - 消融发现'fbf_jump_risk_high'是负面的
    - 移除后Sharpe从1.2提升到1.32

Feature Group Search:
  假设: "不知道哪些特征有用,需要探索"
  适用: 特征池很大,需要选择
  目标: 找到最优特征组合
  
  例子:
    - 候选池有50个特征组
    - 搜索发现只需要其中5个
    - Sharpe从0 (baseline)提升到1.8
```

### 区别 3：搜索空间

```python
# 规则重要性分析
规则数: 5
搜索空间: 
  - 单规则消融: 5次运行
  - Shapley值: 2^5 = 32次运行
  
# Feature Group Search
特征组数: 20
搜索空间:
  - Greedy: 最多20 * max_steps次运行
  - Beam (width=3): 最多60次运行
  - Exhaustive: 2^20 = 1,048,576次 (不现实)
```

### 区别 4：发现能力

```yaml
规则重要性分析:
  ❌ 不能发现"缺失的规则"
  ✅ 可以发现"冗余的规则"
  ✅ 可以发现"负面的规则"
  
  例子:
    - 只能在现有5个规则中评估
    - 无法发现"其实还应该加一个vp_entropy规则"

Feature Group Search:
  ✅ 可以发现"有用的新特征"
  ✅ 可以发现"特征组合的协同效应"
  ❌ 不直接告诉你"为什么某个特征重要"
  
  例子:
    - 从50个候选中发现vp_skewness有用
    - 发现vp_skewness + sr_distance组合效果好
```

---

## 四、实战场景对比

### 场景 1：你已经有 FBF 规则系统

**用规则重要性分析**：
```python
# 当前FBF有这些规则
fbf_rules = [
    'fbf_wick_exhaustion',
    'fbf_sr_distance',
    'fbf_vp_skewness',
    'fbf_reflexivity_shd',
    'fbf_jump_risk_high',
    'fbf_vpin_spike'
]

# 问题: "这6个规则中,哪些真正有用?"
# 方法: 消融实验
ablation_results = analyze_rule_importance(fbf_rules)

# 发现:
# - fbf_jump_risk_high是负面的 → 移除
# - fbf_vpin_spike贡献很小 → 考虑移除
# - fbf_wick_exhaustion最重要 → 保留且可能增强
```

**不适合 Feature Group Search，因为**：
- 规则已经定义好了
- 不是从特征池中选择，而是评估现有逻辑

---

### 场景 2：你要为 SR Reversal 策略选特征

**用 Feature Group Search**：
```python
# 候选特征池
candidate_groups = {
    'poc_hal': ['poc_hal_features_f'],
    'atr': ['atr_f'],
    'vp_volatility': ['volume_profile_volatility_features_f'],
    'vp_skewness_singleton': ['vp_skewness'],  # 单列
    'vp_entropy_singleton': ['vp_entropy'],
    'macd': ['macd_f'],
    'rsi': ['rsi_f'],
    # ... 50个候选
}

# 问题: "从这50个中,选哪几个能最大化Sharpe?"
# 方法: Greedy Forward Search
search_result = feature_group_search(
    candidate_groups,
    objective='Sharpe_mean',
    max_steps=10
)

# 发现:
# selected = ['poc_hal', 'atr', 'vp_volatility']
# Sharpe从0.5 (只用poc_hal)提升到1.8
```

**不适合规则重要性分析，因为**：
- 还没有规则，只有特征候选
- 需要主动搜索最优组合

---

### 场景 3：优化已有 FBF 的特征选择

**先用 Feature Group Search，再用规则重要性分析**：

```python
# Step 1: Feature Group Search (找最优特征)
# 假设FBF原本用了10个特征组
original_features = ['poc_hal', 'atr', 'vp_volatility', 'macd', 'rsi', ...]

# 搜索发现:只需要其中3个
search_result = feature_group_search(
    candidates=all_feature_groups,
    base_features=['poc_hal', 'atr']  # 必需的
)
# → selected = ['poc_hal', 'atr', 'vp_volatility']

# Step 2: 用这3个特征重新设计规则
fbf_rules_v2 = [
    'fbf_sr_distance',      # 基于poc_hal
    'fbf_wick_exhaustion',  # 基于atr
    'fbf_vp_high_entropy',  # 基于vp_volatility (新增!)
]

# Step 3: 规则重要性分析 (优化规则逻辑)
ablation_results = analyze_rule_importance(fbf_rules_v2)
# → 发现所有3个规则都有正贡献
```

---

## 五、何时用哪个？

### ✅ 用规则重要性分析（Ablation）

```yaml
场景:
  - 规则系统已经建立
  - 想知道哪些规则有用/冗余
  - 想优化现有规则集
  - 想做归因分析 (哪个语义失效了)

优势:
  - 快速 (n次运行)
  - 可解释性强
  - 直接指导规则修改

劣势:
  - 无法发现新规则
  - 局限于现有规则集
```

**详细方法**：参见[树模型规则导出与维护方法](./树模型规则导出与维护方法.md#三规则类特征重要性判断)

### ✅ 用 Feature Group Search

```yaml
场景:
  - 特征池很大 (50+ 特征组)
  - 不确定哪些特征有用
  - 想最大化某个目标 (Sharpe)
  - 想发现特征组合的协同效应

优势:
  - 主动搜索最优组合
  - 可以发现意外的有效特征
  - 全局优化

劣势:
  - 计算量大
  - 可能过拟合
  - 不直接告诉你"为什么"
```

**实现位置**：`src/time_series_model/diagnostics/feature_group_search.py`

---

## 六、结合使用的工作流 ⭐⭐⭐

```
阶段1: Feature Group Search
  ├─ 输入: 50个候选特征组
  ├─ 输出: 最优5个特征组
  └─ 目标: 找到"哪些特征有用"

阶段2: 规则设计
  ├─ 基于这5个特征组设计规则
  └─ 例如: vp_skewness → fbf_vp_skewness_confirms规则

阶段3: 规则重要性分析
  ├─ 输入: 设计的10个规则
  ├─ 输出: 规则重要性排序
  └─ 目标: 优化规则逻辑

阶段4: 实盘监控
  ├─ 用规则重要性分析定位问题
  └─ 例如: "vp_skewness规则失效了" → 调整或移除
```

**完整流程**：参见[树模型规则导出与维护方法](./树模型规则导出与维护方法.md)

---

## 七、Feature Group Search 的高级特性

### 7.1 Successive Halving（多阶段预筛选）

```python
# 规则重要性分析没有这个!
# Feature Group Search特有:

stages = [1, 3, 5]  # 用1个seed → 3个seed → 5个seed

# Stage 1 (1 seed): 快速筛选,保留top 30%
candidates = 50
survivors = filter_top_30_percent(candidates, seeds=1)  # 15个

# Stage 2 (3 seeds): 中度验证,保留top 50%
survivors = filter_top_50_percent(survivors, seeds=3)  # 7个

# Stage 3 (5 seeds): 完整验证
final = evaluate_all(survivors, seeds=5)  # 7个
```

**优势**：避免在差的候选上浪费计算

### 7.2 Beam Search（保留多条路径）

```python
# 规则重要性分析没有这个!
# Feature Group Search特有:

# 每一步保留top-K路径
beam = [
    ['poc_hal'],
    ['atr'],
    ['vp_volatility']
]

# 下一步,每条路径都尝试加入新特征
next_beam = []
for path in beam:
    for new_feature in remaining:
        candidate = path + [new_feature]
        sharpe = evaluate(candidate)
        next_beam.append((candidate, sharpe))

# 保留top-K
beam = top_k(next_beam, k=3)
```

**优势**：避免贪心算法的局部最优

### 7.3 SFFS（前向+后向）

```python
# 规则重要性分析只有"后向"(移除规则)
# Feature Group Search有"前向+后向":

# 前向: 加入最优特征
selected.append(best_feature)

# 后向: 尝试移除已选特征 (如果移除后更好)
for feature in selected:
    if sharpe_without(feature) > current_sharpe:
        selected.remove(feature)
```

**优势**：纠正早期的错误选择

**实现细节**：参见 `src/time_series_model/diagnostics/feature_group_search.py`

---

## 八、总结对比表

| 方法 | 本质 | 适用场景 | 输出 | 计算复杂度 |
|------|------|---------|------|-----------|
| **规则重要性分析** | 评估已有规则 | 规则系统已建立，需优化 | 规则排序，移除建议 | O(n) - O(2^n) |
| **Feature Group Search** | 搜索最优特征 | 特征池大，需选择 | 最优特征子集 | O(m²) - O(2^m) |

**你的情况**：
- 如果 archetype 规则已经定义好 → 用规则重要性分析
- 如果要为 archetype 选择特征 → 用 Feature Group Search
- **最佳实践**：先 Search 选特征，再 Ablation 优化规则

---

## 九、相关文档

- [树模型规则导出与维护方法](./树模型规则导出与维护方法.md) - 详细的规则重要性分析方法
- [架构总览](./ARCHITECTURE.md) - 系统整体架构
- [项目 README](../../README_CN.md) - 项目总览
- [Feature Group Search 实现](../../src/time_series_model/diagnostics/feature_group_search.py) - 代码实现

---

**最后更新**: 2026-01-28
