# VPIN 特征列表和策略选择建议

## 一、VPIN 特征完整列表

`extract_order_flow_features` 函数实际生成的特征（共 **21 个**）：

### 基础特征（2个）
1. `vpin` - 基础 VPIN 值（0-1范围）
2. `vpin_signed_imbalance` - 方向性信号（-1到1，正数=买方压力，负数=卖方压力）

### 滚动统计特征（6个）
3. `vpin_ma5` - 5 期移动平均
4. `vpin_ma10` - 10 期移动平均
5. `vpin_ma20` - 20 期移动平均
6. `vpin_max5` - 5 期最大值
7. `vpin_max10` - 10 期最大值
8. `vpin_max20` - 20 期最大值

### 变化率特征（2个）
9. `vpin_change` - 一阶差分
10. `vpin_change_pct` - 百分比变化率

### Z-score 特征（2个）
11. `vpin_zscore_20` - 20 期 Z-score（识别异常高的订单流不平衡）
12. `vpin_zscore_50` - 50 期 Z-score

### 分位数排名特征（2个）
13. `vpin_quantile_rank_20` - 20 期分位数排名（0~1，相对位置）
14. `vpin_quantile_rank_50` - 50 期分位数排名

### 波动率特征（2个）
15. `vpin_volatility_10` - 10 期波动率（衡量订单流稳定性）
16. `vpin_volatility_20` - 20 期波动率

### Spike 标志特征（2个）
17. `vpin_spike_flag_20` - 20 期异常突增标志（基于 MAD）
18. `vpin_spike_flag_50` - 50 期异常突增标志

### 动量特征（1个）
19. `vpin_momentum` - VPIN 动量（vpin_ma5 - vpin_ma20，捕捉不平衡加速）

### Signed Imbalance Z-score 特征（2个）
20. `vpin_signed_imbalance_zscore_20` - 20 期 Signed Imbalance Z-score
21. `vpin_signed_imbalance_zscore_50` - 50 期 Signed Imbalance Z-score

---

## 二、当前配置文件问题

### 问题：`output_columns` 不完整

当前 `config/feature_dependencies.yaml` 中的 `vpin_features` 定义：

```yaml
output_columns: ["vpin", "vpin_ma5", "vpin_ma10", "vpin_ma20", "vpin_max5", "vpin_max10", "vpin_max20", "vpin_change", "vpin_change_pct"]
```

**只列出了 9 个特征，但实际生成了 21 个！**

这意味着：
- 如果特征加载器严格按照 `output_columns` 过滤，会丢失 12 个特征
- 如果特征加载器返回所有生成的列，则 `output_columns` 配置无效

---

## 三、策略特征需求分析

### SR 反转策略
**核心需求**：识别 SR 区域的反转信号，需要：
- ✅ `vpin` - 基础不平衡度
- ✅ `vpin_signed_imbalance` - 方向性（买方/卖方压力）
- ✅ `vpin_spike_flag_20` - 异常突增（反转前兆）
- ✅ `vpin_zscore_20` - 异常检测
- ✅ `vpin_momentum` - 不平衡加速
- ⚠️ `vpin_ma*` - 平滑信号（可选）
- ⚠️ `vpin_volatility_*` - 稳定性（可选）

**推荐特征**（8-10个）：
- 核心：`vpin`, `vpin_signed_imbalance`, `vpin_spike_flag_20`, `vpin_zscore_20`, `vpin_momentum`
- 辅助：`vpin_ma20`, `vpin_volatility_20`, `vpin_signed_imbalance_zscore_20`

### SR 突破策略
**核心需求**：识别突破质量和真假突破，需要：
- ✅ `vpin` - 基础不平衡度
- ✅ `vpin_signed_imbalance` - 方向性（突破方向验证）
- ✅ `vpin_spike_flag_20` - 突破时的订单流突增
- ✅ `vpin_momentum` - 突破动量
- ✅ `vpin_zscore_20` - 异常检测
- ⚠️ `vpin_change` - 变化率（捕捉突增）
- ⚠️ `vpin_max*` - 峰值（突破强度）

**推荐特征**（8-10个）：
- 核心：`vpin`, `vpin_signed_imbalance`, `vpin_spike_flag_20`, `vpin_momentum`, `vpin_zscore_20`
- 辅助：`vpin_change`, `vpin_max20`, `vpin_signed_imbalance_zscore_20`

### 压缩区突破策略
**核心需求**：压缩区突破时的订单流验证，需要：
- ✅ `vpin` - 基础不平衡度
- ✅ `vpin_signed_imbalance` - 方向性
- ✅ `vpin_momentum` - 突破动量
- ✅ `vpin_spike_flag_20` - 突破突增
- ⚠️ `vpin_volatility_*` - 压缩期稳定性 vs 突破期突增
- ⚠️ `vpin_quantile_rank_*` - 相对位置（压缩 vs 突破）

**推荐特征**（7-9个）：
- 核心：`vpin`, `vpin_signed_imbalance`, `vpin_momentum`, `vpin_spike_flag_20`
- 辅助：`vpin_volatility_20`, `vpin_quantile_rank_20`, `vpin_zscore_20`

### 趋势跟踪策略
**核心需求**：趋势中的订单流验证和趋势质量，需要：
- ✅ `vpin` - 基础不平衡度
- ✅ `vpin_signed_imbalance` - 方向性（与趋势对齐）
- ✅ `vpin_momentum` - 趋势动量
- ✅ `vpin_volatility_20` - 趋势稳定性（低波动=健康趋势）
- ✅ `vpin_quantile_rank_50` - 长期相对位置
- ⚠️ `vpin_ma*` - 平滑趋势信号
- ⚠️ `vpin_signed_imbalance_zscore_*` - 极端买卖压力

**推荐特征**（8-10个）：
- 核心：`vpin`, `vpin_signed_imbalance`, `vpin_momentum`, `vpin_volatility_20`, `vpin_quantile_rank_50`
- 辅助：`vpin_ma20`, `vpin_signed_imbalance_zscore_50`, `vpin_zscore_50`

---

## 四、建议方案

### 方案 A：全部加载（推荐）✅

**优点**：
- ✅ 简单：无需维护策略特定的特征列表
- ✅ 灵活：模型可以自动选择有用特征
- ✅ 统一：所有策略使用相同的 VPIN 特征集
- ✅ 未来扩展：新增特征自动可用

**缺点**：
- ⚠️ 特征数量多（21个），可能增加模型复杂度
- ⚠️ 部分特征可能对某些策略无用

**适用场景**：
- 使用特征选择（如 XGBoost 的 feature_importance）
- 有足够的计算资源
- 希望模型自动发现特征重要性

### 方案 B：按策略选择（精细控制）

**优点**：
- ✅ 特征数量少，模型更专注
- ✅ 减少噪声特征
- ✅ 更快的训练和推理

**缺点**：
- ❌ 需要维护策略特定的特征列表
- ❌ 可能遗漏有用特征
- ❌ 配置复杂

**适用场景**：
- 计算资源有限
- 需要严格控制特征数量
- 对特征有明确的理论预期

---

## 五、最终推荐

### 推荐：方案 A（全部加载）+ 更新配置文件

**理由**：
1. **特征选择由模型完成**：XGBoost/CatBoost/LightGBM 都有内置的特征重要性评估
2. **避免遗漏**：某些看似不相关的特征可能在特征交互中发挥作用
3. **维护成本低**：无需为每个策略维护特征列表
4. **统一管理**：所有策略使用相同的 VPIN 特征集，便于对比

**需要做的**：
1. 更新 `config/feature_dependencies.yaml` 中的 `output_columns`，包含所有 21 个特征
2. 或者移除 `output_columns` 限制，让系统返回所有生成的列

---

## 六、特征重要性参考

基于 VPIN 理论和实践经验，特征重要性排序：

### 高重要性（所有策略都需要）
1. `vpin` - 基础指标
2. `vpin_signed_imbalance` - 方向性信号
3. `vpin_momentum` - 动量信号
4. `vpin_spike_flag_20` - 异常检测

### 中重要性（策略特定）
5. `vpin_zscore_20` - 异常检测（SR 反转、SR 突破）
6. `vpin_volatility_20` - 稳定性（趋势跟踪、压缩区突破）
7. `vpin_quantile_rank_20/50` - 相对位置（趋势跟踪）
8. `vpin_signed_imbalance_zscore_20` - 极端压力（所有策略）

### 低重要性（辅助特征）
9. `vpin_ma*` - 平滑信号
10. `vpin_max*` - 峰值
11. `vpin_change*` - 变化率
12. `vpin_volatility_10` - 短期波动率

---

## 七、实施建议

### 立即行动
1. **更新配置文件**：将 `output_columns` 更新为完整的 21 个特征列表
2. **保持全部加载**：所有策略使用 `vpin_features`，不进行筛选
3. **依赖模型选择**：让 XGBoost/CatBoost/LightGBM 自动选择有用特征

### 后续优化（可选）
如果发现某些特征确实无用（通过特征重要性分析），可以考虑：
- 在特征选择器中过滤（`select_*_features` 函数）
- 或创建精简版 `vpin_features_minimal`（只包含核心特征）

---

## 八、配置文件更新示例

```yaml
vpin_features:
  module: enhanced
  compute_func: extract_order_flow_features
  dependencies: []
  required_columns: ["open", "close", "high", "low", "volume"]
  output_columns:
    # 基础特征
    - "vpin"
    - "vpin_signed_imbalance"
    # 滚动统计
    - "vpin_ma5"
    - "vpin_ma10"
    - "vpin_ma20"
    - "vpin_max5"
    - "vpin_max10"
    - "vpin_max20"
    # 变化率
    - "vpin_change"
    - "vpin_change_pct"
    # Z-score
    - "vpin_zscore_20"
    - "vpin_zscore_50"
    # 分位数排名
    - "vpin_quantile_rank_20"
    - "vpin_quantile_rank_50"
    # 波动率
    - "vpin_volatility_10"
    - "vpin_volatility_20"
    # Spike 标志
    - "vpin_spike_flag_20"
    - "vpin_spike_flag_50"
    # 动量
    - "vpin_momentum"
    # Signed Imbalance Z-score
    - "vpin_signed_imbalance_zscore_20"
    - "vpin_signed_imbalance_zscore_50"
  category: order_flow
  description: "VPIN (Volume-Synchronized Probability of Informed Trading) - 完整特征集（21个特征）"
  compute_params:
    vpin_n_buckets: 50
    vpin_adaptive: true
  run_sequential: true
  pass_full_df: true
```

