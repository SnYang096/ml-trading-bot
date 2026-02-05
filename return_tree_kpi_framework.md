# Return Tree 语义发现能力量化评估框架

> **核心 KPI**: 语义发现能力（而非 Sharpe Ratio）
> 
> **目标**: 识别"好交易中，哪些特征让 RR 更大"的交易逻辑
> 
> **创建时间**: 2026-02-05

---

## 📊 评估框架总览

Return Tree 的价值在于**发现可解释的交易规律**，而非直接产生收益。因此评估需要结合**自动化指标** + **人工审查**。

| 评估维度 | 自动化程度 | 时间成本 | 关键指标 |
|---------|-----------|---------|---------|
| **特征重要性分布** | 100% | 1分钟 | Top10占比 30-60% ✅ |
| **预测有效性** | 100% | 2分钟 | Spearman > 0.6 ✅ |
| **泄露检测** | 80% | 2分钟 | `_symbol` 不在 top 20 |
| **Evidence转换率** | 20% | 30分钟 | 可理解规则 > 40% ✅ |
| **最终验证** | 0% | N/A | 接入完整pipeline看Sharpe |

---

## 🎯 第一阶段：快速筛查（5分钟）

### 1. 特征重要性集中度（自动）

**目的**: 检查模型是否在学有意义的信号，而非噪声

```python
# 计算 top 10 特征的累计重要性占比
top10_importance_ratio = sum(top_10_importances) / total_importance

# 评估标准：
# - > 0.7  → 过度集中，可能过拟合 ❌
# - 0.3-0.6 → 健康分布 ✅
# - < 0.2  → 过于分散，模型无主导逻辑 ⚠️
```

**快速检查命令**：
```bash
cat results/train_final_*/bpc/results.json | \
  python3 -c "import json,sys; d=json.load(sys.stdin); \
  fi=list(d['feature_importance'].values())[:10]; \
  print(f'Top10占比: {sum(fi)/sum(d[\"feature_importance\"].values()):.1%}')"
```

---

### 2. 特征类型分布（半自动）

**目的**: 判断模型是否在学交易逻辑，而非数据泄露

```python
# 分类 top 20 特征
categories = {
    "momentum": 0,      # 动量类（如 momentum_4h, price_change_pct）
    "liquidity": 0,     # 流动性（如 bid_ask_spread, volume_imbalance）
    "regime": 0,        # 市场状态（如 volatility_regime, trend_strength）
    "technical": 0,     # 技术指标（如 rsi, macd）
    "leak_risk": 0,     # 泄露风险（如 _symbol, timestamp 相关）
    "interaction": 0,   # 复杂交互项（难以解释）
}

# 评估标准：
# - leak_risk > 3 → 模型在学数据泄露 ❌
# - interaction > 10 → 过度依赖复杂特征，不可解释 ❌
# - 其他类型均衡分布 → 学到多维交易逻辑 ✅
```

**实现方式**: 写个脚本自动分类特征名（基于正则匹配）

---

### 3. 预测值分位数一致性（自动）⭐

**目的**: 验证模型预测是否真的与实际 RR 相关

```python
# 将 holdout 集预测值分成 5 档
df['pred_quintile'] = pd.qcut(df['pred'], q=5, labels=[1,2,3,4,5])

# 计算每档的实际平均 forward_rr
actual_rr_by_quintile = df.groupby('pred_quintile')['forward_rr'].mean()

# 评估标准：
# - Q5 > Q4 > Q3 > Q2 > Q1 → 模型有效 ✅
# - 单调性: Spearman 相关系数 > 0.7 → 强相关 ✅✅
# - 如果 Q5 < Q3 → 模型失效 ❌
```

**输出示例**：
```
Quintile | Avg RR | Sample Count
---------|--------|-------------
Q1 (Low) | 1.2    | 1200
Q2       | 1.5    | 1200
Q3       | 1.8    | 1200
Q4       | 2.3    | 1200
Q5 (High)| 3.1    | 1200  ← 预测高分的交易确实 RR 更大 ✅
```

**✅ 可以自动化**: 在训练脚本末尾添加这个统计

---

## 🔍 第二阶段：深度审查（30分钟）

### 4. Evidence 转换率（人工标注 + 统计）

**目的**: 衡量有多少规则是可理解、可转化为 Evidence 的

```python
# 人工审查 top 30 规则，标注：
# - ✅ 可理解且有交易语义（如 "momentum > 0.5 AND volatility < 0.3"）
# - ⚠️ 可理解但逻辑可疑（如 "hour == 14 AND symbol == BTC"）
# - ❌ 不可理解或无意义（如复杂交互项 "feat_A * feat_B / feat_C"）

conversion_rate = (可理解规则数 / 30) * 100%

# 评估标准：
# - > 50% → 模型学到了有价值的交易逻辑 ✅✅
# - 30-50% → 部分有效，需筛选 ✅
# - < 30% → 模型主要在拟合噪声 ❌
```

**记录到训练报告中**（手动填写即可）

---

### 5. 特征稳定性（跨币种一致性）

**目的**: 检查 top 特征是否在所有币种都重要（而非某个币种过拟合）

```python
# 对每个币种单独训练，记录 top 10 特征
btc_top10 = [...]
eth_top10 = [...]
bnb_top10 = [...]

# 计算交集
common_features = set(btc_top10) & set(eth_top10) & set(bnb_top10)

# 评估标准：
# - 交集 > 5 → 发现了通用交易规律 ✅
# - 交集 < 3 → 模型在学币种特异性，泛化性差 ❌
```

---

## ✅ 第三阶段：实战验证（由完整 Pipeline 决定）

将 Evidence 候选规则接入完整 pipeline：
```bash
Gate (过滤) → Evidence (Return Tree规则) → PCM (执行)
```
**最终看整体 Sharpe 是否提升**（这才是终极验证）

---

## 📋 评估表模板（每次训练填写）

训练完成后，手动填写这个表（5分钟完成）：

```markdown
## Return Tree 语义发现能力评估

### 训练信息
- 训练目录: `results/train_final_YYYYMMDD_HHMMSS_return_tree/bpc/`
- 训练时间: YYYY-MM-DD HH:MM
- 数据范围: YYYY-MM-DD ~ YYYY-MM-DD
- Holdout: YYYY-MM-DD ~ YYYY-MM-DD

---

### 自动指标
- [ ] Top 10 重要性占比: ___% (目标: 30-60%)
- [ ] 预测分位数 Spearman 相关: ___ (目标: > 0.6)
- [ ] 泄露特征数 (如 _symbol): ___ (目标: 0)
- [ ] Q5 平均 RR / Q1 平均 RR: ___ (目标: > 1.5x)

---

### 人工审查
- [ ] Top 30 规则中可理解规则数: ___ / 30 (目标: > 15)
- [ ] 主导特征类型: 
  - [ ] momentum (动量)
  - [ ] liquidity (流动性)
  - [ ] regime (市场状态)
  - [ ] technical (技术指标)
  - [ ] 其他: ___
- [ ] 是否发现新的交易逻辑: Yes / No
  - 描述: ___

---

### 决策
- [ ] ✅ 通过 - 规则质量高，可转化为 Evidence
- [ ] ⚠️ 部分通过 - 需筛选规则（转换率 30-50%）
- [ ] ❌ 不通过 - 重新调整特征/标签

---

### 备注
- 主要发现: ___
- 需要改进的地方: ___
```

---

## 🛠️ 实施优先级

### 立即实现（下次训练加入）
1. **预测分位数分析** → 最客观的有效性验证
2. **特征重要性集中度检查** → 快速诊断过拟合

### 定期人工审查
3. **Evidence 转换率** → 每次新模型都要做（30分钟）

### 长期优化
4. **跨币种稳定性测试** → 验证泛化能力（需要单独训练）

---

## 📚 参考资料

- 相关配置: `config/strategies/bpc/labels_return_tree.yaml`
- 训练命令: 见 [`cmd: 树模型到archetype.md`](./cmd:%20树模型到archetype.md#训练-return-tree)
- Gate 模块 KPI: Lift < 1.0（见 Failure Analysis）

---

## 🔗 相关文档链接

- [树模型训练流程](./cmd:%20树模型到archetype.md)
- Return Tree 核心定位: 语义发现而非收益生成
- 最终验证: 需结合 Gate + Evidence + PCM 完整 pipeline
