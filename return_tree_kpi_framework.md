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

## 🧱 BPC Gate 规则提炼备忘

> 目标：从树模型规则导出（`bpc_tree_rules.md`）和自动草稿（`risk_gate_draft.yaml`）中，提炼出少量**宪法级 Gate 规则**，只负责结构性 veto，而不干扰 timing / execution。

- **信息来源拆分**：
  - `bpc_tree_rules.md`：告诉我们“模型在什么维度上频繁切割”（高频分裂条件）。
  - `risk_gate_draft.yaml`：把这些高频条件机械展开为一堆一维 soft_filter 草稿。
  - 真实 Gate 需要再经过一层**人类语义筛选**，只留下“结构性失效”维度。

- **优先考虑作为 Gate 的特征族（结构性维度）**：
  - **BPC 形态本身是否成立**：`bpc_pullback_depth`、`bpc_pullback_quality`、`bpc_score_pullback`、`bpc_score_continuation`、`bpc_dir_consistency_long`。
  - **价格相对 SR 的危险位置**：`dist_to_nearest_sr`、`sr_strength_max`（避免在 SR 内部或旧坑深处做 BPC）。
  - **VP / WPT 场景极端态**：`vp_compression_score`、`vp_exhaustion_score`、`wpt_compression_score`、`wick_ignition_score` 等（流动性/能量状态极端时，容易踩结构性坑）。
  - 这几类更适合作为 Gate 的 veto 条件；而 `macd_atr`、`vol_slope_*`、`vol_zscore` 等偏“数学 proxy”的信号，优先交给 Evidence/PCM，而非 Gate。

- **Hard Gate 与 Soft Filter 的分工**：
  - **Hard Gate**：只承载“结构性不合法”的否决，例如：
    - `bpc_pullback_depth` 过小 → 形态过浅、疑似假 BPC（`hard_bpc_pullback_too_shallow`）。
    - `bpc_dir_consistency_long` 很低 → 形态内部方向严重不一致（`hard_bpc_dir_inconsistent`）。
  - **Soft Filter**：对同一维度的“次优区域”做降权，而不直接否决，例如：
    - `filter_bpc_pullback_depth`：在 Hard Gate 之外，对偏浅形态进一步减信心。
    - `filter_dist_to_nearest_sr` / `filter_vp_compression_score` / `filter_vp_exhaustion_score` 等：落在 SR 内部、高压缩或趋势末端时下调置信度。
  - 这样实现 Gate 的职责边界：**只负责 veto 结构性坏场景，不替代趋势判断和执行节奏。**

- **验证路径（必须回到 Trades + Failure Analysis）**：
  - 先在 `risk_gate_draft.yaml` 中按上述原则填充 `hard_gates.rules` 与精简后的 `soft_filters.rules`。
  - 使用 `mlbot train` + 回测，看：
    - `Trades` 数量变化（Hard Gate 不应把交易打到接近 0）。
    - Failure Analysis 中，被 Hard Gate 拦截样本的 `failure_rr_extreme` 是否显著高于全局。
  - 若某条 Hard Gate 触发频率极高但并未显著降低 failure，则应降级为 Soft Filter；反之，则可保留为“宪法级 veto 规则”。

---

## 🧭 模型分析作用域与阈值使用备忘

- **Gate 阈值与回测阈值的角色区分**：
  - 训练脚本和 HTML 报告中的 Gate 评估，应基于 **Failure Analysis 的 lift 曲线** 以及 **lift vs coverage 平坦高原** 来选阈值；
  - 回测里将 `long_entry_threshold` 暂时设为 0.6 之类的数值，只是为了“先有一些交易、验证行为不离谱”，**不能当作 Gate 宪法级阈值**；
  - 后续正式使用时，应先在百分位空间（Top20/30/40% 等）扫一遍 lift plateau，再反推对应的分数/概率阈值。

- **残差分析的作用域**：
  - `analyze_failure_distribution` / HTML 报告中的 Failure Analysis，是 **在 Gate 模型自身的输出空间上评估 veto 能力**，适合作为 Gate KPI；
  - `mlbot analyze gate-residual` 更适合作为 **早期 debug 工具**：在某个临时选定的 Gate cut（如 success_prob ≥ 某值）上，画出 Gate 通过后剩余失败的“画像”；
  - 严格意义上的残差拆解（Gate vs Evidence vs PCM 谁背锅），应该在 **完整分层策略的回测输出** 上做，而不是直接在 Gate Tree 或 forward_rr Tree 上做，因为这些树模型本身只是“规则发现器”，不是最终交易模型。

- **树规则与最终规则边界的关系**：
  - LightGBM Split 提供的是“模型在什么维度、什么大致阈值附近频繁切割”的 **语义线索**，适合用来验证先验、发现候选规则族；
  - 最终用于 Gate/Evidence 的规则边界，建议通过 **在 holdout 上做分箱曲线 / plateau 搜索** 来重新定位（例如在 `bpc_pullback_depth` 对 failure_rate 的曲线上找一段平坦高原，再选中段作为阈值）；
  - 像 imodels 这类显式规则拟合工具可以作为“第二意见”或局部实验，用来压缩成少量 if-then 规则，但当前主线仍以 “GBDT 分裂点 + plateau 手工调整” 为准，以保证规则语义与分层架构职责一致。

---

## 🔗 相关文档链接

- [树模型训练流程](./cmd:%20树模型到archetype.md)
- Return Tree 核心定位: 语义发现而非收益生成
- 最终验证: 需结合 Gate + Evidence + PCM 完整 pipeline
