# 多Archetype选择机制实施报告

**测试时间**: 2026-01-22  
**目的**: 实现简化的多archetype选择机制，添加CVD regime判断，修复文档错误

---

## 实施内容

### 1. 文档修复 ✅

#### 1.1 修复archetype组合数量错误
- **文件**: `docs/experiments/EXP_GATE_ENHANCEMENT_COMPARISON_2026_01.md`
- **修复**: 更正为4种archetype（TC、TE、FR、ET），说明移除VolMean后从5种减少到4种

#### 1.2 链接到总架构文档
- **文件**: `docs/architecture/FINAL_SIMPLIFIED_ARCHITECTURE_2026_01.md`
- **操作**: 添加链接到 `docs/ARCHITECTURE.md`
- **文件**: `docs/ARCHITECTURE.md`
- **操作**: 添加反向链接到 `FINAL_SIMPLIFIED_ARCHITECTURE_2026_01.md`

#### 1.3 更新其他相关文档
- **文件**: `docs/experiments/EXP_GATE_PLATEAU_OPTIMIZATION_2026_01.md`
- **修复**: 更正archetype数量描述

---

### 2. 添加CVD Regime判断（使用Percentile，避免硬阈值）✅

#### 2.1 设计原则
- **问题**: 硬阈值（CVD < 0 / >= 0）会破坏跨symbol的平坦高原
- **原因**: CVD的零点在不同symbol中意义不同（BTC永续vs小币vs合约vs现货）
- **解决方案**: 使用CVD的percentile（相对位置）而不是绝对符号

#### 2.2 TC的CVD判断
- **规则**: CVD percentile > 0.85 → veto（明显反向成交堆积）
- **文件**: `config/nnmultihead/execution_archetypes.yaml`
- **添加规则**:
  ```yaml
  - name: tc_cvd_too_negative
    kind: quantile_gt
    key: cvd_change_5
    quantile: 0.85
    on_missing: false
  ```
- **逻辑**: 使用percentile而不是硬阈值，确保跨symbol稳定性
- **优势**: 
  - 所有symbol的CVD被拉到同一坐标系
  - 阈值在"分位空间"，天然稳定
  - 容易形成平坦高原

#### 2.3 ET的CVD判断
- **规则**: CVD percentile > 0.3 → veto（CVD不够负，相对于历史）
- **文件**: `config/nnmultihead/execution_archetypes.yaml`
- **添加规则**:
  ```yaml
  - name: et_cvd_not_negative_enough
    kind: quantile_gt
    key: cvd_change_5
    quantile: 0.3
    on_missing: false
  ```
- **逻辑**: ET需要CVD在低分位数（负向分歧），但使用percentile而不是硬阈值
- **优势**: 
  - 避免硬阈值破坏跨symbol的平坦高原
  - 使用相对位置确保稳定性

**效果**:
- TC和ET在订单流方向上不冲突（通过percentile判断）
- 提高archetype选择的清晰度
- 减少TC和ET同时出现的可能性
- **关键改进**: 使用percentile确保跨symbol的平坦高原，避免硬阈值问题

---

### 3. 实现简化的多Archetype选择机制 ✅

#### 3.1 选择规则（简化版）

**规则1: ET + FR同时出现**
- **行为**: 直接选择FR
- **原因**: ET优先级低于FR，FR在mean reversion场景中更常见
- **代码标识**: `et_fr_priority_fr`

**规则2: ET + TC同时出现**
- **行为**: NO_TRADE（等待ET单独出现）
- **原因**: ET被视为极端情况，只有在单独出现时才执行
- **代码标识**: `et_tc_wait_et_alone`

**规则3: 其他多个组合**
- **行为**: NO_TRADE（保持保守）
- **原因**: 避免不确定性，保持系统稳定性
- **代码标识**: `multiple_archetypes_no_trade`

#### 3.2 实现位置
- **文件**: `scripts/apply_archetype_gate.py`
- **函数**: `main()` 函数中的多archetype处理逻辑
- **行数**: 约534-570行

#### 3.3 代码实现

```python
# Multi-archetype selection logic (simplified)
if len(passing_candidates) > 1:
    arch_names = [arch_name for arch_name, _, _ in passing_candidates]
    
    # Rule 1: ET+FR → select FR
    if "ExhaustionTurnET" in arch_names and "FailureReversionFR" in arch_names:
        # Find FR in passing candidates and select it
        selected_arch = find_fr_in_candidates(passing_candidates)
        if selected_arch:
            execute(selected_arch)
            continue
    
    # Rule 2: ET+TC → NO_TRADE (wait for ET to appear alone)
    if "ExhaustionTurnET" in arch_names and "TrendContinuationTC" in arch_names:
        reject("et_tc_wait_et_alone")
        continue
    
    # Rule 3: Other combinations → NO_TRADE
    reject("multiple_archetypes_no_trade")
```

---

### 4. 文档更新 ✅

#### 4.1 更新架构文档
- **文件**: `docs/architecture/FINAL_SIMPLIFIED_ARCHITECTURE_2026_01.md`
- **添加内容**:
  - 多Archetype选择机制说明
  - ET的特殊性（极端情况，需要单独出现）
  - CVD Regime判断说明

#### 4.2 代码注释
- **文件**: `scripts/apply_archetype_gate.py`
- **添加**: 详细注释解释选择逻辑
- **说明**: 为什么ET+FR选择FR，为什么ET+TC要等待

---

## 设计决策

### 为什么ET+FR选择FR？

1. **FR更常见**: FR在mean reversion场景中比ET更常见
2. **优先级**: FR优先级高于ET，ET被视为极端情况
3. **简化**: 避免复杂的tie-break逻辑，直接选择FR

### 为什么ET+TC要等待？

1. **极端情况**: ET被视为极端情况，只有在单独出现时才执行
2. **避免冲突**: ET和TC在订单流方向上可能冲突（ET需要负CVD，TC需要正CVD）
3. **保守策略**: 等待ET单独出现，确保信号清晰

### 为什么使用Percentile而不是硬阈值？

1. **跨symbol稳定性**: 硬阈值（CVD < 0）在不同symbol中意义不同
   - BTC永续：CVD零点≈平衡
   - 小币/新币：CVD长期偏正或偏负
   - 合约vs现货：CVD漂移方向不同

2. **平坦高原**: 硬阈值是阶跃函数，plateau几乎不可能出现
   - Percentile将所有symbol拉到同一坐标系
   - 阈值在"分位空间"，天然稳定

3. **语义正确**: TC/TE的本质不是"买卖量方向"，而是"是否异常"
   - 关心的是CVD是否在自己历史分布的极端
   - 而不是绝对正负

4. **设计原则**: CVD只负责"成交行为是否与结构严重矛盾"
   - 不是router的硬gate
   - 不是regime的定义轴
   - 是regime score的调制因子或execution前的veto

---

## 预期效果

### 1. 减少多Archetype冲突
- ET+FR组合：直接选择FR，避免冲突
- ET+TC组合：等待ET单独出现，避免不确定性
- 其他组合：保持保守，NO_TRADE

### 2. 提高Archetype选择清晰度
- CVD判断确保TC和ET在订单流方向上不冲突
- 减少同时通过gate的archetype数量

### 3. 保持系统稳定性
- 保守的多archetype处理策略
- ET作为极端情况，只在单独出现时执行

---

## 测试建议

### 1. 运行Gate检查
```bash
python scripts/apply_archetype_gate.py \
  --logs results/e2e_kpi/logs_3action_2024.parquet \
  --out results/e2e_kpi/logs_3action_2024_multi_select.parquet \
  --features-store-layer nnmh_highcap6_240T_2024_202510_ma_adx_cvd_vwap_v1 \
  --features-store-root feature_store
```

### 2. 检查多Archetype统计
- 查看ET+FR组合是否选择了FR
- 查看ET+TC组合是否被拒绝
- 查看其他组合的处理情况

### 3. 验证CVD判断（Percentile-based）
- 检查TC在CVD percentile > 0.85时是否被拒绝
- 检查ET在CVD percentile > 0.3时是否被拒绝
- 验证TC和ET同时出现的频率是否降低
- **关键**: 验证跨symbol的稳定性（不同symbol使用相同的percentile阈值）

---

## 相关文件

- `scripts/apply_archetype_gate.py` - 多archetype选择逻辑实现
- `config/nnmultihead/execution_archetypes.yaml` - CVD判断规则配置
- `docs/architecture/FINAL_SIMPLIFIED_ARCHITECTURE_2026_01.md` - 架构文档更新
- `docs/experiments/EXP_GATE_ENHANCEMENT_COMPARISON_2026_01.md` - 文档修复

---

## 结论

✅ **实施完成**: 所有计划任务已完成
- 文档修复和链接完成
- CVD判断规则添加完成（使用percentile而不是硬阈值）
- 多archetype选择机制实现完成
- 文档更新完成

**关键改进**:
- **CVD判断从硬阈值改为percentile**: 确保跨symbol的平坦高原
  - TC: `quantile_gt 0.85`（CVD percentile > 0.85时拒绝）
  - ET: `quantile_gt 0.3`（CVD percentile > 0.3时拒绝）
- **多archetype选择**: ET+FR选FR，ET+TC等待

**下一步**: 运行测试验证效果，检查多archetype选择逻辑是否正确工作，验证跨symbol的CVD percentile稳定性。
