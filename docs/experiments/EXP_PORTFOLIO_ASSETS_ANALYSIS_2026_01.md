# portfolio_assets.yaml 使用情况分析

**分析时间**: 2026-01-22  
**目的**: 分析portfolio_assets.yaml的实际使用情况，评估是否需要更新

---

## 执行摘要

### 关键发现

1. **当前配置**：
   - 定义了5个portfolio assets（GLOBAL_TREND, GLOBAL_MEAN, GLOBAL_CASH, HIGH_BETA_OVERLAY, DEFENSIVE_MEAN）
   - 使用`router_to_weights`从router aggregate signals映射到asset weights
   - 需要`p_trend`, `p_mean`, `regime_entropy`, `crowding_score`等信号

2. **实际使用**：
   - ✅ 在`counterfactual_eval_3action.py`中被使用（生成诊断artifacts）
   - ✅ 在`pipeline-3action-e2e`中设置环境变量`MLBOT_PORTFOLIO_ASSETS_YAML`
   - ✅ 用于PCM wiring和诊断

3. **问题**：
   - ⚠️ 当前系统没有router，无法生成`RouterAggregateSignals`
   - ⚠️ `aggregate_from_symbol_modes`函数需要symbol modes（TREND/MEAN/NO_TRADE）
   - ⚠️ 如果要去掉regime，这个函数需要适配新的archetype架构

---

## 详细分析

### 1. 配置内容

**文件**: `config/portfolio_assets/portfolio_assets.yaml`

**定义的Assets**:
- GLOBAL_TREND: max_weight 0.40, can_be_zero: true
- GLOBAL_MEAN: max_weight 0.35, min_weight: 0.20, can_be_zero: false
- GLOBAL_CASH: min_weight: 0.10, max_weight: 1.00
- HIGH_BETA_OVERLAY: max_weight: 0.10, can_be_zero: true
- DEFENSIVE_MEAN: max_weight: 0.25, can_be_zero: true

**router_to_weights映射**:
- 需要输入信号：`p_trend`, `p_mean`, `p_notrade`, `confidence`, `regime_entropy`, `crowding_score`
- 输出：5个portfolio assets的权重

### 2. 实际使用位置

**counterfactual_eval_3action.py** (第746-769行):
```python
if cfg.portfolio_assets_yaml and cfg.timestamp_col in test_df.columns:
    pa_cfg = load_portfolio_assets_config(str(cfg.portfolio_assets_yaml))
    # ... aggregate signals from symbol modes ...
    sig = aggregate_from_symbol_modes(
        decisions=decisions, key_symbols=list(cfg.portfolio_key_symbols)
    )
    w = compute_portfolio_asset_weights(
        cfg=pa_cfg, sig=sig, gate_veto=gate_veto, portfolio_drawdown=dd_proxy
    )
```

**aggregate_from_symbol_modes函数** (portfolio_assets.py:70-122):
- 输入：symbol modes（TREND/MEAN/NO_TRADE）
- 计算：`p_trend`, `p_mean`, `p_notrade`, `confidence`, `regime_entropy`
- 输出：`RouterAggregateSignals`

### 3. 问题分析

**问题1: 没有router**
- 当前系统没有router来生成aggregate signals
- `aggregate_from_symbol_modes`直接从symbol modes计算，但需要mode信息

**问题2: 依赖regime**
- `aggregate_from_symbol_modes`需要TREND/MEAN/NO_TRADE modes
- 如果去掉regime，需要适配archetype架构

**问题3: 信号缺失**
- `crowding_score`目前总是0.0（未实现）
- `regime_entropy`从modes计算，如果去掉regime需要重新设计

### 4. 建议

**选项1: 保留当前配置（推荐）**
- 标记为"未来使用"（当router实现后）
- 在文档中说明当前限制
- 保持配置结构，便于未来扩展

**选项2: 适配archetype架构**
- 修改`aggregate_from_symbol_modes`使用archetype替代regime
- 从archetype（TC, TE, FR, ET）计算aggregate signals
- 需要重新定义`p_trend`, `p_mean`的含义

**选项3: 简化配置**
- 移除router_to_weights（因为当前没有router）
- 只保留assets定义
- 等待router实现后再启用

---

## 相关文件

- `config/portfolio_assets/portfolio_assets.yaml` - 配置文件
- `src/time_series_model/portfolio/portfolio_assets.py` - 实现
- `src/time_series_model/rl/counterfactual_eval_3action.py` - 使用位置

---

## 结论

**当前状态**: portfolio_assets.yaml配置完整，但部分功能无法使用（因为没有router）

**建议**: 
1. 保留配置，标记为"未来使用"
2. 在文档中说明当前限制
3. 等待router实现或archetype架构迁移后再启用完整功能
