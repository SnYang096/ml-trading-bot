# 特征更新总结：添加 vp_boundary_stability_score 和 sr_strength_max

## 📅 更新日期
2026-01-28

## ✅ 更新内容

### 新增特征

1. **`vp_boundary_stability_score`** (Volume Profile 边界稳定性分数)
   - **特征节点**: `volume_profile_volatility_features_f`
   - **输出列**: `vp_boundary_stability_score`
   - **说明**: 计算 raw VP 和 WPT VP 的边界一致性分数，用于判断市场结构稳定性
   - **值域**: (0, 1]，1.0 = 完美一致性（结构一致），接近 0 = 结构不一致（噪音/过渡）

2. **`sr_strength_max`** (最大 SR 强度)
   - **特征节点**: `sr_strength_max_f` 或 `sr_strength_max_close_f`
   - **输出列**: `sr_strength_max`
   - **说明**: 计算所有 SR 边界的最大强度，用于判断支撑/阻力位的有效性
   - **值域**: [0, 1]，值越大表示边界越强

## 📋 各策略更新状态

### 1. sr_reversal_rr_reg_long ✅
- **状态**: 已包含两个特征
- **特征**:
  - `sr_strength_max_close_f` ✅
  - `volume_profile_volatility_features_f` ✅

### 2. compression_breakout ✅
- **状态**: 已添加两个特征
- **更新**:
  - 添加 `sr_strength_max_f`
  - 添加 `volume_profile_volatility_features_f`

### 3. sr_breakout ✅
- **状态**: 已添加两个特征
- **更新**:
  - 添加 `sr_strength_max_close_f`
  - 添加 `volume_profile_volatility_features_f`

### 4. trend_following ✅
- **状态**: 已添加两个特征
- **更新**:
  - 添加 `sr_strength_max_f`
  - 添加 `volume_profile_volatility_features_f`

## 🎯 特征作用

### vp_boundary_stability_score

**用途**:
- 判断 Volume Profile 边界的稳定性
- 识别市场结构是否一致
- 用于路由决策（边界依赖型 vs 趋势型策略）

**市场状态映射**:
| score 区间 | 市场状态语义 |
|-----------|------------|
| > 0.75 | 结构稳定、边界可信 |
| 0.4–0.75 | 结构在，但开始松动 |
| < 0.4 | 微观噪音主导 / regime 过渡 |

### sr_strength_max

**用途**:
- 评估当前最强的支撑/阻力位强度
- 判断边界是否有效
- 用于交易信号过滤

**计算方式**:
- 基于价格行为（反弹次数、成交量、压缩强度）
- 归一化到 [0, 1] 范围
- 值越大表示边界越强

## 🔄 测试计划

### 重新运行所有测试

**短时间测试（滚动训练）**:
- 6个月训练 → 1个月预测
- 输出目录: `results/rolling_short/<strategy>/`

**长时间测试（固定训练）**:
- 3年训练 (2023-01-01 到 2025-12-31) → 15%测试集
- 输出目录: `results/fixed_long/<strategy>/`

### 预期改进

1. **边界稳定性判断**:
   - 在低稳定性时减少无效交易
   - 在高稳定性时提高交易质量

2. **SR 强度评估**:
   - 更准确地识别有效支撑/阻力位
   - 提高突破/反转信号的准确性

## 📊 测试状态

所有测试已重新启动，包含新特征：
- ✅ 4个策略 × 2种测试类型 = 8个测试
- ✅ 所有测试在后台运行
- ✅ 结果保存到不同目录

## 🔗 相关文档

- Volume Profile 边界设计: `docs/architecture/VOLUME_PROFILE_WPT_BOUNDARY_DESIGN.md`
- SR Strength 说明: `config/feature_dependencies.yaml` (sr_strength_max_f)
- 特征检查总结: `config/strategies/FEATURE_QUANTILE_CHECK_SUMMARY.md`
