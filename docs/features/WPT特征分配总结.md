# WPT 特征分配总结

## 概述

本文档总结了 WPT（小波包变换）特征在四个策略和波动率模型中的分配情况。

## WPT 特征列表

### 基础 WPT 特征（来自 `wpt_price_reconstructed` 和 `wpt_price_fluctuation`）

| 特征名 | 说明 | 输出列 |
|--------|------|--------|
| `wpt_price_reconstructed` | WPT 重构价格（趋势+波动） | `wpt_price_trend`, `wpt_price_fluctuation`, `wpt_price_reconstructed` |
| `wpt_price_fluctuation` | WPT 价格波动 + 能量比 | `wpt_price_fluctuation`, `wpt_price_trend`, `wpt_price_energy_low_ratio`, `wpt_price_energy_mid_ratio`, `wpt_price_energy_high_ratio` |
| `wpt_cvd_fluctuation` | WPT CVD 波动（资金流去趋势） | `wpt_cvd_fluctuation` |

### 高级 WPT 特征

| 特征名 | 说明 | 输出列 |
|--------|------|--------|
| `wpt_volume_energy` | WPT + Volume 能量协同分析 | `wpt_vper_low`, `wpt_vper_mid`, `wpt_vper_high`, `wpt_energy_cascade`, `wpt_multi_scale_consistency`, `wpt_breakout_confidence`, `wpt_false_breakout_risk` |
| `wpt_vpvr` | WPT 降噪的 VPVR | `vpvr_pvp`, `vpvr_hvn_count`, `vpvr_lvn_count`, `vpvr_lvn_distance`, `vpvr_volume_density`, `vpvr_price_in_lvn` |

### WPT 波动率增强特征（通过 `enhance_wpt_vol_features` 动态生成）

| 特征名 | 说明 | 用途 |
|--------|------|------|
| `wpt_price_high_energy_ratio` | 高频能量占比（噪声强度） | 波动率模型 |
| `wpt_price_fluct_l1_l2_ratio` | 波动信号的 L1/L2 范数比（尖峰程度） | 波动率模型 |
| `wpt_vhph_sync` | 体积-价格高频同步性 | 波动率模型 |

## 策略分配

### 1. SR Reversal（SR 反转策略）

**WPT 特征**：
- ✅ `wpt_price_reconstructed` - 多尺度 SR 结构
- ✅ `wpt_price_fluctuation` - 能量比（识别噪声水平）
- ✅ `wpt_cvd_fluctuation` - CVD 资金流去趋势
- ✅ `wpt_vpvr` - 流动性聚集区

**用途**：
- 识别多尺度 SR 结构（低频 POC）
- 过滤高频噪声
- 确认流动性聚集区

### 2. SR Breakout（SR 突破策略）

**WPT 特征**：
- ✅ `wpt_price_reconstructed` - 多尺度突破分析
- ✅ `wpt_price_fluctuation` - 能量比（识别假突破）
- ✅ `wpt_volume_energy` - 突破置信度、假突破风险
- ✅ `wpt_cvd_fluctuation` - 资金流去趋势（识别假突破）
- ✅ `wpt_vpvr` - 流动性真空区识别
- ✅ `liquidity_void` - 流动性真空区

**用途**：
- 识别真假突破（`wpt_breakout_confidence`, `wpt_false_breakout_risk`）
- 多尺度一致性验证（`wpt_multi_scale_consistency`）
- 能量下移检测（`wpt_energy_cascade`）

### 3. Compression Breakout（压缩区突破策略）

**WPT 特征**：
- ✅ `wpt_price_reconstructed` - 多尺度压缩区分析
- ✅ `wpt_price_fluctuation` - 能量比（压缩期 vs 突破期）
- ✅ `wpt_volume_energy` - 压缩能量、突破确认
- ✅ `wpt_cvd_fluctuation` - 资金流去趋势
- ✅ `liquidity_void` - 流动性真空区

**用途**：
- 识别压缩期能量蓄势（低频能量高）
- 突破时噪声检测（高频能量不应飙升）
- 压缩能量 × 订单流强度交互

### 4. Trend Following（趋势跟踪策略）

**WPT 特征**：
- ✅ `wpt_price_reconstructed` - 多尺度趋势分析
- ✅ `wpt_price_fluctuation` - 能量比（趋势主导 vs 噪声）
- ✅ `wpt_cvd_fluctuation` - 资金流去趋势

**用途**：
- 识别趋势主导（低频能量占比高）
- 确认趋势质量（能量集中在低频）
- 过滤噪声（高频能量低）

### 5. Volatility Model（波动率模型）

**WPT 特征**（在 `volatility_model.yaml` 中配置）：
- ✅ `wpt_features` - 完整的 WPT 特征组
  - 基础特征：`wpt_price_trend`, `wpt_price_fluctuation`, `wpt_price_energy_*_ratio`, `wpt_volume_energy_low_ratio`, `wpt_cvd_energy_low_ratio`, `wpt_vper`
  - 增强特征（动态生成）：`wpt_price_high_energy_ratio`, `wpt_price_fluct_l1_l2_ratio`, `wpt_vhph_sync`

**用途**：
- 多尺度波动率预测
- 识别噪声强度（高频能量占比）
- 尖峰程度检测（L1/L2 比）
- 量价一致性（体积-价格高频同步）

## 特征选择原则

1. **SR Reversal**：关注多尺度 SR 结构和流动性确认
2. **SR Breakout**：关注真假突破识别和多尺度一致性
3. **Compression Breakout**：关注压缩期能量蓄势和突破确认
4. **Trend Following**：关注趋势主导和噪声过滤
5. **Volatility Model**：关注多尺度波动率预测和噪声强度

## 注意事项

1. **WPT 特征具有相位滞后**：约 `window//2` 的滞后，适用于中低频策略（日线/4H）
2. **性能考虑**：对于 4H/日线，`update_step=1`（每根K线都计算）完全可行
3. **特征增强**：波动率模型会自动调用 `enhance_wpt_vol_features` 生成增强特征
4. **配置驱动**：所有特征通过 `feature_dependencies.yaml` 统一管理

