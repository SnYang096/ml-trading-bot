# BPC (BreakoutPullbackContinuation) 策略

## 策略概述

BreakoutPullbackContinuation archetype：趋势中回踩后延续原方向。

**核心理念**：
- Breakout: 突破压缩区/前高，放量+CVD同向
- Pullback: 回踩但不破结构，缩量+CVD吸收
- Continuation: 重新启动，再次放量+CVD恢复

## 训练配置

- **数据集**: highcap6 (BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, ADAUSDT)
- **时间窗口**: 2023-01-01 至 2024-12-31
- **标签类型**: forward_rr (outcome audit标签，全量数据)
- **模型类型**: LightGBM 回归
- **样本数**: 2132 (训练集)
- **特征数**: 143

## 模型性能

- **交叉验证**: 5-fold时序交叉验证
- **平均CV指标**: -0.0720
- **训练完成时间**: 2026-01-30

## 特征分层

### Atomic 层（原子信号）
- **breakout**: 价格突破强度、成交量确认、CVD确认、VPIN确认
- **pullback**: 回踩深度、回踩质量、缩量确认、CVD吸收
- **continuation**: 恢复强度、动量确认、放量确认、CVD动量、VPIN上升
- **neutral**: 波动率压缩、成交量压缩

### Composite 层（组合信号）
- bpc_score_breakout: 突破综合分
- bpc_score_pullback: 回踩综合分
- bpc_score_continuation: 续行综合分
- bpc_score_neutral: 中性综合分

### Contextual 层（状态信号）
- bpc_breakout_direction: 突破方向
- bpc_direction_confidence: 方向置信度
- bpc_is_after_breakout: 是否在突破后
- bpc_was_in_pullback: 是否经历过回踩

---

## 📜 树模型规则导出（固定训练 LightGBM）

以下为从固定训练产出的 LightGBM 模型中提取的**高频分裂条件**（按出现次数排序），用于可归因性与规则维护。

| 特征 | 条件 | 出现次数 |
|------|------|----------|
| `_symbol` | `_symbol <= 5` | 67 |
| `bpc_dir_flip_count` | `bpc_dir_flip_count <= 0.525` | 39 |
| `cvd_change_1` | `cvd_change_1 <= -3.488e+05` | 28 |
| `cvd_change_1` | `cvd_change_1 <= 7.948e+04` | 26 |
| `cvd_change_1` | `cvd_change_1 <= -4.009e+04` | 26 |
| `trade_cluster_directional_entropy_ma20` | `trade_cluster_directional_entropy_ma20 <= 1` | 26 |
| `trade_cluster_total_run_length` | `trade_cluster_total_run_length <= 0.009964` | 24 |
| `trade_cluster_net_runs_ma5` | `trade_cluster_net_runs_ma5 <= 0` | 23 |
| `cvd_change_5` | `cvd_change_5 <= 1.292e+05` | 23 |
| `bpc_breakout_direction` | `bpc_breakout_direction <= 0` | 20 |
| `trade_cluster_directional_entropy_ma10` | `trade_cluster_directional_entropy_ma10 <= 1` | 20 |
| `_symbol` | `_symbol <= 0` | 20 |
| `trade_cluster_net_runs_ma10` | `trade_cluster_net_runs_ma10 <= -0` | 19 |
| `trade_cluster_directional_entropy_ma20` | `trade_cluster_directional_entropy_ma20 <= 1` | 19 |
| `cvd_change_1` | `cvd_change_1 <= 9.178e+04` | 19 |
| `trade_cluster_net_runs_ma20` | `trade_cluster_net_runs_ma20 <= -0` | 18 |
| `cvd_change_1` | `cvd_change_1 <= -2.827e+05` | 18 |
| `trade_cluster_directional_entropy_ma5` | `trade_cluster_directional_entropy_ma5 <= 1` | 17 |
| `bpc_pullback_duration` | `bpc_pullback_duration <= 0` | 15 |
| `vpin_volatility_20` | `vpin_volatility_20 <= 0.0538` | 15 |
| `macd` | `macd <= 0.08917` | 14 |
| `trade_cluster_imbalance_ratio_ma20` | `trade_cluster_imbalance_ratio_ma20 <= -1e-06` | 14 |
| `cvd_change_1` | `cvd_change_1 <= 1.093e+05` | 14 |
| `cvd_change_1` | `cvd_change_1 <= -3.533e+04` | 14 |
| `bpc_dir_flip_count` | `bpc_dir_flip_count <= 0.575` | 13 |
| `hilbert_cvd_price_env_ratio` | `hilbert_cvd_price_env_ratio <= 0` | 13 |
| `cvd_change_20` | `cvd_change_20 <= -5.3e+05` | 13 |
| `cvd_change_20` | `cvd_change_20 <= -4.181e+07` | 13 |
| `trade_cluster_net_runs_ma5` | `trade_cluster_net_runs_ma5 <= -1.7e-05` | 13 |
| `bpc_vol_pullback_confirm` | `bpc_vol_pullback_confirm <= 0.03036` | 13 |

**模型来源**：`/home/yin/trading/ml_trading_bot/models/bpc`

