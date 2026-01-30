# BPC 软阶段特征设计文档

## 概述

本文档描述 BPC (Breakout-Pullback-Continuation) 策略的软阶段特征系统设计。

**核心理念：**
> Price tells you WHAT, Volume tells you IF, Order flow tells you WHO

## 设计目标

### 从「硬阶段」到「软阶段」

| 维度 | 硬阶段（旧） | 软阶段（新） |
|------|-------------|-------------|
| **输出类型** | `phase: int (0/1/2/3)` | `score: float [0-1]` × 4 |
| **噪声鲁棒性** | 差（微小波动导致跳变） | 好（平滑响应） |
| **信息量** | 低（只有类别） | 高（包含置信度） |
| **模型友好度** | 低（需 one-hot，稀疏） | 高（连续，梯度友好） |
| **可解释性** | "现在是 pullback" | "有 70% 可能处于健康 pullback" |

## BPC 三阶段语义

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                BPC 生命周期                                  │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│     Neutral     │    Breakout     │    Pullback     │    Continuation       │
│  （蓄势阶段）    │  （突破阶段）    │   （回踩阶段）   │     （延续阶段）       │
├─────────────────┼─────────────────┼─────────────────┼───────────────────────┤
│ • 低波动        │ • 价格突破前高   │ • 价格回踩      │ • 重新向原方向启动     │
│ • 低成交量      │ • 放量确认      │ • 缩量回踩      │ • 再次放量            │
│ • 无方向偏好    │ • CVD 同向      │ • CVD 不创新低  │ • CVD 恢复            │
│                │ • VPIN 活跃     │ • 订单流吸收    │ • VPIN 上升           │
└─────────────────┴─────────────────┴─────────────────┴───────────────────────┘
```

## 特征架构

### 1. 核心输出：软阶段分数

| 特征名 | 范围 | 语义 |
|--------|------|------|
| `bpc_score_breakout` | [0-1] | 突破强度（价格×放量×CVD×VPIN） |
| `bpc_score_pullback` | [0-1] | 回踩质量（深度浅×缩量×CVD吸收） |
| `bpc_score_continuation` | [0-1] | 延续动能（恢复×动量×放量×CVD） |
| `bpc_score_neutral` | [0-1] | 中性/蓄势（低波动×低成交量） |

### 2. 各阶段的验证逻辑

#### Breakout 阶段验证

```
bpc_score_breakout = 
    price_breakout_strength × 0.4 +
    price_strength × vol_breakout_confirm × 0.3 +
    price_strength × cvd_breakout_confirm × 0.2 +
    price_strength × vpin_breakout_confirm × 0.1
```

**验证条件：**
- ✅ 价格突破前高/低 + ATR 阈值
- ✅ 成交量 > 1.5 倍均量
- ✅ CVD 同向（多头 breakout 时 CVD > 0）
- ✅ VPIN > 0.6（信息交易活跃）

#### Pullback 阶段验证

```
bpc_score_pullback = 
    is_after_breakout × pullback_quality × 0.3 +
    is_after_breakout × pullback_quality × vol_pullback_confirm × 0.3 +
    is_after_breakout × pullback_quality × cvd_absorption × 0.4
```

**验证条件：**
- ✅ 近期有有效 breakout（score > 0.3）
- ✅ 回踩深度 < 70%（结构未破坏）
- ✅ 成交量百分位低（缩量）
- ✅ CVD 不创新极值（吸收信号）

#### Continuation 阶段验证

```
bpc_score_continuation = 
    was_in_pullback × recovery_strength × momentum_confirm × 0.3 +
    was_in_pullback × recovery_strength × vol_continuation_confirm × 0.3 +
    was_in_pullback × recovery_strength × cvd_momentum × 0.25 +
    was_in_pullback × recovery_strength × vpin_rising × 0.15
```

**验证条件：**
- ✅ 曾处于 pullback 阶段
- ✅ 价格从 pullback 极值恢复
- ✅ 短期动量与 breakout 方向一致
- ✅ 成交量再次放量（> 1.2 倍均量）
- ✅ CVD 恢复 + VPIN 上升

#### Neutral 阶段验证

```
bpc_score_neutral = 
    bb_compression × 0.4 +
    vol_compression × 0.4 +
    (1 - max_other_score) × 0.2
```

**验证条件：**
- ✅ 波动率压缩（BB width 低）
- ✅ 成交量压缩（百分位低）
- ✅ 其他阶段分数都低

## 辅助特征

### 结构特征

| 特征 | 语义 | 范围 |
|------|------|------|
| `bpc_pullback_depth_long` | 多头回踩深度 | [0-1] |
| `bpc_pullback_depth_short` | 空头回踩深度 | [0-1] |
| `bpc_pullback_depth_pct` | 自适应回踩深度 | [0-1] |
| `bpc_pullback_duration` | 回踩持续 bars | [0-1] |
| `bpc_pullback_speed` | depth/(duration+1) | [0-1] |
| `bpc_impulse_return_atr` | impulse 收益/ATR | [-1,1] |
| `bpc_impulse_direction_match` | 方向匹配度 | [0,1] |

### 方向一致性特征

| 特征 | 语义 | 窗口 |
|------|------|------|
| `bpc_dir_consistency_short` | 短期方向一致性 | 5 bars |
| `bpc_dir_consistency_mid` | 中期方向一致性 | 20 bars |
| `bpc_dir_consistency_long` | 长期方向一致性 | 50 bars |
| `bpc_dir_flip_count` | 方向翻转次数 | 20 bars |

### 量能/订单流特征

| 特征 | 语义 |
|------|------|
| `bpc_volume_compression_pct` | 成交量压缩百分位 |
| `bpc_pullback_delta_absorption` | 回踩期间 delta 吸收强度 |
| `bpc_vol_ratio` | 当前成交量/均量 |
| `bpc_cvd_z` | CVD z-score |

## 实现细节

### 状态安全规范

根据 `BPC阶段建模函数状态安全规范`：

1. **状态变量初始化**：所有状态变量在函数内部初始化，禁止跨样本残留
2. **输出为时间序列**：`bpc_breakout_direction` 等必须为数组，禁止标量广播
3. **严格因果性**：每个时间点 i 的判断仅依赖 [0:i] 历史数据
4. **分组调用**：多 instrument 数据必须分组调用，避免状态污染

### 鲁棒性设计

根据 `特征工程鲁棒性设计规范`：

1. **方向信息保留**：`bpc_impulse_return_atr` 保留符号
2. **防数值爆炸**：`bpc_pullback_speed = depth/(duration+1)`
3. **z-score 标准化**：`bpc_pullback_delta_absorption` 使用滚动 z-score
4. **Side-aware 设计**：所有特征支持多空场景

## 使用示例

### 1. 在策略配置中使用

```yaml
# config/strategies/bpc/features.yaml
feature_pipeline:
  requested_features:
    - bpc_soft_phase_f  # 核心软阶段分数
    - bpc_pullback_depth_pct_f
    - bpc_pullback_delta_absorption_f
    # ...
```

### 2. 在 Outcome-Based 审计中使用

BPC 软阶段分数可用于：
- 识别失败模式：`bpc_score_breakout > 0.7 AND bpc_score_pullback > 0.3 AND forward_rr < -1`
- 导出负规则：树模型学习「什么情况下 BPC 结构会失败」
- 构建 Gate：基于软阶段分数设置入场条件

### 3. 特征组合解释

| 场景 | 软阶段分数组合 | 解释 |
|------|---------------|------|
| 强势 BPC | breakout=0.8, pullback=0.7, continuation=0.6 | 完整的 BPC 结构正在形成 |
| 假突破 | breakout=0.6, pullback=0.2, neutral=0.5 | 突破后未形成有效回踩 |
| 结构破坏 | breakout=0.3, pullback=0.1, neutral=0.7 | BPC 结构已经失效 |

## 依赖关系

```
bpc_soft_phase_f
├── atr_f
├── cvd_change_5_pct_f (可选)
├── vpin_features_f (可选)
├── ofci_pct_f (可选)
└── bb_width_normalized_pct_f (可选)

bpc_pullback_speed_f
├── bpc_pullback_depth_pct_f
└── bpc_pullback_duration_f

bpc_pullback_delta_absorption_f
├── atr_f
└── cvd_change_5 (raw column)
```

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-01-30 | 初始版本：软阶段分数 + 成交量 + 订单流 |

## 参考文档

- `docs/architecture/6种对称策略的启发式规则.md`
- `config/archetypes/bpc/hypotheses.yaml`
- `todo-bpc.md` - BPC 特征设计讨论
