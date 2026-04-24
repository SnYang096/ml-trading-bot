# ME 特征语义分析与实现方案

## 一、ME 因果定义（最终版）

**ME = Energy × Acceleration × Participation**

BPC 是结构三阶段模型（Breakout → Pullback → Continuation），ME 是能量三因子模型。

### 与 BPC 对比

| | BPC | ME |
|---|---|---|
| 因果核心 | 位置事件：结构被打破 | 速度事件：资金在加速 |
| 前提 | 有区间/SR/压缩 | 不需要任何结构 |
| 失败原因 | 结构破坏（回踩失败） | 动能衰竭（加速度归零） |
| 三要素 | Breakout/Pullback/Continuation | Energy/Acceleration/Participation |
| 赚钱环境 | 结构清晰、区间明显 | 趋势中段、资金接力、清算驱动 |

**判断标准**：如果没有任何 SR 结构，信号还成立吗？成立→ME，不成立→BPC

## 二、三因子详解

### 1️⃣ Energy（能量环境）

> 市场是否允许扩张？

| 特征 | 计算 | 范围 | 语义 |
|---|---|---|---|
| `me_atr_pct` | ATR rolling percentile | [0,1] | 波动是否处于高位？ |
| `me_vol_regime` | ATR 5K 变化率 percentile | [0,1] | 波动是否在扩张？ |

不考虑结构位置。它回答：市场是否在"能动"的状态？

### 2️⃣ Acceleration（方向加速）

> 价格速度是否在增强？

| 特征 | 计算 | 范围 | 语义 | 用途 |
|---|---|---|---|---|
| `me_accel_2k` | 2根K线二阶导/ATR | [-3,3] | 微观瞬时加速 | **Entry** |
| `me_accel_5k` | 3K均速 vs 8K均速差/ATR | [-3,3] | 中观持续加速 | **Evidence** |
| `me_accel_persistence` | 近5根正acceleration比例 | [0,1] | 加速是否持续？ | Evidence |
| `me_multi_tf_alignment` | abs(sign(r3)+sign(r5)+sign(r10))/3 | [0,1] | 多周期共振 | Evidence |

**关键约束**：不能退化成 trend-following。二阶导数测的是"速度变化"，不是"速度本身"。
- 2K 过敏感 → 用于 Entry 微观触发
- 5K 更稳定 → 用于 Evidence 强度评估

### 3️⃣ Participation（参与确认）

> 是真实资金推动，还是虚假波动？

| 特征 | 计算 | 范围 | 语义 |
|---|---|---|---|
| `me_cvd_alignment` | (sign(price)×sign(CVD)+1)/2 | [0,1] | 订单流方向一致性 |
| `me_cvd_strength` | CVD.abs()/CVD_rolling_std | [0,1] | 订单流相对强度 |
| `me_volume_surge` | (volume/vol_MA) percentile | [0,1] | 成交量爆发度 |
| `me_volume_accel` | 短期vol/中期vol 变化率 pct | [0,1] | 成交量也在加速？ |
| `me_delta_net_flow` | delta_zscore × price_direction | [-1,1] | 净买卖力×方向（可选） |

## 三、层级结构

### Gate（环境允许）— 不看 acceleration

```
me_atr_pct >= threshold       (Energy 扩张OK)
me_cvd_alignment >= threshold (Flow 一致性OK)
me_volume_surge >= threshold  (Volume 放量OK)
```

Gate 只回答"环境允许吗？"，不判断加速度。

### Evidence（强度形成）— 乘法

```
me_evidence = me_atr_pct × |me_accel_5k_norm| × participation
```

其中 participation = cvd_alignment × (volume_surge 或 cvd_strength)。三者共振才有信号。

### Entry（微观触发）

```
me_accel_2k 瞬时爆发 (2K加速度)
+ CVD burst 或 Volume spike
```

2K 加速度做微观确认，抓住加速启动瞬间。

## 四、ME 禁止包含的特征

| 禁止特征 | 原因 |
|---|---|
| SR (支撑阻力) | 结构位置 → BPC 语义 |
| HAL / Fib | 结构位置 |
| LVN (Liquidity Void) | 结构位置 |
| Pullback depth | BPC 回踩结构 |
| Breakout level | BPC 突破结构 |
| BB compression | BPC 前置压缩 |

否则会无意识把 ME 拉回结构模型。

## 五、可复用但语义正交的特征

以下特征在 BPC 也有使用，但对 ME 语义解读不同，**允许复用**（由树模型决定是否有用）：

| 特征 | BPC 语义 | ME 语义 |
|---|---|---|
| WPT ignition/exhaustion | 小波结构能量 | 动量释放/衰竭 |
| Hurst | 结构完整性 | 动量持续性 |
| Path efficiency | 趋势结构 | 方向效率 |
| VPIN | 入场确认 | 信息交易活跃度 |
| EVT | 尾部风险 | 极端动能概率 |
| Spectrum | 频谱结构 | 频谱动能分布 |

## 六、特征数量

| 函数 | 输出数 | 用途 |
|---|---|---|
| Core (soft_phase) | 11 | Energy(2) + Acceleration(4) + Participation(5) |
| Gate | 4 | expansion_ok, flow_ok, volume_ok, gate_pass |
| Evidence | 1 | 乘法组合信号 |
| Entry | 3 | micro_accel, flow_burst, confirm |
| Failure | 4 | false_expansion, vol_divergence, flow_exhaustion, failure_score |
| Context | 3 | jump_risk, reflex_risk, regime_suitable |
| **合计** | **26** | |

## 七、测试要求

1. **基本功能**: 输出列存在、值范围正确
2. **未来函数检测**: 修改未来数据不影响历史计算
3. **流式一致性**: 全量 vs 增量结果一致
4. **NaN 安全**: 缺失输入时输出中性值
