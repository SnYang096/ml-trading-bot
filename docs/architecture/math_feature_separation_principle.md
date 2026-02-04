# 数学特征分离原则（Math Feature Separation Principle）

## 概述

数学特征分离原则是路径2.5架构的核心设计理念，旨在解决数学特征（Hurst/WPT/Spectrum/Hilbert/EVT等）在交易系统中的滥用问题。该原则规定数学特征只能在特定层级使用，不得参与决策层的判断，以避免数据泄露和过拟合。

## 分离原则

### 1. Gate层（❌ 禁止使用）
- **禁止特征**：Hurst指数、WPT变换、Spectrum分析、Hilbert变换、EVT极值理论等数学特征
- **允许特征**：结构特征、订单流特征、规制特征
- **目的**：避免数学特征将failure压到0，防止过拟合

### 2. Evidence层（❌ 禁止使用）
- **禁止特征**：execution_noise_penalty、原始数学特征
- **允许特征**：结构/订单流/规制特征
- **目的**：保持alpha质量评估的纯粹性，仅评估"是否值得交易"

### 3. Execution层（✅ 唯一合法使用位置）
- **必须使用**：evidence_score（来自Evidence层）和noise_penalty（基于数学特征）
- **动态调整**：sl_r、tp_r、size_multiplier等参数
- **目的**：根据市场噪声调整"如何执行"，不影响"是否执行"

## 权重分配原则

### 基于功能分工的权重设计：
- **WPT (0.35)**：多尺度分解，结构破碎度主信号
- **Spectrum (0.30)**：频域分析，频谱稳定性
- **Hilbert (0.20)**：相位/包络不稳定性
- **Hurst (0.15)**：长期记忆性，背景环境指标

### EVT特征处理：
- **不采用线性加权**：而是作为"保险丝"机制
- **触发条件**：当EVT尾部风险超过阈值时，额外增加惩罚
- **目的**：在极端市场条件下提供额外保护

## 实现架构

### BPC策略V2架构
```
Gate层决策 → Evidence层评估 → Execution层参数调整
     ↓              ↓                ↓
结构/订单流    结构/订单流      evidence_score + noise_penalty
特征决策      alpha质量评估      → Tier选择 → 参数调整
```

### 核心组件
- **BPCEvidenceCalculator**：仅基于结构/订单流特征计算证据分数
- **ExecutionNoisePenalty**：基于数学特征计算噪声惩罚因子
- **ExecutionController**：整合evidence_score和noise_penalty
- **TierSelector**：根据evidence_score选择档位，应用noise_penalty调整

## 配置文件分离

### Gate/Evidence配置
- `config/strategies/bpc/features.yaml`：仅包含结构/订单流/规制特征
- 数学特征已从此配置中移除

### Execution配置
- `config/execution/bpc_execution_tiers.yaml`：包含noise_penalty配置
- 定义权重分配和参数调整规则

## 验证要点

1. **Gate层不使用数学特征**：确保决策层纯净
2. **Evidence层不使用数学特征**：保持alpha评估纯粹
3. **Execution层同时消费evidence_score和noise_penalty**：正确实现分层
4. **噪声惩罚值在[0, 0.8]区间内**：防止完全阻断交易
5. **系统能正常处理交易决策和参数调整**：功能完整性

## 核心理念

> "一个成熟系统的标志不是'用了多少高级特征'，而是'每个特征只在它该说话的地方说话'"

数学特征只在Execution层"说话"，影响"如何执行"，而在Gate和Evidence层保持"沉默"，不影响"是否执行"的决策。