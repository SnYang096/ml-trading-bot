# Archetype 上线前 Checklist（v0）

> 目的：把 TC/TE/FR/ET 在上线前必须验证的项目固化，避免“只看 Sharpe”。

---

## 一、四种 Archetype 的最常见失败模式（必须写清楚）

> 这些来源于 `docs/architecture/6种archetype简化成4种的原因.md` 与  
> `docs/archive/architecture/archetype灭绝级回测.md` 的约束假设。

- **TC（Trend Continuation）**  
  - 最常见失败：**假结构**（结构确认错误 / 回踩后失效）
  - 常见触发：高位追涨、回踩深度过深、趋势 R2 走弱
  - 典型信号：CVD 续航不足、bb_width 压缩、成交量萎缩
- **TE（Trend Expansion）**  
  - 最常见失败：**假加速**（动能衰竭 / 爆发后无延续）
  - 常见触发：量能未放大、波动未扩张、突破后被吸收
  - 典型信号：volume_ratio 偏低、bb_width 扩张不足、CVD 回落
- **FR（Failure Reversion）**  
  - 最常见失败：**回归假设错误**（世界不回归，趋势延续）
  - 常见触发：缺少吸收/反转证据、流动性不足、vpin 不显著
  - 典型信号：cvd 仍在扩张、回撤浅、反转后无跟随
- **ET（Exhaustion Turn）**  
  - 最常见失败：**衰竭判断错误**（趋势续命）
  - 常见触发：高潮量能非顶、波动回落但趋势继续
  - 典型信号：atr_percentile 未极值、vpvr_lvn_distance 偏大

---

## 二、交易样本充足性（硬门槛）

- ✅ 每个 archetype 至少 **100+ 笔真实交易样本**  
  - 数据源：实盘 / 回测 / 事件驱动日志  
  - 统计字段必须包含：symbol、timestamp、archetype、entry/exit
  - 最新回测样本（`tier01_highcap6_2024H1_20260118_gate_strong5`）：
    - allow=482；FR=280、ET=202、TC=0、TE=0
    - 真实交易样本仍需补齐（当前仅回测）

---

## 三、使用频率分布（避免极端偏置）

- ✅ 统计 **archetype 使用频率分布**（占比、rank、熵）  
  - 是否出现长期 1 个 archetype 独占
  - TE / ET 作为高风险，应低频
  - FR / TC 在正常区间应稳定出现
  - 最新统计（`tier01_highcap6_2024H1_20260118_gate_strong5`）：
    - allow=482；FR=280（58.1%）、ET=202（41.9%）；熵=0.68

---

## 四、互斥约束检查（物种隔离）

> 物种隔离必须可执行、可审计、可追踪。

- ✅ TE 与 FR **不同时 active**
- ✅ TE 与 ET **不同时 active**
- ✅ TC 与 TE **不同时重仓**
- ✅ FR 与 TC **不同时高权重**

---

## 五、Gate 规则强度（可组合约束算子）

- ✅ Gate 使用 `deny_if / allow_if` 可组合约束算子  
- ✅ TC/TE = `deny_if` 宽松 veto  
- ✅ FR/ET = `allow_if` 严格放行（default deny）
- ✅ `allow_mode` 支持 `any / all / min2`（至少 2 个证据）
- ✅ `deny_if` 优先级最高，任何 veto 直接拒绝
- ✅ quantile 规则必须有 `evidence_quantiles.json` 支撑

---

## 六、MEAN 持仓时间约束（避免“均值变趋势”）

- ✅ MEAN 的 `max_holding_bars` 必须显式限制
- ✅ MEAN 禁止 add-on（或严格限制）
- ✅ MEAN 的持仓时间分布应集中在短窗口

---

## 七、互斥与排序（实现级注意）

- ✅ TE 与 FR/ET 互斥（同一时刻只允许一个）  
- ✅ TC 与 TE 互斥（趋势方向不重复下注）  
- ✅ 互斥应在 gate 选择阶段解决（按候选顺序逐个 gate 放行）  
- ✅ 必须记录 `gate_decision` 与 `gate_reasons` 便于审计

---

## 八、上线前最小证据链（建议项）

- ✅ 每个 archetype 的 gate 触发率、veto 率
- ✅ 每个 archetype 的 per-symbol KPI（AUC/IC/稳定性）
- ✅ 失败模式对齐验证（是否与预期失败模式一致）

