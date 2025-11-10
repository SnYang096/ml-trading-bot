# 仓位与风险控制方案（Regime 感知 + 波动率目标 + 反马丁格尔）

目标：在严格控制风险与回撤的前提下，利用 Regime 与多模型信号，实现“牛市积极放大、熊市防守”、并通过反马丁格尔（胜后加仓、败后缩仓）提取收益曲线的凸性（convexity）。

适用范围：对接本仓库的两条主线
- 时序模型（单资产，多周期）：`src/time_series_model/*`
- 横截面模型（多资产，单时点）：`src/cross_sectional/*`

与现有代码的衔接点
- 风控模块：`time_series_model/pipeline/risk_management.py`（需增强）
- Regime：`regime_detection/*`（已具备规则+HMM；建议补充分类器与概率输出）
- 多周期管线：`time_series_model/pipeline/multi_tf_pipeline.py`

---

## 1. 仓位核心框架（组合级）

```text
原始模型信号（方向/收益/分位/波动） 
     └→ 概率校准（Platt / Isotonic）→ 期望收益 E[r̂] 与成功率 p
         └→ 多周期一致性/强度合成 → 原始仓位建议 s0
             └→ Regime 权重 g(Regime)（牛>震>熊） → s1
                 └→ 波动率目标（target vol）规模化 → s2
                     └→ 反马丁格尔加仓/失败冷却 → s3
                         └→ 组合约束（总敞口/单品种/相关/流动性）→ s*
                             └→ 执行层（滑点/TTL/抑抖/结构止损&分段止盈）
```

定义记号：
- p：方向成功概率（经校准）
- r̂：收益回归预测值或分位带合成的期望
- σ̂：预测波动率或历史 realized vol
- Regime ∈ {RANGE, PRE_BREAKOUT, TRENDING, COLLAPSE}，并有概率分布

---

## 2. 基础仓位函数（单资产基线）

基础风险预算（每笔占净值比例）：
- base_risk_per_trade ∈ [0.3%, 1.0%]（根据策略频率）

期望值与置信度缩放：
```math
s0 = base\_risk\_per\_trade × (2p - 1)\_+ × clip(|r̂| / r\_target, 0, m\_exp)
```
建议：r_target 为经验目标（如 1–2×日波动），m_exp ∈ [1, 3]

波动率规模化（vol targeting，两种等效形式）：
```math
s1 = s0 × \frac{target\_vol}{σ̂}    （名义仓位按波动倒数缩放）
```
或
```math
notional = s0 × \frac{E[|r̂|]}{σ̂}  → 转换为头寸手数
```
约束：`s1 ∈ [0, s_max]`，建议 s_max ∈ [1.5%, 3%] per trade（按净值）

方向：`sign = sign(r̂)` 或由离散方向模型给出；最终单品种头寸 `pos = sign × s1`

---

## 3. Regime 感知的仓位调度

用 Regime 概率对仓位做软加权（而非硬切换），在牛市（TRENDING）放大风险，在崩塌/高风险 Regime 收缩：

```math
g(Regime) = α\_{trend}·P(TRENDING) + α\_{pre}·P(PRE\_BREAKOUT) 
            + α\_{range}·P(RANGE) + α\_{col}·P(COLLAPSE)
```

建议权重（可回测优化，保持平坦高原区间）：
- 牛市：α_trend = 1.5
- 预突破：α_pre = 1.2
- 震荡：α_range = 0.8
- 崩塌：α_col = 0.4

组合到仓位：
```math
s2 = s1 × clip(g(Regime), g_{min}, g_{max})
```
建议：g_min = 0.5，g_max = 1.8

额外守则（强防守）：
- 若 `P(COLLAPSE) > 0.6` 或 波动分位 > 0.9：全局 risk_mode = Defensive（下文）

---

## 4. 反马丁格尔（胜后加仓，败后缩仓）

胜后阶梯加仓，失败后冷却：
- 参数：`max_adds = 2~3`，`max_mult = 3~4`
- 加仓条件（示例，需回测验证）：
  - 新 p ≥ p_add（如 0.8）
  - 同向的 1h/4h 确认（多周期一致性）
  - 流动性门槛满足（volume / depth）

加仓公式（第 k 次加仓）：
```math
add\_size\_k = s2 × λ\_k,    λ\_k ∈ [0.5, 1.5]，且 ∑λ\_k ≤ (max\_mult - 1)
```

失败后：
- 结构止损或 hard stop 被触发 → 进入 cooldown（6–24 小时）内禁止加仓或仅允许半仓
- 连亏 N（如 3） → risk_mode 自动降级

---

## 5. 风险模式（risk_mode）与全局约束

根据 Regime、回撤与实盘表现动态切换：
- Aggressive：`g(Regime)` 高且 drawdown < ½阈值 → `leverage_cap ↑`（如 1.5–2.0×）
- Normal：默认
- Defensive：`P(COLLAPSE)` 高、特征/校准漂移、或回撤超阈 → `leverage_cap ↓`（如 0.3–0.7×）、禁止加仓

组合级约束：
- `max_total_exposure`（如 2× notional）
- `max_per_asset`（如 10–15% of equity）
- `beta_cap`：若与 BTC Index 的 rolling_corr > 0.8 → 降权或对冲
- Turnover & cost budget：每日/每周换手、成本上限

---

## 6. 止损/止盈（结构优先）

结构止损优先于硬止损：
- 结构失败（ZigZag/通道/POC 等被破坏） → 立即减仓/平仓
- 兜底硬止损：`k1 × σ̂`（如 2–3×）

分段止盈（提取凸性）：
- 在 `k2 × σ̂`、`k3 × σ̂`（如 2×、3.5×）分档减仓，留尾仓跟踪（trailing stop = f(σ̂, unrealized PnL))

---

## 7. 横截面层的风险与权重

横截面权重 `w_i` 来自 rank/回归分数并经过中性化（size/行业/Beta）：
```math
w_i' = normalize( rank\_score_i )
w_i = neutralize(w_i', beta, sector, size)
```
组合波动目标：
```math
w_i^{final} = w_i × \frac{target\_vol}{\hat{σ}_{portfolio}} × g(Regime)
```
并叠加交易成本与最小成交量约束，实施权重平滑（半衰期）。

---

## 8. 推理阶段融合策略（时序 × 横截面）

- 横截面给出资产选择与基准权重区间（强/中/弱）
- 时序给出入场/加仓/减仓节奏与方向过滤
- 冲突时降权一致性差的信号；一致时允许最大仓位（受全局约束）

---

## 9. 线上守护与自适应

- 概率校准在线监控：Brier / calibration curve 偏差超阈 → 降权
- 漂移与健康度：特征分布、重要性 JS 散度、回归残差漂移 → 触发 Defensive 模式与再训练
- 回撤阈值：超阈（如 8–12%）自动降杠杆/暂停加仓；超大阈（如 15–20%）全平仓冷却
- 绩效看板：分桶胜率（按 p 与 |r̂|）、单位风险收益、交易成本占比、beta 暴露

---

## 10. 与代码的落地接口（建议修改点）

1) `risk_management.RiskManager` 增强：
   - inputs: calibrated `p`, `r_hat`, `sigma_hat`, `regime_probs`
   - sizing: base_risk → EV/vol scaling → regime gain → anti-martingale
   - states: `risk_mode`（Aggressive/Normal/Defensive），cooldown，drawdown tracker
   - constraints: account-level caps、per-asset caps、beta_cap
   - outputs: `position`, `stop_loss_level`, `take_profit_level`, `notes`

2) `multi_tf_pipeline`：
   - 提供 `regime_probs` 与多周期一致性分数
   - 提供 `expected_return`、`predicted_vol` 合成接口（q10/q50/q90 → 期望/分位带）

3) 横截面：
   - 在组合构建中加入：target_vol、regime overlay、权重平滑、成本/换手约束

---

## 11. 参数默认表（回测起点，可调）

- calibration: Platt（分类）/ 线性缩放（回归）
- base_risk_per_trade: 0.5%（保守）/ 1.0%（激进）
- target_vol（组合年化）: 30%（中频）；波动缩放 cap: [0.5×, 2.0×]
- regime weights: trend 1.5, pre 1.2, range 0.8, collapse 0.4（clip 0.5–1.8）
- max_adds=2, max_mult=3.5, p_add=0.8, cooldown=12h
- hard stop: 2.5×σ̂；分段止盈：2×σ̂、3.5×σ̂，尾仓用 trailing stop
- max_total_exposure=2.0×, max_per_asset=15% equity, beta_cap (BTC corr) ≤ 0.8

---

## 12. 回测与验证要点

- 使用 walk-forward / purged CV + embargo 调仓参数（阈值、regime weights、target_vol）
- 选“平坦高原”而非单点最优；多市场/多阶段验证
- 成本/滑点模型必须启用；对冲与暴露上限约束常开
- 线上 A/B：对 sizing 策略（有/无反马丁格尔、不同 regime 权重）开启影子账户比较

---

## 13. 快速伪代码（时序侧单资产）

```python
def position_sizing(p, r_hat, sigma_hat, regime_probs, state):
    # 1) EV & vol scaling
    base = state.base_risk_per_trade
    s0 = base * max(0.0, 2*p - 1) * clip(abs(r_hat)/state.r_target, 0, state.m_exp)
    s1 = s0 * clip(state.target_vol / max(sigma_hat, 1e-6), state.vol_min, state.vol_max)

    # 2) regime gain
    g = (1.5*regime_probs["TRENDING"] +
         1.2*regime_probs["PRE_BREAKOUT"] +
         0.8*regime_probs["RANGE"] +
         0.4*regime_probs["COLLAPSE"])
    s2 = s1 * clip(g, 0.5, 1.8)

    # 3) anti-martingale
    if state.last_trade_profitable and state.add_count < state.max_adds:
        s3 = s2 * state.add_ladder[state.add_count]
    else:
        s3 = s2

    # 4) constraints
    s = apply_constraints(s3, state)  # total exposure, per-asset, beta cap
    direction = 1.0 if r_hat >= 0 else -1.0
    return direction * s
```

---

以上方案与“timeframe/forward 固定、降维与滚动协同、Regime 引导专家模型”的总体架构一致，重点强化了“牛市积极放大、熊市防守”的仓位与风控体系，并给出明确可实现的参数与接口。***

