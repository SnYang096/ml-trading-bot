# TREND-Only Boundary Management TODO

**Status**: Execution improvements (within Phase 4-C boundary)  
**Goal**: Manage TREND-only system boundaries, not optimize the model  
**Rule**: ✅ **ONLY within Phase 4-C causal loop, NO model changes**

---

## Core Principle

> **Trend-only 不是"有问题"，而是"已经暴露了它的真实边界"。  
> 现在不应该"整体优化"，而应该只做"边界内的定向修复"。**

**Key Principles**:
- ❌ 不需要"重做 trend"
- ❌ 不需要"再调 router / score / guardrail"
- ✅ 只需要 **承认 trend-only 不是对所有 symbol / 所有趋势都成立**
- ✅ 并在 *不破坏 Phase 4-C 因果闭环* 的前提下做**有限改进**

---

## Physics 概念（架构层理解）

**核心转变**：
> **从"策略拼装" → "物理系统设计"**

**Physics 定义**：
> **Physics 不是 symbol，也不是 archetype，  
> 而是：一套"执行物理假设（execution assumptions）"的集合。**

**当前系统本质**：
> **Physics-TREND-CONTINUATION**

它隐含的物理假设：
- 价格是 **平坦高原 + 少量延伸**
- 回撤是 **可接受的**
- 极端反转是 **低概率事件**
- 交易是 **少而精**
- 错一次，成本很高 → **必须严格过滤**

**正确关系**：
```
(archetype) → physics
(symbol, physics) → allow / deny
```

**为什么 BTC/BNB 稳，SOL 爆**：
👉 不是模型问题，是 **物理不匹配**。  
BTC/BNB 是 TREND 物理的稳定载体，SOL 不是。

**下一步**：
- 在代码里显式引入 Physics 概念（enum + config）
- 将当前系统命名为 `Physics-TREND / Execution Guardrail v0`
- 将 MEAN 永久迁移到 `Physics-EXTREME_MEAN`（research-only）

---

---

## ✅ TODO 1: Symbol-Level Execution Allowlist（优先级最高）

**Purpose**: Implement symbol-level filtering for TC-only execution

**Physics 视角**：
> 这不是简单的 symbol 过滤，而是 **Symbol × Physics 合法性矩阵**。  
> BTC/BNB 是 TREND 物理的稳定载体，SOL 不是。

**Classification** (based on Phase 4-C-Split results):

### TC-Enabled Symbols (Strong Trend Assets)
- **BNBUSDT**: Sharpe 4.06 ✅
- **BTCUSDT**: Sharpe 1.12 ✅

### Shadow / Reduced-Risk (Weak Trend, Monitor)
- **ETHUSDT**: Sharpe ~0 (near break-even)
- **XRPUSDT**: Sharpe -0.02 (near break-even)

### Disabled (Structurally Unfriendly)
- **SOLUSDT**: Sharpe -1.22 ❌
  - High volatility
  - Frequent trend breaks
  - False expansion signals
  - Deep drawdowns

**Implementation** (Physics-aware):
```yaml
# meta_router_live_config.yaml
physics:
  TREND:
    description: "Continuation / Expansion in plateau markets"
    max_dd: 0.12
    min_hold_bars: 8
    score_floor_q: 0.05
    allow_switch: false

archetype_physics:
  TrendContinuationTC: TREND
  TrendExpansionTE: TREND

symbol_physics_allow:
  TREND:
    allow: [BTCUSDT, BNBUSDT]
    shadow: [ETHUSDT, XRPUSDT]
    deny: [SOLUSDT, ADAUSDT]
```

**Why Legal**:
- ❌ No changes to router / score / guardrail
- ❌ No changes to execution logic
- ✅ Only **asset universe selection** (market selection, not alpha tuning)
- ✅ **Physics 显式化**（架构层改进，不是模型调优）

**Acceptance Criteria**:
- Sharpe ≥ 1.8 (focusing on strong trend assets)
- Max DD ≤ -8% (reduced from -10%)
- Symbol consistency improved

**Status**: 🔴 Pending

---

## ✅ TODO 2: TC Trades Exit/Holding Policy Analysis（优先级次之）

**Purpose**: Diagnose holding/exit quality (NOT entry signals)

**Analysis Framework** (without changing router):

### 1. MFE/MAE Distribution
- **Question**: Do we have "correct direction, but drawdowns eat all profits"?
- **Metrics**: MFE vs MAE ratio, profit erosion analysis

### 2. Loss Concentration
- **Question**: Are losses concentrated in:
  - Late trend (trend exhaustion)?
  - After volatility expansion?
- **Metrics**: Loss distribution by trend stage, volatility regime

### 3. Holding Time Analysis
- **Question**: Is there "short positions positive, long positions negative"?
- **Metrics**: Sharpe by holding time buckets, optimal exit timing

**Expected Outcome**:
- If found: **Execution holding policy issue**, NOT signal issue
- Action: Adjust exit rules / trailing stops / max holding time
- ❌ **Do NOT change entry signals** (router is fine)

**Implementation**:
- Diagnostic script: `scripts/diagnose_tc_exit_holding.py`
- Output: MFE/MAE analysis, holding time distribution, loss decomposition

**Acceptance Criteria**:
- Clear identification of holding policy issues
- Proposed exit rule adjustments
- No changes to entry signals

**Status**: 🔴 Pending

---

## ⚠️ TODO 3: TE Controlled Reintroduction（待定，仅当 TC-only 稳定后）

**Purpose**: Add TE as additive layer (NOT to replace TC)

**Prerequisites**:
> "TC-only 在可交易 symbol 上稳定成立"

**Correct Approach**:
- TE **only as additive layer**
- Must satisfy:
  - Does NOT lower TC-only Sharpe
  - Does NOT significantly increase switch_rate
- Evaluation:
  - `TC-only` vs `TC+TE` (head-to-head)
  - **NOT compared to historical All-archetype**

**If TE lowers overall**:
> **Direct rollback, no hesitation.**

**Status**: 🟡 Waiting for TC-only stability

---

## ✅ TODO 4: 链路 KPI 诊断清单维护

**Purpose**: 固化“层级分工”的诊断边界，避免 E2E 反调 Physics  

**Checklist Reference**:
- `docs/guides/THRESHOLD_PLATEAU_TUNING_PROTOCOL_CN.md` → 七.4.16

**Status**: 🔴 Pending

---

## ❌ Should NOT Do (Red Lines)

### 1. ❌ Comprehensive "Trend Model Improvement"
- ❌ Do NOT retrain router
- ❌ Do NOT change score geometry
- ❌ Do NOT add new trend features

**Reason**:
> **You already have Sharpe 1.66 baseline.  
> Changing model = destroying your only established anchor.**

---

### 2. ❌ Modify TE Now
**Reason**:
- Current TC-only not yet stable (symbol differentiation obvious)
- Complete TC-only + symbol filter stability assessment first
- Only when TC-only stable, TE becomes an *additive lever*

---

### 3. ❌ Tune Execution Gate / Score / Floor
**Reason**:
- Current gate already stable
- Continued tuning pollutes causal judgment
- Already exceeded benefit ceiling of these actions

---

## Key Mental Shift

> **Not "system must cover all markets",  
> But "system must clearly know where it should NOT work".**

**Current System Has**:
- ✅ Self-pruning capability (MEAN isolated)
- ✅ Causal explainability (Phase 4-C complete loop)
- ✅ Operational baseline (TREND-only production baseline)

**This is more important than "doing a bit more Sharpe".**

---

## Priority Order

1. **TODO 1**: Symbol Allowlist (highest priority, most certain benefit)
2. **TODO 2**: Exit/Holding Analysis (needs diagnosis)
3. **TODO 3**: TE Reintroduction (wait for TC-only stability)

---

**Created**: 2025-01-XX  
**Status**: Active - Boundary Management Phase
