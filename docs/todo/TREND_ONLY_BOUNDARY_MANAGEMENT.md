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

## ✅ TODO 1: Symbol-Level Execution Allowlist（优先级最高）

**Purpose**: Implement symbol-level filtering for TC-only execution

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

**Implementation**:
```yaml
# meta_router_live_config.yaml
tc_symbol_allowlist:
  enabled: [BTCUSDT, BNBUSDT]
  shadow: [ETHUSDT, XRPUSDT]
  disabled: [SOLUSDT, ADAUSDT]
```

**Why Legal**:
- ❌ No changes to router / score / guardrail
- ❌ No changes to execution logic
- ✅ Only **asset universe selection** (market selection, not alpha tuning)

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
