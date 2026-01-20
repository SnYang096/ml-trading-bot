# MEAN Research Track (Research-Only, Not Execution)

**Status**: Research-only track, parallel to execution development  
**Goal**: Understand MEAN failure and redesign if needed  
**Rule**: ❌ **NEVER enter execution, NEVER affect allow_rate, NEVER participate in performance summary**

---

## Phase M1: Failure Decomposition（失败解剖）

**Purpose**: Answer "MEAN 到底输在什么条件下？"

**Required Analysis** (3 tables):

### 1. By Market Regime
- High volatility vs Low volatility
- Trend period vs Ranging period
- Market state classification

### 2. By Entry Distance
- Distance to MA / VWAP / SR (z-score)
- Whether truly "far enough" from support/resistance
- Entry quality metrics

### 3. By Holding Time
- Short positions (< N bars) vs Long positions (≥ N bars)
- Exit timing analysis
- "MEAN 很多时候不是输方向，是输在拿太久"

**Expected Outcome**:
- If found: "短持仓是正的，长持仓是负的"
  → Execution hypothesis directly falsified
- Clear understanding of failure conditions

---

## Phase M2: Hypothesis Revision（假设层纠偏）

**Purpose**: Identify and correct wrong assumptions (NOT tuning thresholds)

**Typical Wrong Assumptions**:

### ❌ Error 1: Counter-trend as Mean-Reversion
- Reality: Just "counter-trend in weak trend"
- Fix: Better regime classification

### ❌ Error 2: Structural Pullback as Mean Reversion
- Reality: No microstructure support
- Fix: Add microstructure features

### ❌ Error 3: Same MEAN for All Markets
- Reality: MEAN **strongly depends on symbol characteristics**
- Fix: Symbol-specific MEAN rules

**Action**:
- NOT tuning k or thresholds
- Write one sentence: **"MEAN 只在什么情况下成立？"**

---

## Phase M3: MEAN Archetype Redesign（重新定义 MEAN archetype）

**Purpose**: Restructure MEAN from monolithic to specialized archetypes

**Possible Split**:

- **Liquidity Reversion**（流动性回归）
  - Focus: Liquidity-driven mean reversion
  - Features: Order flow, volume profile, liquidity voids

- **Volatility Exhaustion**（波动率耗尽）
  - Focus: Volatility-driven reversals
  - Features: Volatility compression, ATR expansion, volatility regime

- **Range Micro-Mean**（区间内微均值）
  - Focus: Range-bound trading
  - Features: Support/resistance quality, range boundaries, range persistence

**Instead of**: One monolithic `MEAN/FR` archetype

**This is**: **Archetype reconstruction**, not bug fixing

---

## Phase M4: Shadow Mode Backtest（Shadow Mode 回测）

**Purpose**: Test all new MEAN versions without affecting production

**Rules**:
- ❌ **DO NOT enter execution**
- ❌ **DO NOT affect allow_rate**
- ❌ **DO NOT participate in performance summary**
- ✅ **ONLY run in shadow mode**

**KPI**:
> **Conditional Sharpe (在它声称成立的子空间)**

**Process**:
1. Define claimed subspace (e.g., "high SR quality + low volatility")
2. Filter logs to that subspace
3. Compute conditional Sharpe
4. If positive → consider for Phase M5 (integration test)
5. If negative → return to Phase M2/M3

---

## Research Discipline（研究纪律）

### ✅ Parallel Track (Research)
- 📊 MEAN failure decomposition
- 🧠 Rewrite MEAN hypothesis
- 🧪 Shadow backtest

### ❌ NOT Execution Track
- ❌ Not in execution
- ❌ Not affecting allow_rate
- ❌ Not participating in performance summary

---

## Key Principle（关键原则）

> **一个成熟系统的标志，不是"什么都能做"，
> 而是"知道什么时候不该做"。**

**Current State**:
- TREND path is **successful** (Sharpe 1.36-1.82)
- Execution is **clean, stable, explainable**
- MEAN needs to **return to its proper place** (research-only)

---

## Next Steps

1. **Phase M1**: Run failure decomposition analysis
2. **Phase M2**: Revise assumptions based on M1 findings
3. **Phase M3**: Redesign MEAN archetypes if needed
4. **Phase M4**: Shadow backtest all new versions
5. **Phase M5**: Integration test (only if M4 shows promise)

---

**Created**: 2025-01-XX  
**Status**: Research-only, parallel to execution development
