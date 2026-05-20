# Regime和Gate实验对比报告

## 实验元信息

- **实验时间**: 2026-01-22 03:00:08
- **实验目的**: 对比不同regime过滤和gate veto配置的KPI表现
- **数据时间范围**: 2025-05-01 到 2025-10-31
- **数据Symbols**: ADAUSDT, BNBUSDT, BTCUSDT, ETHUSDT, SOLUSDT
- **总样本数**: 2930
- **FeatureStore Layer**: `nnmh_highcap6_240T_2024_202510`
- **Timeframe**: 240T (4H)

---

# Regime and Gate Experiment Comparison Report

## Experiment Overview

This report compares 6 configurations:

1. **Baseline**: With regime filter + with gate veto + with semantic veto
2. **No Regime Filter**: Without regime filter + with gate veto + with semantic veto
3. **No Gate Veto**: With regime filter + without gate veto + with semantic veto
4. **No Semantic Veto**: With regime filter + with gate veto + without semantic veto
5. **No Regime No Veto**: Without regime filter + without gate veto + with semantic veto
6. **All Veto Off**: Without regime filter + without gate veto + without semantic veto

---

## Overall KPI Comparison

| Configuration | Sharpe | Trades | Win Rate | Profit/Loss Ratio |
|--------------|--------|--------|----------|-------------------|
| baseline | 2.565 | 1074 | 34.3% | 1.17 |
| only_gate_rules | 2.276 | 1146 | 34.1% | 1.17 |
| no_regime_filter | -1.516 | 2611 | 37.2% | 0.91 |
| no_gate_veto | 1.902 | 1373 | 35.9% | 1.09 |
| no_semantic_veto | 2.276 | 1146 | 34.1% | 1.17 |
| no_regime_no_veto | -0.001 | 11284 | 37.7% | 1.00 |
| all_veto_off | -0.045 | 11720 | 37.7% | 1.00 |

---

## Detailed Results

### baseline

**Description**: With regime filter + with gate veto + with semantic veto (baseline)

- Gated file: `results/experiments_optimized_v2/baseline_gated.parquet`
- KPI report: `results/experiments_optimized_v2/baseline_kpi.md`

#### Per Symbol Summary

| Symbol | Sharpe | Trades | Win Rate | Profit/Loss Ratio |
|--------|--------|--------|----------|-------------------|
| ADAUSDT | -2.643 | 66 | 10.6% | 1.31 |
| BNBUSDT | 4.340 | 165 | 44.8% | 1.07 |
| BTCUSDT | 3.121 | 364 | 45.1% | 1.19 |
| ETHUSDT | 1.977 | 407 | 26.0% | 1.21 |
| SOLUSDT | 6.549 | 72 | 23.6% | 1.49 |

#### Per Archetype Summary

| Archetype | Sharpe | Trades | Win Rate | Profit/Loss Ratio |
|-----------|--------|--------|----------|-------------------|
| FR | 5.184 | 27 | 44.4% | 1.50 |
| TC | 2.477 | 1047 | 34.0% | 1.16 |
| TE | 0.000 | 0 | 0.0% | nan |

---

### only_gate_rules

**Description**: With regime filter + with gate veto only (no semantic veto)

- Gated file: `results/experiments_optimized_v2/only_gate_rules_gated.parquet`
- KPI report: `results/experiments_optimized_v2/only_gate_rules_kpi.md`

#### Per Symbol Summary

| Symbol | Sharpe | Trades | Win Rate | Profit/Loss Ratio |
|--------|--------|--------|----------|-------------------|
| ADAUSDT | -2.604 | 68 | 10.3% | 1.31 |
| BNBUSDT | 4.192 | 173 | 44.5% | 1.09 |
| BTCUSDT | 2.162 | 392 | 44.6% | 1.15 |
| ETHUSDT | 1.779 | 435 | 26.2% | 1.18 |
| SOLUSDT | 6.924 | 78 | 23.1% | 1.55 |

#### Per Archetype Summary

| Archetype | Sharpe | Trades | Win Rate | Profit/Loss Ratio |
|-----------|--------|--------|----------|-------------------|
| FR | 5.184 | 27 | 44.4% | 1.50 |
| TC | 2.183 | 1119 | 33.9% | 1.15 |
| TE | 0.000 | 0 | 0.0% | nan |

---

### no_regime_filter

**Description**: Without regime filter + with gate veto + with semantic veto

- Gated file: `results/experiments_optimized_v2/no_regime_filter_gated.parquet`
- KPI report: `results/experiments_optimized_v2/no_regime_filter_kpi.md`

#### Per Symbol Summary

| Symbol | Sharpe | Trades | Win Rate | Profit/Loss Ratio |
|--------|--------|--------|----------|-------------------|
| ADAUSDT | -4.548 | 148 | 16.2% | 0.75 |
| BNBUSDT | 2.143 | 247 | 44.9% | 1.01 |
| BTCUSDT | -2.604 | 1031 | 45.1% | 0.90 |
| ETHUSDT | -1.264 | 1058 | 32.0% | 0.89 |
| SOLUSDT | -2.242 | 127 | 24.4% | 1.03 |

#### Per Archetype Summary

| Archetype | Sharpe | Trades | Win Rate | Profit/Loss Ratio |
|-----------|--------|--------|----------|-------------------|
| FR | -3.576 | 1328 | 38.3% | 0.85 |
| TC | 1.197 | 1283 | 35.9% | 1.02 |

---

### no_gate_veto

**Description**: With regime filter + without gate veto + with semantic veto

- Gated file: `results/experiments_optimized_v2/no_gate_veto_gated.parquet`
- KPI report: `results/experiments_optimized_v2/no_gate_veto_kpi.md`

#### Per Symbol Summary

| Symbol | Sharpe | Trades | Win Rate | Profit/Loss Ratio |
|--------|--------|--------|----------|-------------------|
| ADAUSDT | -2.716 | 84 | 14.3% | 1.03 |
| BNBUSDT | 5.911 | 209 | 47.8% | 1.07 |
| BTCUSDT | 1.806 | 485 | 44.7% | 1.10 |
| ETHUSDT | 0.879 | 501 | 27.7% | 1.12 |
| SOLUSDT | 5.892 | 94 | 26.6% | 1.26 |

#### Per Archetype Summary

| Archetype | Sharpe | Trades | Win Rate | Profit/Loss Ratio |
|-----------|--------|--------|----------|-------------------|
| FR | 5.184 | 27 | 44.4% | 1.50 |
| TC | 1.469 | 640 | 34.7% | 1.11 |
| TE | 2.096 | 706 | 36.7% | 1.05 |

---

### no_semantic_veto

**Description**: With regime filter + with gate veto + without semantic veto (legacy, same as only_gate_rules)

- Gated file: `results/experiments_optimized_v2/no_semantic_veto_gated.parquet`
- KPI report: `results/experiments_optimized_v2/no_semantic_veto_kpi.md`

#### Per Symbol Summary

| Symbol | Sharpe | Trades | Win Rate | Profit/Loss Ratio |
|--------|--------|--------|----------|-------------------|
| ADAUSDT | -2.604 | 68 | 10.3% | 1.31 |
| BNBUSDT | 4.192 | 173 | 44.5% | 1.09 |
| BTCUSDT | 2.162 | 392 | 44.6% | 1.15 |
| ETHUSDT | 1.779 | 435 | 26.2% | 1.18 |
| SOLUSDT | 6.924 | 78 | 23.1% | 1.55 |

#### Per Archetype Summary

| Archetype | Sharpe | Trades | Win Rate | Profit/Loss Ratio |
|-----------|--------|--------|----------|-------------------|
| FR | 5.184 | 27 | 44.4% | 1.50 |
| TC | 2.183 | 1119 | 33.9% | 1.15 |
| TE | 0.000 | 0 | 0.0% | nan |

---

### no_regime_no_veto

**Description**: Without regime filter + without gate veto + with semantic veto

- Gated file: `results/experiments_optimized_v2/no_regime_no_veto_gated.parquet`
- KPI report: `results/experiments_optimized_v2/no_regime_no_veto_kpi.md`

#### Per Symbol Summary

| Symbol | Sharpe | Trades | Win Rate | Profit/Loss Ratio |
|--------|--------|--------|----------|-------------------|
| ADAUSDT | 0.078 | 716 | 21.2% | 0.95 |
| BNBUSDT | -1.102 | 1424 | 41.9% | 0.96 |
| BTCUSDT | 0.115 | 4216 | 46.1% | 1.02 |
| ETHUSDT | 0.170 | 4240 | 32.4% | 1.01 |
| SOLUSDT | 0.223 | 688 | 28.2% | 0.94 |

#### Per Archetype Summary

| Archetype | Sharpe | Trades | Win Rate | Profit/Loss Ratio |
|-----------|--------|--------|----------|-------------------|
| ET | -2.451 | 2821 | 38.2% | 0.90 |
| FR | -2.451 | 2821 | 38.2% | 0.90 |
| TC | 2.687 | 2821 | 37.2% | 1.13 |
| TE | 2.687 | 2821 | 37.2% | 1.13 |

---

### all_veto_off

**Description**: Without regime filter + without gate veto + without semantic veto (all veto off)

- Gated file: `results/experiments_optimized_v2/all_veto_off_gated.parquet`
- KPI report: `results/experiments_optimized_v2/all_veto_off_kpi.md`

#### Per Symbol Summary

| Symbol | Sharpe | Trades | Win Rate | Profit/Loss Ratio |
|--------|--------|--------|----------|-------------------|
| ADAUSDT | 0.145 | 744 | 21.0% | 0.96 |
| BNBUSDT | -1.085 | 1464 | 41.9% | 0.96 |
| BTCUSDT | -0.102 | 4396 | 45.9% | 1.01 |
| ETHUSDT | 0.165 | 4396 | 32.4% | 1.01 |
| SOLUSDT | 0.335 | 720 | 28.6% | 0.94 |

#### Per Archetype Summary

| Archetype | Sharpe | Trades | Win Rate | Profit/Loss Ratio |
|-----------|--------|--------|----------|-------------------|
| ET | -2.398 | 2930 | 38.2% | 0.91 |
| FR | -2.398 | 2930 | 38.2% | 0.91 |
| TC | 2.533 | 2930 | 37.2% | 1.12 |
| TE | 2.533 | 2930 | 37.2% | 1.12 |

---

## Conclusions

### Key Findings

1. **Regime Filter Impact**: Compare baseline vs no_regime_filter
2. **Gate Veto Impact**: Compare baseline vs no_gate_veto
3. **Combined Impact**: Compare baseline vs no_regime_no_veto

### FR/ET Trading Statistics

Check the detailed KPI reports for FR/ET archetype trading statistics.

