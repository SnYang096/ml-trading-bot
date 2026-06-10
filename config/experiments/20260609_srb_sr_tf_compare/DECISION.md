# SRB SR 时间周期对比 — DECISION（待填）

> **流程**：Phase 1 扫描 → 人读 `quick_scan/*.md` → 定 τ/lookback → 
> 创建 `config_experiments/srb_l2_only_strategies/` 等静态树 →
> Phase 3 grid 回测 → trading maps 人审 → 再填本 DECISION。

## 当前配置（基线）

```yaml
# config/strategies/srb/archetypes/prefilter.yaml
rules:
  - feature: sr_strength_max          # L2 (160-bar POC) 结构强度
    value: 0.42
  - feature: spectrum_price_high_freq_ratio  # 突破频谱
    value: 0.22
  - feature: srb_l3_breakout_age_decay  # L3 (240-bar wide SR) 真突破新鲜窗口
    value: 0.35
```

**核心疑问**：`srb_l3_breakout_age_decay >= 0.35` 这个 L3 大周期门是否过于严格？

- L3 窗口：wide_window=240 bar (~20天), anchor_shift=12 → **视角覆盖 ~1个月**
- 如果 L2（160-bar POC ~2周）的 `sr_strength_max` 已经能有效区分关键位，
  L3 门槛可能在 **过度过滤** 潜在的盈利 trades。

## Phase 1 结果（待跑）

| 扫描 | 目标 | 产出 |
|------|------|------|
| `sr_strength_max_plateau.md` | L2 强度的 plateau 拐点 | τ_L2 |
| `l3_breakout_age_decay_plateau.md` | L3 年龄衰减的 plateau 拐点 | τ_L3 |
| `wide_sr_dist_atr_plateau.md` | L3 距离的 plateau 拐点 | τ_wide |
| `spectrum_high_freq_ratio_plateau.md` | 频谱的 plateau 拐点 | τ_spec |
| `l2_vs_l3_condition_set.md` | L2-only vs L2+L3（当前）的 label 质量对比 | ΔR |
| `bull_l2_vs_l3_condition_set.md` | 牛市下 L2 vs L2+L3 | ΔR_bull |
| `bear_l2_vs_l3_condition_set.md` | 熊市下 L2 vs L2+L3 | ΔR_bear |
| `ic_decay_sr_core.md` | 核心 SR 列 vs forward_rr 的 IC | IC decay |
| `strength_x_l3_age_pair.md` | strength × L3 age 交互 | pair matrix |

**关键读数**（跑完后填）：
- L2-only 的 label meanR 是否 ≥ L2+L3_prod？
- L3 age_decay 的 plateau 拐点是否 < 0.35？
- Bul/bear 分段是否对称？

## Phase 3 变体（待定）

读 Phase 1 结果后确定以下 3 个变体：

| 变体 | Prefilter | 假设 |
|------|-----------|------|
| **A_prod** (基线) | sr_strength_max≥0.42 AND spectrum≥0.22 AND l3_age_decay≥0.35 | 当前生产 |
| **B_l2_only** | sr_strength_max≥τ_L2 AND spectrum≥τ_spec | 去掉 L3，仅靠 L2+频谱 |
| **C_l3_relaxed** | sr_strength_max≥0.42 AND spectrum≥0.22 AND l3_age_decay≥τ_L3_low | L3 存在但放宽 |

## Phase 3 结果（待跑）

| variant | bear R | bull R | recent R | sum R | maxDD | trades |
|---------|--------|--------|----------|-------|-------|--------|
| A_prod  |        |        |          |       |       |        |
| B_l2_only |      |        |          |       |       |        |
| C_l3_relaxed |    |        |          |       |       |        |

## Trading Map 人审（Phase 4）

TODO：跑完 Phase 3 后，用 `run_trading_maps.sh` 生成 maps，
人审入场语义是否正确（突破位、回踩位、假突破过滤）。

## 最终决策

- **Promote**：TODO
- **Reject**：TODO
- **理由**：TODO