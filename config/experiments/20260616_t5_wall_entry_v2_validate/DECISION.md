# T5 wall entry v2 — DECISION（待 Phase 3）

## 设计修正（相对 v1）

- 保持 prod `combination_mode: or`
- wall / regime 作为 **可选 OR tier**，非全局 AND
- W6/W7 conditions 对齐 Phase 1c S5（ema±0.10 + chop + depth + wall τ）

## v1 对照（勿忘）

| variant | Σ R | Σ trades |
|---------|-----|----------|
| E0_prod | +7.91 | 206 |
| W1 (AND) | -0.68 | 103 |

## Phase 3 结果（待填）

| variant | bear R | bull R | recent R | Σ R | Σ trades |
|---------|--------|--------|----------|-----|----------|
| E0_prod | | | | | |
| W5_or_long2 | | | | | |
| W6_or_bull_pullback | | | | | |
| W7_or_regime_asym | | | | | |

## Promote?

- [ ] Σ R vs E0 过关（三阶段）
- [ ] maxDD 不恶化
- [ ] 笔数未塌缩（警惕 OR 过度放宽 → 需 trading map）
- [ ] trading map 语义（墙=支撑/阻力）
