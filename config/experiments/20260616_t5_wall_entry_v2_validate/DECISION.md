# T5 wall entry v2 — DECISION

## 设计修正（相对 v1）

- 保持 prod **`combination_mode: or`**（顶层 filter 之间并联）
- wall / regime 作为 **可选 OR tier**，非全局 AND
- 单个 wall tier **内部** conditions 为 AND（ema + chop + depth + wall τ）

### W7 entry 组合语义（回答「都是 or 吗？」）

| 层级 | 模式 | 说明 |
|------|------|------|
| **顶层** `combination_mode` | **OR** | `vol_confirm` **或** 牛墙 tier **或** 熊墙 tier，任一满足即可入场 |
| **tier 内** conditions | **AND** | 例：牛 tier = ema≥0.10 ∧ chop≤0.40 ∧ depth≤0.85 ∧ wall≤2.0 |
| **direction** | 分腿 | 牛 tier 仅 `long`；熊 tier 仅 `short`（另一方向 vacuous pass） |

prod `tpc_deep_pullback_vol_confirm` 路径 **保留**；W7 新增两条 regime 对齐的 wall OR 腿。

## v1 对照

| variant | Σ R | Σ trades |
|---------|-----|----------|
| E0_prod | +7.91 | 206 |
| W1 (AND) | -0.68 | 103 |

## Phase 3 结果（2026-06-16，BTC+ETH）

| variant | bear R | bull R | recent R | **Σ R** | Σ trades | worst maxDD |
|---------|--------|--------|----------|---------|----------|-------------|
| E0_prod | -4.17 | +4.95 | +7.13 | **+7.91** | 206 | -9.0% |
| W5_or_long2 | +3.84 | +1.49 | +9.90 | +15.23 | 229 | -6.4% |
| W6_or_bull_pullback | +3.84 | +1.49 | +9.90 | +15.23 | 229 | -6.4% |
| **W7_or_regime_asym** | **+7.65** | **+10.76** | **+12.01** | **+30.42** | 236 | -6.5% |

**W7 vs E0**：Δ **+22.51R**，三阶段均为正，笔数 +30，maxDD 略改善。

**W5 ≡ W6**：牛 S5 条件 tier 未产生额外成交（W5 无 regime 的 `wall≤2 long` OR 腿已覆盖）。

## Trading map（Phase 4）

```bash
bash config/experiments/20260616_t5_wall_entry_v2_validate/run_trading_maps.sh
```

产物（W7 vs E0 × 三阶段）：

`results/tpc/experiments/t5_wall_entry_v2_20260616/<E0_prod|W7_or_regime_asym>/<segment>/trading_map_tpc_event.html`

**人工核对**：多单是否在 bid 墙附近回踩、空单是否在 ask 墙附近；有无追价穿墙。

## Promote?

- [x] Σ R vs E0 过关（三阶段）— W7 +22.5R
- [x] maxDD 不恶化 — worst -6.5% vs -9.0%
- [x] 笔数未塌缩 — 236 vs 206
- [ ] trading map 语义对齐 — **待 map 人审**

**倾向**：W7 候选 promote 至 `config/strategies/tpc/archetypes/entry_filters.yaml`（OR tier 两条 wall 腿）；W5/W6 不 promote（冗余或弱于 W7）。**等 map 确认后再 lock。**

## 后续

1. Map 人审通过后 promote W7 wall tiers（`locked: true`）
2. Track B：修 `ema_1200_position` prepare-only pipeline → Phase 1d 复扫
3. 可选：6 coin 扩展 grid
