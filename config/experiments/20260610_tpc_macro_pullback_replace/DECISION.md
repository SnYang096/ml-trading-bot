# TPC macro_pullback replace — DECISION（待填）

> **方法学**：τ 来自 Phase 1 `scan_tpc_pullback_lookback.py`（见 [`README.md`](README.md)）。流程符合 [`LAYER_PROMOTION_CRITERIA.md`](../LAYER_PROMOTION_CRITERIA.md) §标准 R&D 阶段。

## Phase 2 定参（2026-06-04）

见 [`PHASE1_REPORT.md`](PHASE1_REPORT.md)。

- **M_replace_L15_S12**：label bull macro≥0.15 \|z\|=3.5 + OHLC τ 支持 → **主候选**
- **M_replace_L20_S15**：long≥0.20 label \|z\|<2 → 仅作对照
- Grid 全量重跑（作废旧 partial）

## 假设

macro `tpc_macro_pullback_pct` prefilter **替代** prod `depth<=0.85` 后，在 canonical 三阶段上相对 E0_prod：

1. **bull_2023_2024** trading map 入场落在大回调区（非小震荡）
2. Total R 提升或 trade-off 可解释（R vs maxDD vs 笔数）

## 结果（2026-06-09 grid 全量）

| variant | bear_2022 R | bull R | recent R | sum R | worst maxDD | trades |
|---------|-------------|--------|----------|-------|-------------|--------|
| E0_prod | +4.47 | +18.82 | +13.44 | **+36.73** | −12.1% | 208 |
| M_replace_L15_S12 | −0.50 | −6.26 | +1.82 | −4.94 | −12.0% | 63 |
| M_replace_L20_S15 | −0.91 | −6.35 | +2.04 | −5.22 | −6.5% | 32 |

**读数**：macro 替代 depth 两变体 **sum R 均为负**，显著劣于 E0_prod；笔数大幅收缩（63/32 vs 208）。L15 bull 段尤其差（−6.26R）。

## Promote

**不 promote** `M_replace_*` → prod。macro prefilter 替代 depth 在本 grid 未过 Total R 杠。

## Follow-up：`M_add_L15_S12`（macro AND depth）

- 变体树：`config/experiments/20260610_tpc_macro_pullback_replace/variants/tpc_macro_add_L15_S12_strategies/`
- 假设：大回撤背景（macro≥τ）+ 当根深回踩（depth≤0.85）交集

| variant | bear R | bull R | recent R | sum R | worst maxDD | trades |
|---------|--------|--------|----------|-------|-------------|--------|
| E0_prod | +4.47 | +18.82 | +13.44 | +36.73 | −12.1% | 208 |
| M_add_L15_S12 | −0.50 | −6.41 | +1.82 | **−5.09** | −13.0% | 25 |

**不 promote**。macro AND depth 同样失败——bull 段 prefilter_deny 10×（12647 vs 2352），信号塌缩到 25 笔，且 72% SL。

## Follow-up：`M_wide_exec`（macro + vol_contraction + 宽止损，去 depth + 去 entry filter）

| variant | bear R | bull R | recent R | sum R | τ vol_contraction |
|---------|--------|--------|----------|-------|--------------------|
| E0_prod | +2.91 | +17.79 | +13.44 | +34.14 | — |
| M_wide (0.40) | +1.57 | +1.82 | −0.36 | +3.03 | 0.40 |
| M_wide (0.30) | −3.16 | +1.82 | −0.36 | **−1.70** | 0.30 (Phase1 scan |z|=3.84) |

**不 promote**。宽止损 + 去掉 depth/entry + macro 背景 + vol_contraction 吸收确认的组合在 bull 段勉强正但 recent 段无信号（4 笔）。

---

## 最终结论（2026-06-09）

**TPC macro swing 全线不成立。保留原版 20-bar TPC，不变名不下线。**

### 根本原因

原版 TPC 的底层逻辑是：

> 趋势中每次小回踩都试一下，错了小亏（4ATR 止损），对了拿住（ema1200 出场）。通过高频试错 + 大数定律累积盈利。

**这不是 bug——这是故意分离**：regime 负责「能不能做」（EMA1200 方向），prefilter+entry 负责「现在做不做」（20-bar 回踩位置），执行层桥接两个尺度（紧止损对应微入场，ema1200 对应宏持仓）。

你要的 macro swing（大回撤 + 吸收完成 + 精选入场）是另一个策略，它面临三个根本困难：

1. **「曾跌过」≠「现在是时机」**：`tpc_macro_pullback_pct` 是静态快照，不告诉你回撤是否已停止、吸收是否完成。Phase 1 扫描发现仅 `vol_contraction` 有统计效果（|z|=3.84），`range_convergence` 在 macro 回撤子集中全为零——当前特征集无法可靠识别「吸收完成」时刻。

2. **信号稀疏 × 单笔不确定性高**：macro 筛选后 bull 段仅 25-29 笔，recent 段 4 笔。少样本下统计不稳健，错过一笔好交易的成本远高于避免一笔坏交易。

3. **原版用大数定律碾压**：牛市 112 笔交易里只要 40 笔中等盈利就够 sum R +17.79。macro 策略永远不可能达到这个频率——这是结构性劣势，不是调参能解决的。

### 能否通过更好的特征工程突破？

可能，但路径很长。需要的是**动态过程度量**而非静态快照：
- 回撤尾部形态（higher lows forming、price structure）
- 回撤段内部 momentum divergence（价格新低但动量不跟）
- Volume profile 位置（当前价相对历史成交密集区）

这些特征的 R&D 周期以周/月计，且成功概率不确定。当前不建议投入——原版 TPC 已经高效捕获了趋势回踩这个 edge。

### 正确的定位

```
原版 TPC (depth@20):  「趋势中每次小回踩都试」→ 高频微入场 → sum R +36.73 ✅
新版 macro swing:    「趋势中只在大回撤后入场」→ 低频精选 → sum R < +4 ❌
```

这是两个不同策略，不是新旧版本关系。原版不改名、不下线、不替换。Macro swing 如将来特征成熟可独立上线。参见 [`B系统入场语义与执行层周期错配_CN.md`](../../docs/strategy/B系统入场语义与执行层周期错配_CN.md)。
