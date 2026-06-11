# BPC 入场语义探索 — 突破 / 回撤 / 延续 vs 反追高

> **用途**：记录 Phase 1 扫描 + 设计讨论结论，供 Phase 3 grid 与 Phase 4 trading map 对照验收。  
> 扫描数据见 [`PHASE1_REPORT.md`](PHASE1_REPORT.md)；定参见 [`DECISION.md`](DECISION.md)；实现见 [`config/experiments/20260611_bpc_lookback_retest_validate/variants/bpc_lb120_retest_strategies/bpc/archetypes/prefilter.yaml`](../../../config/experiments/20260611_bpc_lookback_retest_validate/variants/bpc_lb120_retest_strategies/bpc/archetypes/prefilter.yaml)。

## 1. 问题背景

Trading map 复盘：prod BPC 常在 **突破延续段 / 箱体上沿追高** 入场，而非「压缩区突破 → 回踩 → 再延续」。

典型根因（与 [`docs/strategy/B系统入场语义与执行层周期错配_CN.md`](../../docs/strategy/B系统入场语义与执行层周期错配_CN.md) §2.2 一致）：

| 因素 | prod 现状 | 后果 |
|------|-----------|------|
| `lookback_breakout=20` | soft_phase 只看 ~1.7 天 | 日内上冲易被记成「突破」 |
| `depth<=0.55` 无上界配套 | 浅 depth + 高位 box | 贴顶延续段仍过 prefilter |
| 无 box 位置约束 | 无反追高规则 | 在箱体上沿买第二腿 |

本实验目标：**拉长 lookback** + **用 label 支持的 box/depth 规则** 把入场从「尖顶追高」挪到「突破后的回踩延续区」，同时 **不丢掉** BPC 三锚点语义。

---

## 2. BPC 三层语义（应拆到不同特征 / 时间尺度）

BPC 不是「一根 K 上同时满足所有条件」，而是 **时间上的三拍**：

```text
压缩区 → [突破发生在 lookback 内] → 价格回落到箱内中段 → [recovery] → 入场
                ↑ soft_phase 锚点              ↑ depth 带          ↑ 非 box 顶
```

| 语义层 | 要问什么 | 规则 / 特征 | 时间尺度 |
|--------|----------|-------------|----------|
| **Breakout（曾发生过）** | 近 L 根内是否真突破过？ | `bpc_recent_breakout_strength>=0.40` + `bpc_volume_compression_pct` | rolling / soft_phase（**非**当根 `box_breakout`） |
| **Pullback（正在回踩）** | 从突破位回落但未破结构？ | `bpc_pullback_depth` 带：`>=0.12` 且 `<=0.55` | soft_phase 当前 depth |
| **Continuation（准备再走）** | 回踩后有恢复力度？ | `bpc_recovery_strength>=0.50` | soft_phase |
| **Anti-chase（别在箱顶追）** | 入场时是否仍在箱体高位？ | `box_pos_120<=0.85` | 当前 bar 在 box@120 内的位置 |

**关键区分**：

- **`box_pos` 高** = 价格仍在箱体上沿 / 延续末段 → **要挡**（反追高）
- **`recent_breakout_strength` 高** = 前段窗口内发生过突破 → **要留**（突破语义）

一个是 **当前位置**，一个是 **历史事件**；二者不矛盾。

### 位置直觉图

```text
box 顶 ─────────────── box_pos ≈ 1.0   ← label 差（追高 / 延续末段）
         ╲
          ╲ 突破后拉升
           ╲
box 中腰 ───────────── box_pos ≈ 0.4–0.7  ← 理想：突破已过 + 已回踩 + 未贴顶
           ╱
          ╱ 回踩
box 底 ─────────────── box_pos ≈ 0.0
```

---

## 3. 为什么不用 `box_breakout>=0.5` 当硬过滤

Phase 1 **否定** 的不是「要有突破语义」，而是 **在同一根 bar 上用 `box_breakout_up>=0.5` 当入场扳机**。

| 证据 | 结论 |
|------|------|
| `retest_band`：`depth` 带 + `box_breakout_up>=0.5` | **n=0**（L20 parquet 上组合几乎不触发） |
| `breakout_up/down` 单特征 plateau | 负 Δpp — label 不支持作硬过滤 |
| 与 `depth>=0.12` 组合 | 和 retest 变体原设想冲突：真回踩时当根往往 **已无** breakout 脉冲 |

真实节奏多为：

1. **T−k**：突破（`recent_breakout_strength` 记入 rolling max）
2. **T−1…T**：回踩（depth 升、`box_pos` 从高位回落）
3. **T**：入场 — 此时不一定还有 `box_breakout==1`

因此：

- **突破语义** → `bpc_recent_breakout_strength`（rolling 窗内 max）
- **回撤语义** → `bpc_pullback_depth` 带
- **非追高** → `box_pos_120<=0.85` + `recovery_strength`

---

## 4. Phase 1 label scan 摘要（`chop<=0.40`, n≈3006）

扫描编排：[`rd_loop_bpc_box_pullback_phase1.yaml`](rd_loop_bpc_box_pullback_phase1.yaml)  
产物：`results/rd_loop/bpc_box_pullback_20260611/quick_scan/`

| 扫描 | 结论 | Phase 2 动作 |
|------|------|--------------|
| `depth_floor` depth≥0.12 | \|z\|=5.15, succ +4pp | **保留** depth 下界 0.12 |
| `depth_plateau` depth≤0.35 | \|z\|=3.81, succ 更差 | **不**收紧上界到 0.35；保留 prod 0.55 |
| `anti_chase` box_pos_120≥0.85 | Δpp **−10.3pp**, \|z\|=7.36 | **新增** `box_pos_120<=0.85` |
| `box_scale` box_pos_120≥0.75 | Δpp −7.6, \|z\|=6.4 | 支持反高位 box |
| `retest_band` depth + box_breakout | n=0 | **放弃** box_breakout 硬门槛 |
| lookback L120/L240 bull 子集 | EMA 列全 0，plateau 无效 | lookback 对比靠 **Phase 3 因果 grid** |

---

## 5. Phase 2 落参 — `B_L120_retest` prefilter

当前 retest 树 = **prod 三锚点** + **扫描支持的 depth/box 约束**：

```yaml
# config/experiments/20260611_bpc_lookback_retest_validate/variants/bpc_lb120_retest_strategies/bpc/archetypes/prefilter.yaml（摘要）
bpc_volume_compression_pct >= 0.9295
bpc_recent_breakout_strength >= 0.40    # 锚点 1：Breakout 成立
bpc_pullback_depth <= 0.55              # 锚点 2：Pullback 受控
bpc_recovery_strength >= 0.50           # 锚点 3：Continuation
bpc_pullback_depth >= 0.12              # Phase1：必须有回踩，非贴顶
box_pos_120 <= 0.85                     # Phase1：反追高（替代 box_breakout）
```

**一句话**：突破用 rolling 语义记住；入场点在回踩区且不在箱体上沿 — 不是删掉突破层。

### 与 prod 的差异

| 项 | B0_prod | B_L120 | B_L120_retest |
|----|---------|--------|---------------|
| lookback_breakout | 20 | 120 | 120 |
| prod 三锚点 | ✓ | ✓ | ✓ |
| depth 下界 0.12 | ✗ | ✗ | ✓ |
| box_pos≤0.85 | ✗ | ✗ | ✓ |

---

## 6. Phase 3 / 4 回测验收 — 对照本语义看什么

Grid：[`bpc_lookback_retest_grid.yaml`](bpc_lookback_retest_grid.yaml) → `results/bpc/experiments/lookback_retest_20260611/`  
Trading map：`bash config/experiments/20260611_bpc_lookback_retest_validate/run_trading_maps.sh`

### 6.1 数字指标（DECISION 表）

| 变体 | 期望相对 B0_prod |
|------|------------------|
| **B_L120** | bull 段 R ↑ 或 maxDD ↓；笔数不过度塌缩 |
| **B_L120_retest** | 在 B_L120 基础上，bull **追高型亏损笔减少**；sum R 不劣于 B_L120 |
| **B_L240** | 与 L120 比 trade-off；验证更长窗是否过稀 |

早期 baseline（仅 bear_2022，grid 未跑完）：`B0_prod` total_r≈+8.86，20 trades。

### 6.2 Trading map 语义验收（重点 bull_2023_2024）

对照 **B0_prod vs B_L120 vs B_L120_retest**，逐笔看入场 K 线：

| 验收项 | 好（符合语义） | 差（仍像 prod 问题） |
|--------|----------------|----------------------|
| 箱体位置 | 入场在 box 中带或中上沿以下，非尖顶 | 紧贴 box 上沿、突破阳线顶端追入 |
| 突破时间 | 入场前数根～数十根内可见压缩→突破 | 仅当根脉冲，无前期压缩 |
| depth | 入场时 depth 在 0.12–0.55，有可见回踩 | depth≈0 贴顶延续 |
| recovery | 回踩后出现 recovery 信号再入场 | 无回踩直接延续腿 |

**Promote 前提**（见 [`LAYER_PROMOTION_CRITERIA.md`](../LAYER_PROMOTION_CRITERIA.md)）：Total R 改善、maxDD 不恶化、trading map 可解释 — 三项齐才改 prod locked 规则。

---

## 7. 已知缺口与后续方向（未进当前 grid）

| 缺口 | 说明 | 后续 |
|------|------|------|
| **完整「回测箱顶」序列语义** | 当前 retest 只有 `box_pos` 上界，无「曾高→现回落」序列 | condition-set 或新特征；须单独 Phase 1 |
| **box_pos 中段带** | 扫描支持反高位，未扫最优下界（如 0.25–0.75） | 若 retest 笔数过少，补扫 `box_pos` 双侧带 |
| **box@120 与 lookback@240 对齐** | `box_breakout_*` 硬编码 box_hi_120，与 L240 soft_phase 窗宽不一致 | 改 binding 或统一 box 窗宽后再扫 |
| **scan parquet EMA 全 0** | L120/L240 bull 子集 plateau 无效 | 修 `features_scan_phase1.yaml` 后重扫；lookback 仍以 grid 为准 |

### 候选下一档（仅讨论，未 promote）

- `box_pos_120` **双侧带**（如 `>=0.25 AND <=0.75`）：更贴「回踩到箱内中段」
- 突破后 N 根内 `box_pos` 曾 ≥0.85、**当前** `box_pos` 回落 — 形态序列版 retest

---

## 8. 文档索引

| 文件 | 内容 |
|------|------|
| 本文 | 语义设计 + 回测验收清单 |
| [`PHASE1_REPORT.md`](PHASE1_REPORT.md) | 扫描数字与 Phase 2 定参表 |
| [`DECISION.md`](DECISION.md) | grid 结果表 + promote 判决（跑完后填） |
| [`README.md`](README.md) | 实验卡片与命令 |
