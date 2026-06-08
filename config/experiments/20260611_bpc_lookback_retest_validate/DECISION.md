# BPC lookback + box-retest — DECISION（待填）

> **方法学**：本实验 grid **先于 Phase 1 扫描启动**，属探索性 ablation。Phase 1 用 [`rd_loop_bpc_box_pullback_phase1.yaml`](rd_loop_bpc_box_pullback_phase1.yaml)（**`mlbot research scan`**，勿写 `scan_bpc_*.py`）。promote 须等扫描 + 重跑 grid 后再写。见 [`README.md`](README.md) §R&D 阶段进度。

## Phase 2 定参（2026-06-04）

见 [`PHASE1_REPORT.md`](PHASE1_REPORT.md)。

- **B_L120 / B_L240**：lookback 拉长 → Phase 3 因果验证（label 子集因 EMA 列缺失未比）
- **B_L120_retest**：`box_breakout` 改为 **`box_pos_120<=0.85`** + depth≥0.12 带（反追高 scan \|z\|=7.4）
- Grid 全量重跑 12 runs

语义假设与 trading map 验收项见 [`BPC_SEMANTICS.md`](BPC_SEMANTICS.md) §6。

## 假设

拉长 BPC `lookback_breakout`（120/240）并在 L120 上叠加 **box 反追高 + depth 下界**（非 `box_breakout` 硬门槛），相对 B0_prod：

1. 减少 trading map 上「贴顶追高」入场
2. 更多落在压缩区突破后的回测/延续段
3. canonical 三阶段 trade-off 可接受（R / maxDD / 笔数）

## 结果（2026-06-09 grid 全量）

| variant | bear R | bull R | recent R | sum R | worst maxDD | trades |
|---------|--------|--------|----------|-------|-------------|--------|
| B0_prod | +8.86 | +11.15 | +5.14 | **+25.16** | −6.7% | 106 |
| B_L120 | +5.40 | +4.91 | +7.02 | +17.34 | −7.9% | 106 |
| B_L240 | +7.71 | +6.76 | +3.76 | +18.22 | −8.3% | 111 |
| B_L120_retest | +6.42 | +8.72 | −1.67 | +13.47 | **−4.9%** | 69 |

**读数**：

- **sum R**：B0_prod 仍最佳；拉长 lookback（L120/L240）未提升总 R，bull 段明显变弱。
- **B_L120_retest**：bull maxDD 最好（−4.1% vs −6.7%），笔数少 35%；recent 段负 R（−1.67），sum R 不及 prod。
- 语义假设（少追高）须 **trading map 人审** 验收，见 [`BPC_SEMANTICS.md`](BPC_SEMANTICS.md) §6.2。

## Promote

**暂不 promote**。数字上无变体同时满足 Total R ↑ + maxDD 不恶化：retest maxDD 更好但 sum R 低 12R；L120/L240 未改善 bull。待 trading map（`results/bpc/maps/lookback_retest_20260611/`）看完再决定是否局部采纳 retest 规则或回退 Phase 1 扫 box_pos 中带。
