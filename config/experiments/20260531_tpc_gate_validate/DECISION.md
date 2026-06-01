# TPC gate 验证（20260531）

**Parquet**：`train_final_20260523_122438_rr_extreme` · **Label**：`success_no_rr_extreme`  
**Scan 产物**：`results/rd_loop/tpc_gate_validate/quick_scan/report.html`

固定 **E2_or entry** + **prod prefilter**；本实验 **只验证 gate**。

## Phase 1 结论（label scan）

### G1 `semantic_chop > 0.4` deny

| 方法 | 结果 |
|------|------|
| `feature-plateau`（bull 子集） | 全格 \|z\|<2 → **非瓶颈** |
| `condition-set` deny 带 | **n=0**（bull filter 已含 `chop<=0.40`，与 deny 互斥） |
| `gate-plateau` batch | no_valid_threshold |

→ 在 **bull 诊断子集** 上 chop gate **几乎不生效**；保留主要为全样本 / bear 语义。

### G2 `vol_persistence` bull 宽带 deny

| 方法 | 结果 |
|------|------|
| `condition-set` 现网 deny 带 | hit n=863，succ **47.9%** vs 外 **56.2%**，**Δpp −6.6pp**，**\|z\|=4.38** |
| `gate-plateau` lift（单特征 gt） | 低阈 lift 正，**无稳定 plateau** |
| `gate-plateau` 整规则 batch | no_valid_threshold |

→ **有 veto 价值**（被 deny 的 bar 更不易 success），但宽带 + bull 条件在标注集上 **过杀约 6.6pp good-rate**；与「牛市实盘差」假设 **一致，值得 Phase 2 关断对比**。

### G3 `vol_leverage_asymmetry` bull 宽带 deny

| 方法 | 结果 |
|------|------|
| `condition-set` | Δpp **−1.5pp**，\|z\|**=1.46**（边缘，未达 2） |
| lift 单特征 | **全负 lift** |
| batch | no_valid_threshold |

→ **弱于 vol_persistence**；倾向 **优先动 vol_persist，lev 单关作 Phase 2 可选**。

### G4 `path_efficiency > 0.15` deny（未进 prod）

- 与 0530 `pe_plateau` 一致：bull 子集 **\|z\|<2** → **不 promote PE gate**。

### Bear 对照

- 同一 vol_persist bull deny 带在 **bear filter** 上 **n=0**（ema 不满足 >0.10）→ bull-only 规则 **不会在 bear 段误触**。

## Phase 1 决策（scan 层）

| 规则 | 建议 |
|------|------|
| chop >0.4 | **保留**（全样本）；bull 子集非瓶颈 |
| vol_persistence bull 带 | **怀疑过杀** → Phase 2 **G1 关 bull vol** 优先 |
| vol_leverage bull 带 | **次要**；可 G5 单关复验 |
| PE deny | **仍不 promote** |

## Phase 2（已完成）

BTC+ETH 2023–2025 · `results/tpc/experiments/gate_validate/`

| Variant | trades | totR | CAGR | maxDD |
|---------|--------|------|------|-------|
| **G0** prod gate | 44 | **+6.71R** | 3.17% | **−6.32%** |
| **G1** 关 bull vol×2 | 48 | **+7.88R** | 3.78% | −6.52% |
| **G2** 关 chop | 63 | +8.54R | 3.81% | −6.72% |

→ **G1**：相对 G0 **+1.17R / +4 笔**，maxDD 略差 **0.2pp** — 与 Phase 1「bull vol 可能过杀」一致。  
→ **G2**：R 更高但 **笔数 +43%**、DD 更差；**不**因 smoke 单独关 chop promote。  
→ **倾向**：prod 可试 **关闭 bull `vol_persistence` + `vol_leverage` 中间带**；等 0601 **G6/G7/G9** 后再定是否改形状或 EVT。

## Promote prod gate

**已写入** `config/strategies/tpc/archetypes/gate.yaml` + `live/highcap/.../gate.yaml`（2026-06-01）：两条 bull vol 中间带 `disabled: true`；chop 保留；EVT 仍关。0601 G6/G7/G9 不采纳。
