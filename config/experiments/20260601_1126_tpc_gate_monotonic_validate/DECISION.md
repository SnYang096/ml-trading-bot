# TPC gate 单调单边 condition-set（20260602）

**Parquet**：`train_final_20260523_122438_rr_extreme` · **Label**：`success_no_rr_extreme`  
**Scan**：`results/rd_loop/tpc_gate_monotonic/quick_scan/report.html`

**读法**：`succ_in` = 落在 deny 条件内（若作 gate 会被挡）；要比 `succ_out` **更低** 且 |z|≥2，才说明「单调 deny 有 veto 价值」。

---

## 1. `tpc_semantic_chop` — 单调 deny `>τ`（高 chop 更差）

| 条件 | n | succ_in | succ_out | Δpp | \|z\| |
|------|---:|---:|---:|---:|---:|
| `>0.35`（chop≤0.4 子集） | 2506 | 56.70% | 56.45% | +0.23pp | 0.24 |
| `>0.40`（同子集） | **0** | — | — | — | — |

→ 在 **chop≤0.4 诊断子集** 上 **无法评 prod `>0.4`**（与 prefilter 互斥）。  
→ `>0.35` **无证据**。prod **chop>0.4** 仍保留（全样本 gate；与 0531 一致）。

**单调 chop deny**：**未证明**；维持现网 chop gate，不靠本 scan 改阈。

---

## 2. `vol_persistence`（bull，`ema>0.10`）

| 条件 | succ_in vs out | Δpp（in−out） | \|z\| | 单调 deny？ |
|------|----------------|-------------|------|------------|
| **prod 中间带** | 47.9% vs 56.2% | **−8.3pp** | **4.38** | **U 形中间带**（非单调） |
| **mono `<0.003`** | 60.8% vs 53.1% | **+7.7pp** | 3.71 | **反了** — 低 persist 更好，不能 deny 低尾 |
| **mono `<0.01`** | 58.6% vs 53.3% | +5.3pp | 2.76 | **反了** |
| **mono `>0.062`** | 55.0% vs 53.6% | +0.5pp | 0.82 | 高尾略好，不能 deny 高尾 |

→ **不能**改成「单调 deny 低尾或高尾」；**中间带**在 label 上有强 veto，但 backtest **过杀**（G1 关断）。  
→ **结论**：特征有用，**单调单边无用**；prod 应用 **保持 disabled**，不要改形状。

---

## 3. `vol_leverage_asymmetry`（bull）

| 条件 | Δpp（in−out） | \|z\| | 单调 deny？ |
|------|-------------|------|------------|
| **prod 中间带** | −1.5pp | 1.46 | 弱于 persist |
| **` <0.03`** | **−7.1pp** | **3.79** | **低尾 deny 方向正确** |
| **`<0.05`** | −1.4pp | 1.06 | 边缘 |
| **`>0.15`** | **+3.1pp** | 2.96 | **反了** — 高 vla 更好 |

→ **唯一通过 label 的单调写法**：**`vla < τ`（低尾 deny）**，与 0601 plateau 一致。  
→ **但 G6 backtest**（`vla<0.05`）**+1.28R**，远差于 G1 关断 → **label 有信号 ≠ promote**。  
→ prod：**保持 disabled**；若 Phase 2 再试，只试 **极低尾 `vla<0.03`**，不是中间带。

---

## 4. `evt_var_99`（全样本）

| 条件 | Δpp（in−out） | \|z\| | 单调 deny？ |
|------|-------------|------|------------|
| **prod 中间带** | −2.95pp | 3.03 | U 形，非单调 |
| **` <0.55`** | −0.35pp | 1.13 | **弱** |
| **`>0.75`** | **+1.32pp** | 2.67 | **反了** |
| **`>0.80`** | **+2.52pp** | 4.62 | 高 EVT 更好 |

→ **单调 deny 低尾 / 高尾均不能证明**；高尾 deny **方向错误**。  
→ 与 **G7 回测失败**、prod **已删 EVT 规则** 一致。

---

## 总表：四个特征能否用「单调 gate」？

| 特征 | label 上单调 deny 成立？ | backtest 提示 | prod 建议 |
|------|-------------------------|---------------|-----------|
| **chop** | 未评（子集 n=0） | prod chop 保留 | **保留 `>0.4`** |
| **vol_persistence** | **否**（低尾 deny 反号） | 中间带过杀 → G1 关 | **disabled** |
| **vol_leverage** | **低尾 deny 成立** | 低尾仍伤 R → G6 差 | **disabled** |
| **EVT** | **否** | G7 差 | **已删除** |

**不能**证明「四个单调 gate 都有用」——只有 **vol_lev 低尾 deny** 在 label 上方向对，且 **仍不足以 promote**。

---

## Phase 2（可选）

若仍想验证 **vol_lev `vla<0.03` bull-only**（非 G6 的 0.05）：

- 实验树 + `market_segment.yaml` 三分段 grid（勿与 G1 全关混在同一 promote 决策）。

```bash
# 待建 config/experiments/20260602_tpc_gate_monotonic_validate/tpc_gate_monotonic_phase2_grid.yaml
```
