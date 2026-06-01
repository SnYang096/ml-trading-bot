# TPC regime + gate 扩展（20260601）

**Parquet**：`train_final_20260523_122438_rr_extreme` · **Label**：`success_no_rr_extreme`  
**Scan**：`results/rd_loop/tpc_regime_gate_extend/quick_scan/report.html`

---

## 1. Regime — `ema_1200_position` 好不好？要加 slope 吗？

**有过分析**（20260526 F' + event_backtest），本次在 **同 parquet + chop≤0.4** 上复验。

| 条件 | n | succ_in | Δpp vs base | \|z\| |
|------|---:|---:|---:|---:|
| **prod_H** `\|ema\|>0.10` | 8556 | 56.53% | **+0.06pp** | **0.14** |
| H_loose `\|ema\|>0.08` | 9545 | 56.10% | −0.37pp | 0.96 |
| H_tight `\|ema\|>0.12` | 7592 | 56.59% | +0.11pp | 0.24 |
| **F'** `\|ema\|>0.10` ∧ `\|slope\|>0.002` | 6937 | 57.49% | **+1.01pp** | **2.03** |
| slope_only `\|slope\|>0.002` | 7045 | 57.32% | +0.84pp | 1.71 |

**IC（forward_rr，chop≤0.4）**：`ema_1200_position` h=1 **IC≈−0.033**；`ema_1200_slope_10` 弱负/近零。

**决策（scan 层）**

- **维持 prod_H（\|ema\|≥0.10）**；在 chop≤0.4 子集上 H 相对 base **几乎无 Δpp**（规则主要挡「死区 bar」，不是抬 success）。
- **F' label 更好，但不 promote**（与 20260526 一致）：需 event_backtest 双段；历史 backtest F' **totR/DD 弱于 H**。
- **trend_r2 / bb_width 替代 regime**：\|z\|均 &lt;2 → **不换主 macro**。
- **bull_only vs bear_only**：空头侧 success 更高是子样本现象，**不能**改成只做 bear；保持双向 regime。

**Phase 2 可选**：`Fp_ema_plus_slope_strategies` vs prod（若要做，单独 regime grid，勿与 gate 混跑）。

---

## 2. `vol_leverage` → 单边

| 规则 | succ_in | succ_out | Δpp | \|z\| |
|------|---------|----------|-----|--------|
| prod 中间带 deny | 52.9% | 55.3% | −1.5pp | 1.46 |
| **low_deny &lt;0.03** bull | 47.3% | 55.7% | **−7.1pp** | **3.79** |
| low_deny &lt;0.05 | 53.0% | 54.9% | −1.4pp | 1.06 |
| high_deny &gt;0.15 | 57.5% | 52.7% | **+3.1pp** | 2.96 |

→ **单调：高 vla = 更好 success**；应 **deny 低尾**，**绝不能 deny 高值**。  
→ **τ=0.03** label 最强但过杀（600/4046 bull）；**τ=0.05** 证据弱。  
→ 实验树 **`tpc_gate_G6_vol_lev_low_deny_strategies`**（`vla<0.05`）待 Phase 2；promote 前可增 **G6b `vla<0.03`** 对照。

---

## 3. `evt_var_99` — 单边重设计

| 条件 | Δpp | \|z\| |
|------|-----|------|
| prod 窄带 deny (0.67,0.80) | **−2.95pp** | 3.03 |
| **evt &gt;0.75**（高尾） | hit 侧 **+1.32pp** | 2.67 |
| **evt &gt;0.80** plateau | succ_hit **59.0%** vs other 55.6% | **4.62** |
| **evt &lt;0.55**（低尾 deny 候选） | **−0.35pp** | 1.13 |

→ **单调：高 EVT = 更好 success** → 单边只能试 **deny 低尾** `evt<0.55`，**绝不能** deny 高尾。  
→ 低尾 label 证据 **弱**（\|z\|≈1.1）；旧中间带虽 -2.95pp 但 **lift 全负** + 与单调矛盾。  
→ **实验树 G7**：`evt_var_99 < 0.55` deny、全 regime、**enabled**（Phase 2 回测再判）。  
→ **勿**恢复 prod 0.67–0.80 中带。

## 3b. `vol_persistence` 为何写「仅牛市」？旧结论可信吗？

**历史原因（20260526 variant H）**：6 币、**旧 entry** 的 event_backtest — 全关 vol（B）recent **+60R** 但 2024 bull **maxDD −13.5%**；只在 `ema>0.10` 开 vol（H）换 **DD −7.6%** 换少 **~13R**。这是 **R/DD 折中**，不是 label 证明「bear 不需要」。

**本次 label 复验**（`chop≤0.4`，`success_no_rr_extreme`）：

| 条件 | n | succ_in | Δpp | \|z\| |
|------|---:|---:|---:|---:|
| 现网 bull 带 | 863 | 47.9% | **−8.6pp** | **5.21** |
| **同带 + bear** `ema<-0.10` | 824 | 49.4% | **−7.1pp** | **4.18** |
| **同带 无 ema** | 4555 | 55.2% | −1.2pp | 1.88 |

→ **Bear 上中间带同样挡的是更差 bar**；bull-only 是 **回测折中**，在现 label 口径下 **不能**说「只有牛市需要」。  
→ **E2_or + BTC/ETH** 下旧 ABH **不可直接当 promote 依据**；须重跑 Phase 2。  
→ **实验树 G9**：去掉 `ema>0.10`，对照 H 是否仍 Pareto。

---

## 4. `path_efficiency` — 去掉 `>0.15 deny` 更好吗？

**未进 prod**（0530 E3 已否）。bull 上：

- `deny pe>0.15`：Δpp **+0.04pp**，\|z\|**=0.38**（命中侧略好，不是更差）
- `low pe<0.15`：仅 n=84，\|z\|无意义
- plateau `pe>` 全格 \|z\|&lt;2

→ **不要加 PE gate**；「只有 0.15 有用」是误判 — 0.15 只是「几乎全员命中」的阈值，不是稳定 plateau。  
→ **去掉/不加 = 正确**。

---

## 5. 其它 gate 候选（bull）

| 特征 | scan | 建议 |
|------|------|------|
| `bb_width` / `atr_percentile` / `trend_r2_20` | \|z\|&lt;2 | 暂不 promote |
| `vol_clustering_strength` | 待读 plateau | 次轮 |
| `vol_persistence` | 0531 已评 | Phase 2 G1 优先 |

---

## Phase 2（已完成）

BTC+ETH 2023–2025 · `results/tpc/experiments/regime_gate_extend/`

| Variant | trades | totR | CAGR | maxDD |
|---------|--------|------|------|-------|
| **G0** prod | 44 | **+6.71R** | 3.17% | **−6.32%** |
| **G1** 关 bull vol×2 | 48 | **+7.88R** | **3.78%** | −6.52% |
| G6 vol_lev 低尾 &lt;0.05 | 41 | +1.28R | 0.27% | −10.41% |
| G7 EVT 低尾 &lt;0.55 | 30 | +1.64R | 0.67% | −4.04% |
| G9 vol_persist 全 regime | 40 | +4.83R | 2.16% | −8.15% |

→ **G1 唯一全面优于 G0 的改法**（+1.17R；DD +0.2pp 可接受）。  
→ **G6 / G7 证伪**：单边低尾改形状或开 EVT **大幅伤 R**（勿 promote）。  
→ **G9 证伪「改全 regime vol」**：差于 G1，**维持 bull-only 关断或整段 disabled** 即可，不必扩到 bear。

## Promote prod（与 0530/0531 合并）

1. **`gate.yaml`**：`vol_persistence` + `vol_leverage` bull 规则 **`disabled: true`**（chop 保留；EVT 保持 disabled）。  
2. **不做**：vol_lev 低尾（G6）、EVT 低尾（G7）、vol_persist 全 regime（G9）、PE gate。  
3. **regime**：维持 `\|ema\|≥0.10`，不加 slope（F' 仍不 promote）。
