# BPC 各层规则验证（rd_loop · 2026-05-27）

- **数据源**：`results/train_final/bpc/train_final_20260519_123251_rr_extreme/bpc/features_labeled.parquet`（48049 行，`success_no_rr_extreme` 1:26659 / 0:21390）
- **跑法**：`PYTHONPATH=src:scripts python scripts/rd_loop.py --hypothesis-yaml config/experiments/rd_loop_bpc.yaml`
- **报告目录**：`results/rd_loop/bpc/quick_scan/*.md`
- **状态**：scans ✅；`variant_grid` 暂未跑（无 `EXPERIMENT_INDEX.json`，需先建 `config_experiments/bpc_*_strategies` 树）；`decision_doc` step 因此 step 失败。

## TL;DR

| 层 | 当前生产值 | label 验证 | 结论 |
|---|---|---|---|
| Regime `tpc_semantic_chop<=0.40` | 锁定 | Δ=-0.57pp, \|z\|=1.36 | **结构性过滤，对 label 几乎中性** — 合理，但不是 label-improver |
| Regime `\|ema_1200\|>=0.10`（B-bull 子条件） | 未在 BPC regime | **Δ=-2.22pp, \|z\|=3.33** | ⚠️ 强趋势期 BPC label 反而变差，与 TPC 相反 |
| Prefilter `bpc_volume_compression_pct>=0.9295` | locked | 高原在 **0.95**（\|z\|=2.33） | 阈值偏松 0.02；近期 plateau 在 0.95 |
| Prefilter `bpc_recent_breakout_strength>=0.40` | locked | **方向反了**：rows ≥0.50 succ 54.4% vs <0.50 succ 60.9%（\|z\|=3.49） | ⚠️ 需复核 ① label 口径 ② 是否该 invert / 弱化 |
| Prefilter `bpc_pullback_depth<=0.55` | locked | 全部 \|z\|<2，flat | 阈值对 label 中性，**不卡 label**，保留语义意义 |
| Prefilter `bpc_recovery_strength>=0.50` | locked | 全部 \|z\|<2，flat | 同上 |
| Gate `vol_persistence (0.0029,0.0616) deny` | locked | **deny 区 succ_in=57.2% > out=54.4%**（Δ=+2.24, \|z\|=2.33） | ⚠️ label 上看 gate 在拒绝高成功区（与 TPC 同样的 label-vs-R 矛盾） |
| Gate `vol_leverage_asymmetry (0.0558,0.1482) deny` | locked | Δ=-0.03pp, \|z\|=0.03 | label 上**完全没作用**；仍可能保 R-multiple，需 event_backtest 验 |
| Gate `vpin_ma20>0.3044 deny`（disabled） | disabled | Δ=+0.68, \|z\|=1.16 | 维持 disabled |

## 1. Regime 层

### 1.1 chop 上限

```
condition                   n        succ_in   succ_out   Δpp     |z|
regime_baseline (chop<=0.40)  10854   54.91%    55.65%   -0.57   1.36
regime_strict_chop (<=0.30)    8464   54.99%    55.59%   -0.50   1.01
```

`chop<=0.40` 在 label 上**几乎没差异**，但这是预期：chop 是 *结构空间约束*（防横盘里追势），它的价值在 **R-multiple/maxDD**，不在 label success。**保留当前 0.40，不动。**

### 1.2 EMA1200 强趋势条件（BPC ≠ TPC 关键差异）

EMA bucket 表（基 mask = chop<=0.40）：

```
bucket |ema_1200_position|>=0.10  (n=4994, base_succ=53.26%)
  ema_bull_only (>=+0.10)  n=1839  succ_in=47.6%  Δ=-5.68pp  |z|=6.15  ❌
  ema_bear_only (<=-0.10)  n=3155  succ_in=56.6%  Δ=+3.31pp  |z|=6.15  ✅

bucket |ema_1200_position|<0.10  (n=5860, base_succ=56.31%)
  ema_dead_zone (|ema|<0.03)  n=4126  succ_in=57.7%  Δ=+1.39pp  |z|=3.32
```

- **BPC label 成功率：bear 强趋势 > 弱趋势 > bull 强趋势**（与 TPC 完全相反；TPC strong-bull 是 +6.84pp label-improver）。
- 解释假设：BPC = breakout-pullback-continuation；强 bull 末段做 pullback 更容易追高被反扫，**bear 段反弹做 short-pullback 反而结构干净**。

**待 backtest 验证**：是否要对 BPC 加 **bull-asymmetric regime**（与 TPC H 反向，只在 ema>=0.10 时 deny long 而非 short）。

## 2. Prefilter 层（4 个语义锚）

### 2.1 `bpc_volume_compression_pct >= 0.9295`（当前锁定）

```
threshold   n_hit   succ_hit   succ_other   |z|
0.85        3182    54.97%     54.89%       0.07
0.90        2489    55.73%     54.67%       0.93
0.9295      2005    56.71%     54.50%       1.79
0.95        1653    57.53%     54.44%       2.33  ← plateau
0.97        1269    56.27%     54.73%       1.03
```

**Plateau 在 0.95**（\|z\|=2.33）。0.9295 在边缘。建议：

- 选项 A：**0.9295 → 0.95**（label 略改善，更选择性）
- 选项 B：维持 0.9295 但 plateau-stability 监控加 alert 上沿 0.97

### 2.2 `bpc_recent_breakout_strength >= 0.40`（⚠️ 方向反）

```
threshold   n_hit   succ_hit   succ_other   |z|
0.30        10399   54.85%     56.26%       0.59
0.40        10305   54.77%     57.56%       1.28  ← 当前
0.50        10074   54.45%     60.90%       3.49  ❌
0.60         9671   54.03%     62.13%       5.29  ❌
0.70         9266   54.11%     59.57%       4.04  ❌
```

`succ_hit < succ_other` ：被这个规则**保留**的样本反而比**拒绝**的低 6pp（在阈值 0.5+）。

可能解释：

1. **Label 口径偏差**：`success_no_rr_extreme` 是「未触 RR-extreme 止损」，对**没动**的样本天然友好；breakout_strength 高的本来就是动得猛的样本，**容易触止损但也容易大涨**。这种规则的真正价值在 **R-multiple 的偏度**，不在二元成功率。
2. **门控语义颠倒**：若 BPC 实盘真靠这个规则保 R，则 label 反向是 false alarm。
3. **特征本身退化**：若过去 6 月分布变了，需要 SHAP 重审。

**行动**：
- 跑 event_backtest 双段，对照「prefilter 含 / 不含 breakout_strength」两版 totR / win / maxDD。
- 若 totR 也证伪 → 调整或降级该锚为 audit-only。
- 若 totR 仍正 → 锁的是 R 不是 label，把这条记到 `lock_reason`。

### 2.3 `bpc_pullback_depth <= 0.55` & 2.4 `bpc_recovery_strength >= 0.50`

```
pullback_depth ≤ {0.40, 0.50, 0.55, 0.65, 0.75}    全部 |z|<2，succ_hit≈55%
recovery_strength ≥ {0.30, 0.40, 0.50, 0.60, 0.70}  全部 |z|<2，succ_hit≈55%
```

**label 完全 flat**：这两个锚**不是 label 的瓶颈**，但作为 BPC 语义定义（结构语义），仍应保留。**不动。**

## 3. Gate 层

```
base mask = chop<=0.40, n=10854, base_succ=54.91%
condition                  n      succ_in   succ_out   Δpp     |z|
vp_inside_deny             2147   57.15%    54.36%    +2.24   2.33  ⚠️
vp_outside (<=0.0029)      1750   55.26%    54.84%    +0.35   0.32
vp_high (>=0.0616)         6957   54.13%    56.30%    -0.78   2.18
vla_inside_deny            2968   54.89%    54.92%    -0.03   0.03
vpin_ma20_high_deny        4281   55.59%    54.47%    +0.68   1.16
```

- **`vol_persistence` deny 区**：label 看反了（拒掉的反而是 +2.24pp 高成功区）。这是 [`../_smoke/tpc_gate_vol_ABH_experiment_20260526.md`](../_smoke/tpc_gate_vol_ABH_experiment_20260526.md) §3-§4 的同款张力 — **label success ≠ R-multiple**。TPC promote H 的逻辑是 bull 段 DD 保护；BPC 没做过同样实验。
  - **行动**：复用 TPC ABH 范式做 BPC 的 ABH：A=全开 vol gates、B=全关、H=bull-conditional vol。
- **`vol_leverage_asymmetry`**：label 上**完全没差异**。规则可能在卡极端波动姿态而非平均 label。先保留，与 vol_persistence 一起进 ABH 测试。
- **`vpin_ma20>0.3044`**：维持 disabled。

## 4. IC-decay

`forward_rr` 上各 horizon（注：当前 ic-decay 各 horizon 数值相同，疑似 quick_layer_scan 未实际 shift target；本节仅做跨特征比较）：

| 特征 | rank_IC | p | 说明 |
|---|---:|---:|---|
| `ema_1200_position` | -0.049 | 4e-27 | 高 EMA position → 低 forward_rr（与 §1.2 一致） |
| `vol_leverage_asymmetry` | +0.048 | 5e-26 | 显著正向 — gate locked 区也是高分区 |
| `vol_persistence` | -0.048 | 1e-25 | 高 persistence → 低 R（与 gate 方向一致） |
| `bpc_volume_compression_pct` | -0.041 | 3e-19 | ⚠️ 高 compression → 低 R（与 prefilter `>=0.9295` 反） |
| `vpin_ma20` | -0.016 | 5e-04 | 弱负向 |
| `bpc_recent_breakout_strength` | -0.005 | 0.25 | 不显著 |
| `bpc_pullback_depth` | -0.003 | 0.51 | 不显著 |
| `bpc_recovery_strength` | -0.001 | 0.75 | 不显著 |

**关键发现**：
- `bpc_volume_compression_pct` 的 IC 是 **负的**！但 prefilter 锁的是 `>=0.9295`（要求高 compression）。这两条同时为真的唯一解释还是 §2.2 那个：**label/IC 偏向「没动的样本」，R-multiple 偏向「动得对的样本」**。
- `bpc_recent_breakout_strength` / `pullback_depth` / `recovery_strength` 三个语义锚 IC 都不显著 — 这是**规则特征**（0/1 偏布尔），IC 自然小；用 label success 看更合理（§2 的表格）。

**附注**：ic-decay 各 horizon 数值相同提示 `--target forward_rr` 没有按 horizon 重新计算；后续应改 ic-decay 在 target 模式下实际 shift，或改用 `--label success_no_rr_extreme` 等同时刻 label。这是 quick_layer_scan 工具的 TODO，不影响本次结论。

## 5. 行动清单

| 优先级 | 动作 | 工具 / 命令 |
|:---:|---|---|
| **P0** | BPC ABH 风格 vol-gate 实验（与 TPC 对齐）：A=全开 / B=全关 / H=bull-only-deny | 准备 `config_experiments/{bpc_A,bpc_B,bpc_H}_strategies/`，写 `config/experiments/bpc_variant_grid.yaml` 后跑 `event_backtest --variant-grid` 双段 |
| **P0** | `bpc_recent_breakout_strength` label-direction 反向：跑双段回测对照「含 / 不含」该锚 | 同上，加一 variant `bpc_no_breakout_anchor_strategies/` |
| P1 | `bpc_volume_compression_pct`：0.9295 → 0.95 plateau 微调；加 plateau watchdog 上沿 0.97 | 改 prefilter.yaml + watchdog baseline |
| P1 | regime bull-asymmetric：在 `\|ema\|>=0.10` 时只 deny long（与 TPC H 反向） | regime.yaml + 双段 backtest |
| P2 | quick_layer_scan `ic-decay` 修 horizon shift bug | scripts/quick_layer_scan.py |
| P2 | 用新 `mlbot train final --prepare-only` 产出 box_pos_120 在内的 parquet 重跑本 scan | `mlbot train final --no-docker --prepare-only -c config/strategies/bpc` |

## 6. 复现命令

```bash
cd /home/yin/trading/ml_trading_bot
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/rd_loop_bpc.yaml
```

## 7. P0 event_backtest ABH（✅ 完成，~80min）

完整决策：[`bpc_gate_vol_ABH_experiment_20260527.md`](bpc_gate_vol_ABH_experiment_20260527.md)

**Grid**：`config/experiments/bpc_abh_variant_grid.yaml`  
**实验树**：`config_experiments/bpc_{B_vol_off,H_bull_vol,no_breakout}_strategies/`  
**命令**：

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/bpc_abh_variant_grid.yaml --quiet-signal-logs
```

### 7.1  interim — 2024 bull（已完成 4/4）

| 变体 | trades | totR | ret% | 相对 A |
|---|---:|---:|---:|---|
| **A** baseline | 27 | +16.85 | +7.02% | — |
| **B** vol off | 29 | +17.56 | +7.73% | +0.71R |
| **H** bull-vol | 28 | +16.81 | +6.98% | -0.04R |
| **no_breakout** | 28 | +17.47 | +6.75% | +0.62R |

**初读（bull 段）**：

- 与 TPC 不同：**B（关 vol）在 bull 略优于 A/H**，但差额仅 ~0.7R（27–29 笔，统计力弱）。
- **去掉 breakout_strength 锚**：totR +0.62R vs A，支持 rd_loop「label 反向、R 可能仍受益」假设 — 待 recent 段确认。
- **H 未带来 TPC 式 DD 改善**（本段 ret/ totR 与 A 几乎相同）；是否 promote H 要看 **recent 段** 与 maxDD 分解。

### 7.2 2025-04→2026-04 recent（4/4）

| 变体 | trades | totR | ret% |
|---|---:|---:|---:|
| A | 25 | **-1.22** | +2.74% |
| B | 17 | -4.18 | +2.15% |
| H | 14 | -5.59 | +0.36% |
| no_breakout | 25 | **-1.22** | +2.74% |

**决策**：不 promote B/H；保留 breakout 锚（no_breakout 与 A 相同）。
