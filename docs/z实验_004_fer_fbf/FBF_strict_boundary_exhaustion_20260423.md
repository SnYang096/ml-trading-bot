# FBF 边界外 + 衰竭证据 strict variant · 2026-04-23

## 背景问题

用户图检发现 SOL/MSR/FBF rolling_sim 图上：

1. FBF 开仓并未真的在 Boll / OLS / swing 极值「外侧」——多数落在盒子中段。
2. FBF 赢单偏短（平均 `+0.22 R`，生产 16 月 173 笔 totalR=+37.83）。
3. 对「突破/极致区」反而抓不到。

### 诊断证据

脚本 `scripts/diag_fbf_entry_boundaries.py` 对生产 173 笔计算入场时各边界位置：

| 边界判定 | 比例 |
|---|---:|
| `bb_position ≥ 0.95`（Boll 上轨或外） | 11.0% |
| `bb_position ≤ 0.05`（Boll 下轨或外） | 1.7% |
| `bb_position outside [0,1]` | 8.1% |
| `fer_range_pos_20` 极值（≥0.9 或 ≤0.1） | 9.8% |
| **`fer_ols_pos` 极值（≥0.9 或 ≤0.1）** | **49.1%** |
| `wide_sr_dist_atr ≤ 2`（L3 邻近） | 0.0% |

> 结论：FBF 现有 prefilter
> （`fer_sr_failed_breakout_score ≥ 0.38`, `sr_strength_max ≥ 0.52`,
> `fer_ols_width_norm ≥ 0.22`, `trend_r2_20 ≤ 0.5`）**根本没锁「贴边」**，
> 只有 OLS(96) 维度大约 49% 入场天然落在轨边，其他维度（Boll / swing20）
> 多数入场不贴。

## 离线 filter_then_resim 结果

`scripts/filter_then_resim_fbf.py` 对同一批 173 笔基线（生产 `slow-rolling-sim-exp-trail/20260422_202736`）按新 prefilter 过滤后用同一执行 (`trail_r=0.5 / BE 1R / time_stop 48`) 在 2H 上重放：

| 方案 | n | totalR | meanR | win% | Sharpe | maxDD | maxWin |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASELINE | 173 | +37.83 | +0.219 | 48.0 | +1.21 | -16.11 | +9.84 |
| **A. near_ols_only** (`fer_ols_pos ∉ [0.1, 0.9]`) | 85 | **+44.97** | +0.529 | 41.2 | **+1.75** | -7.37 | +9.94 |
| B. near_any_rail (OLS ∨ swing20 ∨ Boll) | 95 | +45.86 | +0.483 | 40.0 | +1.53 | -10.37 | +9.94 |
| C80 B + 衰竭@q80 | 59 | **+46.29** | +0.785 | 45.8 | +1.80 | -6.00 | +9.94 |
| C90 B + 衰竭@q90 | 37 | +39.44 | +1.066 | 56.8 | **+2.61** | -3.00 | +9.94 |
| **D80 A + 衰竭@q80** | **53** | +43.91 | +0.829 | 47.2 | +1.80 | **-5.00** | +9.94 |
| **D90 A + 衰竭@q90** | **31** | +37.06 | **+1.195** | **61.3** | **+2.70** | **-2.00** | +9.94 |

（衰竭证据 = OR of `fer_efficiency_flip_strength`, `fer_aggressor_absorption`,
`fer_momentum_efficiency_decay`，阈值取全样本池分位）

### 关键洞察

- **仅加「fer_ols_pos 贴边」即可**把 173→85，totalR +19%，Sharpe +45%，maxDD 砍半。
- Boll / swing20 贴边**不正交加分**（B 比 A 多 10 笔但 Sharpe 略降）。
- **q80 衰竭证据是真正的 alpha 来源**（meanR 翻倍 0.5 → 0.83；maxDD → -5）。
- **q90 衰竭**虽 Sharpe 最优（2.70）但 trade 数掉到 31（16 月 ~2 笔/月），稀疏性风险高。
- 最大赢单 9.94R 在所有方案都保留 —— 没砍尾部，只砍中段噪声。

## 落地实验

采用离线 **D80** 作为 strict variant 的预期目标：**更严 prefilter + 衰竭 OR entry filter + pure trail**。

### 新增文件

- `config/strategies/fbf_strict/`（从 `fbf_exp_trail` 克隆）
  - `archetypes/prefilter.yaml`：原 4 条 + 新增 `any_of(fer_ols_pos≥0.9 OR ≤0.1)`
  - `archetypes/entry_filters.yaml`：每向 2 filter（flip OR absorption），共 4 filter，OR 组合
  - 其他文件与 `fbf_exp_trail` 一致（execution.yaml = pure trail `trail_r=0.5`）
- `config/prod_train_pipeline_2h_slow_fbf_strict.yaml`（指向 `fbf_strict`，output 到 `results/fbf/slow-rolling-sim-strict`）

### 实验工具（本次新增）

- `scripts/plot_fbf_boundaries.py` — Plotly HTML 可视化（BTC/SOL/ETH 等），close + Boll/OLS/swing HL/wide SR + FBF 入场点
- `scripts/diag_fbf_entry_boundaries.py` — 逐笔 entry 对各边界的量化 CSV + 汇总表
- `scripts/filter_then_resim_fbf.py` — 按新 prefilter 过滤 baseline trades + 同 exec 重放，A/B 总结
- `scripts/simulate_exec_trail.py` — 原脚本扩展 `scale_out`（支持对面 OLS / 20-range 半仓落袋，离线）

### 预期范围

离线 173 笔过滤到 53（D80）/ 31（D90），实盘 rolling_sim 月度出单 **2–4 笔**（样本极稀）。
需关注：
- **月度 KPI plateau 可能失败**（已把 `target_trades_min=4`, `min_trades=8`）；
- **模型训练样本稀疏**（但 turbo_fixed_features 已禁 feature 搜索，直接用 prefilter/entry_filters locked 规则）；
- 对 `fer_efficiency_flip_strength`, `fer_aggressor_absorption` 的 `0.80 / 0.76` 阈值做稳健性扫：后续可在 `rolling_calibration.prefilter.optimize` 产物里看是否被 optimizer 改动（locked 已固化）。

## 下一步

### 回滚

```bash
# 配置可按需回滚（不影响现网 fbf）
rm -rf config/strategies/fbf_strict
rm  config/prod_train_pipeline_2h_slow_fbf_strict.yaml
```

现网 `config/strategies/fbf/` 与 `config/strategies/fbf_exp_trail/` **未改动**，该 variant 与 prod 完全隔离。

### 实跑命令（用户触发）

```bash
mlbot pipeline run --all \
  --config config/prod_train_pipeline_2h_slow_fbf_strict.yaml \
  --stage rolling_sim --skip-shap
```

完成后对比 `results/fbf/slow-rolling-sim-strict/_rolling_sim/<ts>/...` 与
`results/fbf/slow-rolling-sim-exp-trail/_rolling_sim/20260422_202736/...`，关注：

1. 端到端 trade 数（应在 40–90 间；离线 D80 预期 53）
2. totalR、Sharpe、maxDD（应与离线 D80 量级接近）
3. 月度稳定性：如 `stitching.html` 显示有 3+ 个月度 0 trades，需收紧 / 放宽阈值

### 若离线 vs 实跑差距显著（>30%）

可能原因：
- 实盘入场还经过 gate + model prob 阈值校准，过滤率更高；
- 离线假设 baseline 173 笔已经是「真实可见候选池」，但 prefilter 变严可能也放出了
  原先被 gate 砍掉的候选。

此时用 `scripts/plot_fbf_boundaries.py` 对新的 `event_trades_fbf.csv` 再眼检一轮。
