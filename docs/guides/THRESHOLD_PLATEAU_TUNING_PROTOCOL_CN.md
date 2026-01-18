## 阈值“平坦高原”调参协议（Router 阈值）

目标：把调参从“找尖峰”变成“找高原”——在多窗口、bootstrap、局部扰动下仍然稳。  
本协议覆盖 Rule Router 3-action 阈值，并把 **启发式分布约束** 与 **TREND 频率约束** 纳入默认流程。
架构原则参考：`docs/ARCHITECTURE.md`（Sharpe 仅在 Portfolio/PCM 层观察）。

---

### 0) 目标 / 非目标（避免误解）

**目标（阈值合理）**
- 阈值落在“真实分布可达区间”（避免离谱阈值）
- Router 分布不塌缩（trade_rate / trend_rate / mean_rate / no_trade_rate 不病态）
- 行为稳定、低抖动（切换率不过高）

**非目标（收益优化）**
- 不优化 Sharpe / PnL / 回撤（这些属于 Execution / Portfolio 层）
- 不追求“最优分布”，只保证“不病态”

---

### 1) 输入与产物

**输入**
- `preds_*.parquet`：NN 多头推理输出（含 `pred_dir_prob/pred_mfe_atr/pred_mae_atr/pred_t_to_mfe`）
- `logs_3action.parquet`：统一 logs（含 `ret_mean/ret_trend`）
- `model.pt`：用于判断 `preds_in_log1p`
- `router_thresholds_baseline.json`：7 个 Router 阈值

**产物**
- `candidates.csv`：候选阈值与窗口评分
- `summary.json`：最佳阈值与评分摘要
- `router_thresholds_best.json`：最终阈值
- `report.md / report.html`：可读报告

---

### 2) 推荐命令（默认流程）

```bash
mlbot diagnose threshold-plateau --no-docker \
  --preds results/nnmh_e2e/tier01/preds \
  --logs  results/nnmh_e2e/tier01/logs_3action.parquet \
  --model <PATH_TO_MODEL_PT_FROM_TRAIN> \
  --baseline-json results/nnmh_e2e/tier01/router_thresholds_baseline.json \
  --out results/plateau/router3action_tier01_oos_v1 \
  --heuristic-bounds --heuristic-qmin 0.05 --heuristic-qmax 0.95 \
  --trend-rate-min 0.005 --trend-rate-penalty 2.0
```

---

### 3) Router KPI（不含 Sharpe）

Router 的职责是 **“分布合理 + 稳定 + 低抖动”**，不负责收益。

**必须指标**

**可控开关（CLI）**
- `--mean-rate-min/--mean-rate-max`
- `--no-trade-rate-min/--no-trade-rate-max`
- `--disable-dist-rate-constraints`（关闭 mean/no_trade 区间约束）
- `trade_rate / trend_rate / mean_rate / no_trade_rate`
  - 目标：防止分布塌缩（用区间 + 软惩罚，不用固定 target）
  - 默认区间：
    - `trend_rate ∈ [10%, 60%]`
    - `mean_rate ∈ [5%, 40%]`
    - `no_trade_rate ∈ [10%, 70%]`
- `switch_rate`（动作切换率）
  - 目标：抑制抖动（建议区分 raw / effective）
- `stability`（多窗口 p25 / std）
  - 目标：分布在时间上稳定

**必须补充的 2 个约束型指标**
- `conditional_correctness`（弱监督）
  - 例：`P(future_MFE > mfe_threshold | action = TREND)`
  - 目的：Router 的“趋势判断”要物理一致
  - 现实现阶段：缺少真实 MFE 时，可用 `ret_trend > 0` 作为弱代理
- `action_entropy`（熵下限）
  - 目的：避免 Router collapse 到单一动作

**评分建议（v0）**
```
window_score =
  - trade_rate_penalty
  - trend_rate_penalty
  - mean_rate_penalty
  - no_trade_rate_penalty
  - switch_rate_penalty
  - correctness_penalty
  - entropy_penalty

robust_score = mean(window_score) - std(window_score)
```

---

### 4) 启发式分布约束（防止阈值离谱）

**目的**：让阈值跟实际市场分布对齐，而不是被极端样本拉偏。  
**做法**：

1) 用 `compute_mode_3action`（默认阈值）在 `preds` 上计算派生字段：  
   `mfe_atr / eff / t_to_mfe / dir_conf`
2) 全符号汇总这些字段，取 `qmin~qmax` 分位数作为合理区间  
3) 对候选阈值做硬钳制：

```
threshold = clamp(raw_threshold, quantile(qmin), quantile(qmax))
```

映射关系：
- `dir_conf_trend_min` ← `dir_conf` 分位数区间
- `mfe_trend_min` / `mfe_min` ← `mfe_atr` 分位数区间
- `ttm_trend_min` / `ttm_mean_max` ← `t_to_mfe` 分位数区间
- `eff_min` / `eff_mean_min` ← `eff` 分位数区间

> 这一步只收缩搜索空间，不改变评分逻辑。

---

### 5) TREND 频率约束（防止 TREND 被阈值杀光）

**目的**：避免 `TREND` 长期为 0，确保输出分布合理。  
**做法**：在 window_score 上加入趋势频率惩罚项。

当设置 `--trend-rate-min` 时：
```
trend_rate_penalty = max(0, trend_rate_min - trend_rate) * trend_rate_penalty
```

当设置 `--trend-rate-target` 时：
```
trend_rate_penalty = max(0, |trend_rate - target| - tol) * trend_rate_penalty
```

---

### 6) 解读关键指标（建议关注）

- `trade_rate / trend_rate / mean_rate / no_trade_rate`：分布是否塌缩
- `switch_rate`：是否抖动（越低越稳）
- `conditional_correctness`：TREND 是否有“展开证据”
- `action_entropy`：是否 collapse 到单一 action
