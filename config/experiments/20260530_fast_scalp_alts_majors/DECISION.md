# fast_scalp alt/majors split — deployment decision

**实验目录（canonical）：** `config/experiments/20260530_fast_scalp_alts_majors/`  
**前置实验：** `[20260529_fast_scalp/](../20260529_fast_scalp/)` Phase 1 IC + pooled train  
**Holdout：** 2025-10-01 → 2026-04-01  
**结果根：** `results/rd_loop/fast_scalp_ic_plateau/`

---

## Go / No-Go 总表


| Slug                    | 结论              | 含义                                                         |
| ----------------------- | --------------- | ---------------------------------------------------------- |
| `**fast_scalp_alts`**   | **条件 promote**  | Holdout 回测通过，可 **paper / 小仓 shadow**；不是「不能上线」，而是 **带约束上线** |
| `**fast_scalp_majors`** | **条件 promote**  | Dedicated BTC/ETH 重训 holdout 回测优于 pooled；同样建议 **paper 先行** |
| `**fast_scalp`（6 币一体）** | **reject live** | majors 拖后腿；拆成 alts + majors 两 slug                         |


**「条件 promote」≠ reject。** 下列 caveat 是 **上线约束**（τ 冻结、BNB 处理、仓位上限、监控项），不是否决 OOS 正收益。

---

## 1. Alt-only (`fast_scalp_alts`)

### 1.1 两种训练路径对比


| Approach                             | q        | mean Sharpe | mean Return% | Holdout Pearson | Artifact                          |
| ------------------------------------ | -------- | ----------- | ------------ | --------------- | --------------------------------- |
| **Pooled 6-coin model → alt subset** | **0.05** | **1.31**    | **29.6%**    | +0.025          | `train_final/.../141451_ic_top35` |
| Alt-only retrain (4 coins)           | 0.05     | 0.13        | 5.7%         | +0.010          | `train_final/.../203930_alts`     |


4 币单独重训 **CV 更好（+0.028）但 holdout 回测更差** → **不采用 alt-only artifact**；部署仍用 **6 币 pooled 模型**，只在执行层限制 4 alt 符号。

### 1.2 推荐 τ（pooled 模型，alt 子集）

- **q=0.05**（top/bottom 各 5%，很 selective）
- long ≥ **0.3701**，short ≤ **-0.0074**
- 273 trades / 4 coins / ~6 months holdout

Per-symbol @ q=0.05:


| Symbol | Sharpe | Return% | Win%  | Trades |
| ------ | ------ | ------- | ----- | ------ |
| SOL    | 2.16   | +52.1   | 60%   | 65     |
| ADA    | 1.83   | +45.0   | 60%   | 78     |
| XRP    | 1.32   | +26.8   | 62.5% | 64     |
| BNB    | -0.07  | -5.4    | 51.5% | 66     |


### 1.3 为何仍可条件上线？

**通过项（OOS）：**

- Holdout mean Sharpe **1.31**，3/4 币 Sharpe 为正
- 相对 monolithic 6 币（majors 负贡献），alt 子集 **已验证有 edge**

**约束项（上线时必须接受）：**


| 风险                | 说明                   | 建议                                        |
| ----------------- | -------------------- | ----------------------------------------- |
| τ 很 selective     | q=0.05；q≥0.10 收益明显下滑 | **冻结 q=0.05**，不做 live 再优化                 |
| BNB 不稳定           | 唯一负 Sharpe alt       | v1 **降权或暂 exclude BNB**；或单独监控             |
| Pearson 弱         | +0.025，排序模型非校准       | 只看 **分位 τ**，不看绝对 score 阈值漂移               |
| 样本偏少              | ~270 trades / 4 币    | paper 期盯 **rolling Sharpe / trade count** |
| 非 alt 重训 artifact | 推理与 6 币训练耦合          | 重训 pooled 模型时需 **重跑 alt holdout τ**       |


**reject 线（若出现则下线）：** paper 段 rolling Sharpe < 0 持续 4 周，或 BNB+某 alt 连续亏损超预算。

Config: `config/strategies/tree_strategies/fast_scalp_alts/`  
Results: `results/rd_loop/fast_scalp_ic_plateau/alts_holdout_rr_from_6coin/`

---

## 2. Majors BTC/ETH (`fast_scalp_majors`) — 单独重训

### 2.1 训练（`train_final_20260530_203931_majors`）


| 指标                  | 数值                             |
| ------------------- | ------------------------------ |
| Train samples       | 11,338（BTC+ETH pooled）         |
| Holdout samples     | 3,188                          |
| CV metric           | +0.0019（弱正，不稳定）                |
| **Holdout Pearson** | **-0.009**（几乎无线性相关）            |
| Features            | 206（同 fast_scalp top-35 IC 特征） |


Pearson 为负但 τ 回测为正 → 典型 **排序/分位信号**：树在 extremal quantile 有用，整体线性相关差。

### 2.2 τ 扫描 vs pooled 6 币模型（BTC/ETH 子集）


| Approach                     | q        | mean Sharpe | mean Return% | BTC        | ETH        |
| ---------------------------- | -------- | ----------- | ------------ | ---------- | ---------- |
| Pooled 6-coin → BTC/ETH      | 0.12     | 0.62        | 5.7%         | +14%       | -2.7%      |
| **Dedicated majors retrain** | **0.08** | **0.89**    | **11.0%**    | **+10.9%** | **+11.1%** |


推荐 τ：**q=0.08** — long ≥ **0.3536**，short ≤ **0.0672**（125 trades）。

Per-symbol @ q=0.08:


| Symbol | Sharpe | Return% | Win%  | Trades |
| ------ | ------ | ------- | ----- | ------ |
| BTC    | 1.00   | +10.9   | 61.4% | 57     |
| ETH    | 0.78   | +11.1   | 50.0% | 68     |


### 2.3 上线判断

**通过项：** dedicated 重训在 holdout 上 **双币均正收益**，且明显优于 pooled 6 币同段。  
**约束项：** Pearson≈0、CV 弱、τ 敏感（q=0.05 为负）、holdout 仅 ~125 trades → **paper / 小仓先行**，冻结 q=0.08。

Config: `config/strategies/tree_strategies/fast_scalp_majors/`  
Artifact: `results/train_final/fast_scalp_majors/train_final_latest/fast_scalp_majors/`  
Results: `results/rd_loop/fast_scalp_ic_plateau/majors_holdout_rr/`

---

## 3. Combined PCM

- Live：**两 slug 并行** — `fast_scalp_alts`（4 alt，pooled 模型推理）+ `fast_scalp_majors`（BTC/ETH，dedicated 模型）
- `**fast_scalp` 6 币一体不再 live**；保留为 R&D + pooled 训练源（供 alt 推理）
- PCM 新槽位；不与 B/C evidence 合并

---

## 4. Results paths


| 路径                              | 内容                                |
| ------------------------------- | --------------------------------- |
| `alts_holdout_rr_from_6coin/`   | **部署依据** — pooled 模型 + alt τ      |
| `alts_holdout_rr/`              | 4 币重训（弱，参考）                       |
| `majors_holdout_rr/`            | **部署依据** — majors 重训 + τ          |
| `majors_holdout_rr_from_6coin/` | pooled 模型 + BTC/ETH（劣于 dedicated） |
| `segment_matrix/`               | market_segment 四段 τ 验证（见 §6）      |


---

## 5. 与 sr_breakout 树对比（为何 alts 不是同一类 reject）


|                   | sr_breakout 树  | fast_scalp_alts         |
| ----------------- | -------------- | ----------------------- |
| Holdout RR @ 推荐 τ | **负**（≈ -0.4%） | **正**（mean Sharpe 1.31） |
| 决策                | reject promote | **条件 promote**          |


Caveat 多 ≠ 不能上线；**OOS 收益符号 + 可冻结 τ** 才是 promote/reject 分水岭。

---

## 6. market_segment 分段稳定性（2026-06-01）

**配置：** `fast_scalp_segment_tau_grid.yaml`（canonical 实验目录内，无独立脚本）  
**产物：** `results/rd_loop/fast_scalp_ic_plateau/segment_matrix/{alts,majors}/<segment>/`


| Slug       | Segment              | 样本性质                        | q    | mean Sharpe | mean Return% | trades | +Sharpe coins |
| ---------- | -------------------- | --------------------------- | ---- | ----------- | ------------ | ------ | ------------- |
| **alts**   | bear_2022            | pre-train walk-forward      | 0.05 | 0.03        | -9.5         | 620    | 3/4           |
| **alts**   | bull_2023_2024       | pre-train walk-forward      | 0.05 | **1.21**    | **141.3**    | 822    | 4/4           |
| **alts**   | recent_range_to_bear | mixed（含 train + 部分 OOS）     | 0.05 | 0.64        | 40.7         | 738    | 3/4           |
| **alts**   | **recent_6m_oos**    | **OOS holdout（promote 门禁）** | 0.05 | **1.31**    | **29.4**     | 273    | 3/4           |
| majors     | bear_2022            | pre-train walk-forward      | 0.08 | -0.46       | -13.9        | 218    | 1/2           |
| majors     | bull_2023_2024       | pre-train walk-forward      | 0.08 | -0.79       | -51.0        | 642    | 0/2           |
| majors     | recent_range_to_bear | mixed                       | 0.08 | -1.11       | -42.0        | 270    | 0/2           |
| **majors** | **recent_6m_oos**    | **OOS holdout**             | 0.08 | **0.88**    | **10.7**     | 125    | 2/2           |


**方法说明：**

- 模型 train：2024-01-01 → holdout 2025-10-01；`bear_2022` / `bull_2023_2024` 为 **train 之前** walk-forward 诊断，**不作 promote 门禁**。
- 历史段：`artifact_dir` 现推 score + **冻结 deploy q**（段内重算 pred 分位阈值）。
- `recent_6m_oos`：与 Phase 2 相同，用 train artifact 的 **holdout predictions.parquet**（alts 先 filter 4 币）。

**解读：**


| Slug                  | 结论                                                                                                                |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **fast_scalp_alts**   | OOS holdout 与 Phase 2 一致（Sharpe 1.31）；bull 段极强、bear 段 flat — **维持条件 promote**，监控 bear/range 环境                    |
| **fast_scalp_majors** | **仅 recent_6m_oos 为正**（双币 Sharpe +）；train 前 / mixed 段均为负 — **维持条件 promote**，但 regime 敏感，paper 期必须盯 rolling Sharpe |


**跑法：**

```bash
PYTHONPATH=src:scripts:. python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260530_fast_scalp_alts_majors/fast_scalp_segment_tau_grid.yaml
```

