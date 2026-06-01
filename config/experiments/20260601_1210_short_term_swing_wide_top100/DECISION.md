# short_term_swing wide top-100 on recent_6m_oos — 决策记录

**实验目录（canonical）：** `config/experiments/20260601_1210_short_term_swing_wide_top100/`  
**前置：** 20260601 wide prepare + IC prune（252 pass → top-100）  
**策略配置 slug：** `short_term_swing_wide_top100_test`  
**结果根：** `results/rd_loop/short_term_swing_wide_top100_recent6m/`

---

## 关键时间窗与数据集（本次实验定义）

**训练 (Train)**
- **时间窗**：2024-01-01 → 2026-03-31（内部按 2025-10-01 切分，train 部分 < 2025-10-01）
- **数据集**：`results/train_final/short_term_swing/prepare_wide_all_20260601/short_term_swing/features_labeled.parquet`
  - 6 币 pooled：BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, ADAUSDT
  - 时间框架：120T
  - 特征来源：`features_tree_full_120T_958f665062`（full shared FeatureStore，~882 列，来自 `config/strategies/_shared/features_all.yaml`）
  - 实际进模型列：`config/strategies/tree_strategies/short_term_swing_wide_top100_test/features.yaml`（IC 排序前 100 列 + `atr_f`）
  - 样本量（参考历史同 wide prepare）：约 35,000 训练行

**OOS / 评估 (Holdout)**
- **时间窗**：2025-10-01 → 2026-03-31（严格 `recent_6m_oos`，来自 `config/market_segment.yaml`）
- **数据集**：同上 wide prepare parquet 的 holdout 部分（~6,500–10,000 行，视精确切分）
- **数据上限**：~2026-03-30（本次 OOS 窗完全在数据可用范围内，无越界）
- **目的**：干净的最近 6 个月“当前 regime” OOS，用于最终 promote / reject 决策（与 TPC gate 等实验的 current-regime 评估口径对齐）

---

**注意**：本次实验的 OOS 窗比之前 wide top-20 使用的 2026-04-01 更保守、更干净，严格对齐 `recent_6m_oos` 定义。

---

## 1. 实验动机（简述）

5 月底 fast_scalp 在 curated 小池子 + alt 子集上曾看到条件 promote（mean Sharpe 1.31 @ q=0.05）；
20260601 wide 全池子 top-20 在 6 币 pooled 上全面 reject（负 RR）。
本次用**同一 IC 排序的前 100 列** + **干净的最近 6 个月 OOS**（`recent_6m_oos`）再探一次：
- 放宽列数是否能让树在当前 regime 下找到可交易 edge？
- 是否至少在 alt 子集或极紧 τ 下出现正 Sharpe（类似 5 月现象）？

---

## 2. 运行记录

- 启动命令：
  ```bash
  PYTHONPATH=src:scripts:. python scripts/rd_loop.py \
    --hypothesis-yaml config/experiments/20260601_1210_short_term_swing_wide_top100/rd_loop_short_term_swing_wide_top100.yaml
  ```
  （中间因 predictions 路径含策略 slug 名而报错，已手动修正 yaml 后单独补跑 tau-scan）
- 关键 artifact：
  - Train: `results/train_final/short_term_swing/train_wide_top100_recent6m_20260601/short_term_swing_wide_top100_test/`
  - τ scan: `results/rd_loop/short_term_swing_wide_top100_recent6m/holdout_rr_recent6m/`
- 实际使用 OOS 窗：2025-10-01 → 2026-03-31（严格 `recent_6m_oos`）
- 运行日期：2026-06-01

### 2.1 训练结果（100 features, recent_6m_oos holdout）

- Train samples: 35,004 | Test (OOS): 9,891
- n_features (model): 100
- Avg CV metric: **+0.0168**
- Holdout Pearson: **-0.0680**（比之前 wide top-20 的 -0.055 略差）
- 失败分析警告：模型选中的 trades 失败率反而更高（lift>1 on failure_rr_extreme / no_opportunity），与 top-20 wide 一致

### 2.2 τ / RR 结果（recent_6m_oos, 6 币 pooled）

- 行数：9,891
- 推荐 q=0.05（long≥1.0917, short≤0.2224）
- **全部 quantile Sharpe 均为负**，最佳（最不差）q=0.05：
  - Sharpe: **-1.228**
  - Return: **-28.48%**
  - Trades: 146
- Per-symbol（@ q=0.05）全部负：
  - BTC: -1.92 | ETH: -2.97（最差） | SOL: -0.24（相对最好） | BNB: -0.74 | XRP: -0.56 | ADA: -0.94

（完整表格见 `tau_scan_holdout_rr.md`）

---

## 3. 核心结果与对比

| 维度 | May fast_scalp (curated, ~35-76 cols) | wide top-20 (20260601, 旧窗到04-01) | **wide top-100 (本次, recent_6m_oos)** | 备注 |
|------|---------------------------------------|-------------------------------------|---------------------------------------|------|
| Holdout Pearson | +0.025 ~ +0.034 | -0.055 | **-0.068** | 本次最差 |
| CV metric | 弱正/接近0 | 0.056 | **+0.0168** | |
| 6 币 pooled @ 推荐 τ | 混合（alt 正，majors 拖累） | -1.16 (q=0.05) | **-1.228 (q=0.05)** | 全面负 |
| Return % (pooled) | +10.7% (H_recent alt 子集参考) | -29% | **-28.5%** | |
| 推荐 q | 0.05（alt 子集） | 0.05 | 0.05 | |
| Trades (pooled @ q) | ~270 (alt 4 币) | 154 | 146 | |
| Alt 子集 (SOL/ADA/XRP/BNB) | **条件 promote** (mean Sharpe +1.31) | 未单独报 | 未单独报（整体负） | 5 月唯一亮点 |
| Majors (BTC/ETH) | 需 dedicated 重训后才正 | 严重负贡献 | **ETH -2.97 最差** | |
| 失败率 lift (selected trades) | - | >1（更差） | >1（更差） | 一致问题 |

**本次 top-100 结论**：**reject**（比 top-20 更差或持平，无任何改善迹象）。

**Per-symbol 亮点 / 雷区**（本次）：
- 所有 6 币均为负 Sharpe
- ETH 灾难性（-2.97）
- SOL 相对最不差（-0.24），但仍负
- BNB 仍是老问题（-0.74）

（历史 May 结果仅在 alt 子集 + 极紧 τ 下出现正；6 币 pooled 当时也已 reject）

---

## 4. Go / No-Go 判断

**Promote 门禁（参考 5 月 fast_scalp 标准）**：
- 至少 alt 子集或 6 币整体在 recent_6m_oos 上 @ 某个合理 τ（q≤0.15）出现 **mean Sharpe > 0.5 且多数币正**？
- 或者至少 **3/4 alt 正贡献** 且 BNB 不严重拖累？
- Pearson 仍可接受地弱（排序模型特性）。

**当前结论（2026-06-01 跑完后填写）**：
- [x] **reject**（top-100 在干净 recent_6m_oos 上比 top-20 更差或持平，6 币 pooled 全面负，无可交易 edge）
- [ ] **需更多实验**（见第 5 节建议：tree_core 受控子池、regime-conditional IC、forward_rr target、或彻底放弃大池子自动选材）

**Reject 线**（若触发则下线）：
- rolling 4 周 Sharpe < 0
- BNB 或某 major 持续大亏超预算
- 交易频率过低（无法覆盖成本）

---

## 5. 洞见与行动项

（跑完后填写）

- 是否发现某些 archetype 家族（dtw / box / cvd / spectrum 等）在 100 列里占比过高/过低？
- `recent_6m_oos` vs 之前稍长窗的差异是否显著？
- 是否值得把 `recent_6m_oos` 也加入 TPC gate 的 variant-grid segment_matrix（当前只有三个 canonical）？
- 下一步实验建议（如果本次仍 reject）：
  1. 只在 `tree_core_120T`（~95 nodes）上做 controlled wide（减少垃圾特征）。
  2. IC 阶段加 stability across segments / monotone / regime-conditional 过滤。
  3. 直接用 forward_rr 做 IC target（而非 label）。
  4. 放弃大池子自动选材，回到 5 月“手工 hypothesis pool + 严格 RR 验证”的打法。

---

## 6. 产物与追溯

- 实验定义：本目录（README + rd_loop yaml + 本 DECISION）
- 策略配置：`config/strategies/tree_strategies/short_term_swing_wide_top100_test/`
- IC 排名来源：`results/rd_loop/short_term_swing_ic_plateau/ic_prune_wide_20260601/`
- 市场段定义：`config/market_segment.yaml`（`recent_6m_oos` 条目）
- 最终模型（如 promote）：对应 train artifact + 冻结 τ

---

**更新时间**：2026-06-01（跑数完成 + 结果填入）  
**状态**：实验完成，结论 = **reject**（100 列 wide 池子在最近 6m OOS 上仍不可交易）。
