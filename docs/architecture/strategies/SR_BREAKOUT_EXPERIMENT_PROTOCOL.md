# SR Breakout 实验协议（Layer A/B 研究记录）

本文件记录 `sr_breakout` 的 **可复现**实验流程与结论，重点覆盖：
- Layer A（label/sample/backtest 语义固定）
- Layer B（feature group search + 语义化 scene 特征验证）

> 本文的目标不是“写漂亮的总结”，而是：**任何人拿到相同数据/同一版本代码，可以复跑并得到同样结论**。

---

## 0. 关键工件（你应该看哪些文件）

- **最终建议特征 YAML（Pool A suggested）**：`config/strategies/sr_breakout/features_suggested.yaml`
- **feature-group-search 输出目录**：`results/feature_group_search/sr_breakout_best_combo_v4/`
  - `feature_group_search_report.html`：总览（baseline、stop_reason、rejected groups、history）
  - `feature_group_search_why.html`：每一步候选对比细节（why selected / why rejected）
  - `feature_group_search_result.json`：结构化结果（程序读写用）
  - `feature_group_search_candidates.csv`：每步所有候选分数
  - `feature_group_search_history.csv`：每步选中的一条轨迹
- **scene 方向对齐 A/B 试验输出**：`results/ab_tests/sr_breakout_scene_direction_ab/`

---

## 1. Layer A（固定项）

本轮结论建立在以下固定项之上（不应在 Layer B 过程中变动）：

- **策略配置目录**：`config/strategies/sr_breakout`
- **时间周期**：`240T`
- **样本窗口**：通过 `TRAIN_START_DATE` / `TRAIN_END_DATE` 环境变量裁剪（由工具传入）
- **评估切分**：`test_size=0.3`
- **评分目标**：`objective=Sharpe_mean`
- **最少交易数约束**：`min_trades=10`
- **多 seed 稳定性**：`seeds=1..5`

---

## 2. Layer B：feature-group-search（Greedy Forward Selection）

### 2.1 运行方式（命令模板）

（示例，实际以你运行时的参数为准）

```bash
mlbot diagnose feature-group-search \
  -c config/strategies/sr_breakout \
  -s BTCUSDT -t 240T \
  --start-date 2023-01-01 --end-date 2025-10-31 \
  --test-size 0.3 --seeds 1,2,3,4,5 \
  --objective Sharpe_mean --min-trades 10 \
  --max-steps 6 \
  --writeback-yaml config/strategies/sr_breakout/features_suggested.yaml \
  --output-dir results/feature_group_search/sr_breakout_best_combo_v4 \
  --deterministic --no-docker
```

### 2.2 baseline（step0）与最终结果（收敛）

来自：`results/feature_group_search/sr_breakout_best_combo_v4/feature_group_search_result.json`

- **baseline_score（Sharpe_mean）**：`0.8322`
  - Sharpe_std：`1.4016`（提示：sr_breakout 的 seed 方差较大）
- **selected_groups**：`trade_cluster_scene` → `wick_scene`
- **stop_reason**：`no_improvement`

### 2.3 为什么选了 `trade_cluster_scene` + `wick_scene`？

从 `feature_group_search_history.csv` / `feature_group_search_report.html` 可见：

- **step1 加 `trade_cluster_scene`**：Sharpe_mean 从 `0.8322` → `0.9286`
- **step2 再加 `wick_scene`**：Sharpe_mean 从 `0.9286` → `1.3029`，且 Sharpe_std 从 `0.86` 进一步降到 `0.54`（更稳）

这两组对 breakout 的直觉解释：

- **trade_cluster_scene**：更像“点火/延续质量”相关的订单流语义（更契合突破策略的 label）
- **wick_scene**：突破失败/回抽的“拒绝”语义，对真假突破判断更直接

（更细的“why selected”逐项对比，见 `feature_group_search_why.html`）

### 2.4 为什么 `vpin_scene / wpt_scene` 等被拒绝？

在 `feature_group_search_report.html` 的 “Rejected groups” 中已记录：

- `vpin_scene`
- `liquidity_void_scene`
- `wpt_scene`
- `volume_profile_scene`
- `fp_scene`

它们在 step1 的 Sharpe_mean 均未超过 baseline，因此被拒绝（Greedy Forward 的规则：必须严格提升 objective 才会进入下一步）。

---

## 3. 写回 YAML：最终建议特征集合是什么？

写回文件：`config/strategies/sr_breakout/features_suggested.yaml`

- **final requested_features**：在原 baseline 基础上新增：
  - `trade_cluster_scene_semantic_scores_f`
  - `wick_scene_semantic_scores_f`
- 文件里包含 `feature_group_search` provenance 元数据（selected_groups / stop_reason / groups_source 等），用于复盘与审计。

---

## 4. 方向对齐 A/B：invert absorption/exhaustion 能救回 vpin_scene / wpt_scene 吗？

背景：我们怀疑 “scene 语义特征包含 absorption/exhaustion 可能对 breakout 是反信号”，于是做快速 A/B：

- **vpin_scene_raw**：baseline + `vpin_scene_semantic_scores_f`
- **vpin_scene_invert_abs_exh**：在 raw 基础上对输出列 `vpin_absorption_score` / `vpin_exhaustion_scene_score` 做 inversion（×-1）
- **wpt_scene_raw**：baseline + `wpt_scene_semantic_scores_f`
- **wpt_scene_invert_abs_exh**：对输出列 `wpt_absorption_score` / `wpt_exhaustion_score` 做 inversion（×-1）

输出目录：`results/ab_tests/sr_breakout_scene_direction_ab/`

### 4.1 结果（BTCUSDT, seeds=1,2）

（均值仅用于快速方向判断；严谨结论以 seeds=1..5 为准）

- **vpin_scene_raw**：Sharpe ≈ `-0.705`（mean），trades ≈ `24`
- **vpin_scene_invert_abs_exh**：Sharpe ≈ `-0.586`（mean），trades ≈ `83`
  - 结论：inversion 改变了交易行为（交易数上升），但仍显著为负，未“救回”。

- **wpt_scene_raw**：Sharpe ≈ `-0.108`（mean），trades ≈ `127`（但 seed 间极不稳定：-1.09 vs +0.87）
- **wpt_scene_invert_abs_exh**：Sharpe ≈ `-0.428`（mean），trades ≈ `100`
  - 结论：inversion 反而更差，且把一个 seed 的正结果也抹掉。

### 4.2 结论

对 sr_breakout 来说，**vpin_scene / wpt_scene 的“方向对齐（invert absorption/exhaustion）”不是主要问题**。
更可能的原因是：

- 这两组 scene 特征对当前的 label/execution 组合整体不匹配，或噪声过大；
- sr_breakout 本身 seed 方差大，单独加这两组会放大不稳定性。

---

## 5. 下一步建议（保持工业化节奏）

- **继续 Layer A 稳定化**：降低 baseline 的 seed 方差（例如固定执行阈值、检查标签分布/预测分布/交易数诊断块）
- **Layer B 扩展到多 symbol**：BTC/ETH/SOL 复核（防止单品种偶然）
- **再考虑对 `wpt_scene` 做“子集/条件化评估”**：如果它只在特定 regime（如 compression_high）有效，应通过 slice 评估确认，而不是全局均值判断。


