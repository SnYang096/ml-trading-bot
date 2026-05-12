# CDR — Conversion Divergence Reversal (Bad Candidate)

CDR 原意是在大级别结构转换 / 失败突破与订单流背离共振时做反向，与 BPC/TPC/ME/SRB 的「确认后延续」区分开。

该策略已从 `config/strategies/cdr` **迁入本目录**，不再作为默认生产候选路径。

## 为什么降级

### 1. Turbo rolling 全样本成交极度稀疏

在 `turbo_fixed_features` 滚动配置下，`results/cdr/calibrate_roll.default/_rolling_sim/` 最新一轮（例如 `20260426_213705`）的 `stitched_summary.json` 显示：

- 约 **27 个月** 窗口内 **`stitched_total_trades` ≈ 30**（月均约 1 笔量级），
- 同期 **`stitched_total_r` ≈ 9R** 量级，

与此前同管线较早 run（例如 **234 笔 / ~20R**）相比，**容量与统计可信度都不足以支撑「上生成 / 上生产」决策**。过稀样本下，连续交易地图上的观感与汇总指标都容易被单笔或单月主导。

### 2. 研究管线仍可复现

历史产物仍保留在 `results/cdr/`；若需继续实验，请使用已更新路径的管线配置：

- `config/prod_train_pipeline_2h_turbo_cdr_only.yaml` 内 `strategies.cdr.config` 指向 **`config/strategies/bad-candidates/cdr`**。

将策略放在 `bad-candidates` 是为了在仓库根策略列表中明确 **「非主推、需额外评审」**，避免与 BPC/TPC/ME/SRB 等主族混淆。
