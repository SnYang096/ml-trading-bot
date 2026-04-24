# FBF 改进矩阵筛选结果（2026-04-17）

## 背景与目标

本次目标是针对 FBF 的三类问题做快速筛选:

- 止损占比高（用户体感“容易被洗”）
- 入场偏右侧
- short 侧可能拖累整体收益

为控制实验时长，先用代表月做筛选，再决定是否补完整 slow rolling。

## 筛选口径（统一）

- pipeline: `config/prod_train_pipeline_2h_fbf_matrix_screen.yaml`
- stage: `fast_month`
- 代表月: `2023-12`, `2024-02`, `2024-10`, `2024-12`
- 对比维度: `total_r`, `n_trades`, `win_rate`

说明:

- 该口径用于版本筛选，不直接替代完整 slow rolling 结论。
- 所有实验共用相同筛选口径，横向比较有效。

## 实验版本

### V0: baseline（轻量口径基线）

- `entry_filters.yaml`: short/long 均开启
- `archetypes/gate.yaml`: 空 gate
- `execution.yaml`: breakeven 关闭

### V1: long-only

- 在 V0 上只改一项:
  - `config/strategies/fbf/archetypes/entry_filters.yaml`
  - `fbf_short_upper_fail.enabled: false`

### V2: long-only + gate

- 在 V1 上再加一项:
  - 将 `gate_draft` 的统计 deny 规则 promote 到 `archetypes/gate.yaml`

## 结果对比

| 月份 | baseline (R, n) | long-only (R, n) | long+gate (R, n) |
|------|------------------|------------------|------------------|
| 2023-12 | `+0.000, 0` | `+0.000, 0` | `+0.000, 0` |
| 2024-02 | `-1.330, 4` | `-0.227, 3` | `+0.000, 0` |
| 2024-10 | `-1.062, 1` | `-1.062, 1` | `-1.062, 1` |
| 2024-12 | `-3.313, 6` | `-3.313, 6` | `-3.155, 3` |
| **SUM** | **`-5.704, 11`** | **`-4.601, 10`** | **`-4.217, 4`** |

增量（相对 baseline）:

- `long-only`: `+1.103R`，交易数 `-1`
- `long+gate`: `+1.487R`，交易数 `-7`
- `gate` 相对 `long-only`: `+0.384R`，交易数 `-6`

## 解读

### 1. 关闭 short 分支有正贡献

在代表月口径下，`long-only` 明确优于 baseline，核心贡献来自 `2024-02`（从 `-1.33R` 改善到 `-0.23R`）。

这与此前完整样本观察一致:

- short 侧在历史上是净拖累
- FBF 当前更像 long-biased failed-breakout repair

### 2. gate 能进一步“减亏”，但代价是明显降频

`long+gate` 比 `long-only` 再提升 `+0.384R`，但交易数从 `10` 笔降到 `4` 笔。

含义:

- 统计 gate 的方向是对的（过滤掉了一部分坏交易）
- 但当前阈值组合偏激进，可能过度压缩了可交易样本

### 3. breakeven-only 在本筛选口径下无贡献

此前已测 `breakeven-only` 与 baseline 在代表月上结果一致，未观察到改善。

## 当前建议

### 建议保留方向

- 保留 `long-only` 作为 FBF 下一轮默认底座
- 保留 gate，但建议后续做“降杀伤”微调（避免过度减频）

### 建议暂缓方向

- 暂不把 `breakeven-only` 作为主改动推进
- 暂不做更激进 execution 结构改动（先把 entry/gate 语义打磨完）

## 下一步（建议）

1. 在 `long-only + gate` 底座上测试 `entry_filters_ab_B_cvd_confirm.yaml`（重点验证“更贴 SR”）
2. 只对胜出版本补跑完整 slow rolling（16 个月）确认结论稳健性
3. 若完整 slow 仍优于当前生产版，再考虑 promote 到主策略配置
