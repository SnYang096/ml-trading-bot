# SRB 两段式首仓门（cross 2a + EMA1200 2b）接入 event_backtest 的诊断

日期：2026-04-24  
结论：**默认保持 `enabled: false`**。门控是"随机减仓"而不是"挑好单"，但风险调整维度明显更优，留作可选开关。

## 1. 背景

`scripts/experiment_srb_staged_entry_2a2b.py` 离线试过 "先等 cross 确认 (2a)、再等 EMA1200 位置+斜率同向 (2b)"
两段式首仓。本次把它接进 **event_backtest**：

- 新模块 `src/time_series_model/live/srb_staged_entry_2b.py`
  - `SrbStagedEntry2bRuntime.from_execution_block(raw)` 读 yaml-dict
  - API 拆成 `match_arm(sym, side, bar_idx)`（不消费）+ `consume_arm(sym)`
    ——避免 `open_position` 失败时误清 arm
  - `advance(...)` 每根 primary bar 推进：`swing_sr_levels` → `update_cross_state` → post-2a 等待 2b → `_arm`
- `scripts/event_backtest.py`：`_primary_bar_count` 自增后 `advance`；PCM 新开 SRB 母仓前用
  `match_arm` 门控（funnel key：`reject_srb_staged_2b_arm`），`open_position` 成功后再 `consume_arm`
- `config/strategies/srb/archetypes/execution.yaml`：新增 `srb_staged_entry_2b` 块，默认
  `enabled: false`
- `scripts/fast_ab_srb_staged_2b.sh`：baseline vs treatment（仅切 `enabled`）快 AB
- `scripts/summarize_fast_ab_srb_multi.py`：加 `reject_srb_staged_2b_arm` 漏斗列
- `tests/unit/test_srb_staged_entry_2b.py`：冒烟测试

## 2. 快 AB 结果（16 月，2023-09 ~ 2024-12）

`results/reports/srb_fast_ab_staged_2b/`

### 2.1 全部 trade（含加仓）

| arm | n | totalR | expR | winRate | **maxDD_R** | sharpe-like |
|---|---:|---:|---:|---:|---:|---:|
| baseline  | 213 | +230.19 | 1.081 | 48.4% | **−36.35** | 0.226 |
| treatment | 135 | +194.65 | **1.442** | **51.9%** | **−23.22** | **0.291** |

- 每单期望 +33%，胜率 +3.5pp
- **max DD −36%**
- Sharpe-like +29%
- totalR 回吐 −35.6R（−15%）

### 2.2 仅母仓（剔除加仓）

| arm | n | totalR | expR | winRate | maxDD_R |
|---|---:|---:|---:|---:|---:|
| baseline 母仓  | 143 | +36.36 | 0.254 | 44.1% | −13.07 |
| treatment 母仓 |  91 | +25.35 | 0.279 | **48.4%** | **−9.54** |

- 母仓 expR 只从 0.254 → 0.279（+10%）
- 母仓胜率 +4.3pp
- 证据不够支持"门选到了更好入场"，更像"随机减仓 40%"

### 2.3 仅加仓

| arm | n | totalR | expR | winRate | maxDD_R |
|---|---:|---:|---:|---:|---:|
| baseline 加仓  | 70 | +193.84 | 2.77 | 57.1% | −36.10 |
| treatment 加仓 | 44 | +169.30 | **3.85** | 59.1% | **−14.79** |

加仓 expR 2.77 → **3.85**（+39%），加仓回撤减半 —— 母仓被筛掉的那批也带走了后续"差加仓"。

## 3. 关键证据：被 2b 挡掉的 91 单

用 `(symbol, entry_time)` 对齐两臂母仓：

```
baseline 母仓总数       : 143
treatment 母仓总数      : 91
两臂同单（完全重合）    : 52
treatment 独有（新入场）: 39    totalR=+9.38R  expR=+0.241
baseline 独有（被 2b 挡）: 91   totalR=+20.39R expR=+0.224 winRate=41.8%
  top10:  [+9.95, +7.47, +4.54, +3.76, +3.73, +3.73, +3.47, +2.75, +2.59, +2.16]
  bot10:  [−1.08, −1.08, −1.08, −1.06, −1.06, −1.04, −1.03, −1.03, −1.03, −1.03]
```

- 被挡 91 单的 expR (**+0.224R**) 与 treatment 保留的 91 单 expR (**+0.279R**) 几乎一致
- 被挡单里有 3 单 ≥+4R（单笔肥尾），这是 treatment totalR 回吐的主要来源
- **结论**：staged 2b 没有 "system-level 挑到更好入场"；它更像"抽样减仓"

另一个明显信号：**所有月 `reject_srb_staged_2b_arm` 漏斗 key 均为 0**
—— 意味着被减少的 78 单 PCM 尝试根本从未进入"armed 但 mismatch"分支，而是 arm 根本没 fire。
cross+EMA 二段确认在默认参数下（`confirm_k=3, post_2a_max_bars=24, arm_pcm_bars=8,
ema_pos_min=0, ema_slope_bars=2`）偏严，arm 窗口常常错过 PCM 发出的 bar。

## 4. 决策

| 目标 | 选择 |
|---|---|
| 最大化 totalR / 吃肥尾加仓 | **关**（baseline 占优） |
| 控制 max DD / Sharpe / 稳态组合 | **开**（treatment 占优） |

当前 SRB 独立跑时选 **关**（默认 `enabled: false`）。未来做多策略 portfolio 叠加（alpha+beta 分层）时，
可以再把它切回 **开** 并与其它减 DD 手段对齐。

## 5. 待办（不在本次 PR）

1. 参数扫：`confirm_k=2`, `post_2a_max_bars=48`, `arm_pcm_bars=16`,
   `ema_pos_min=-0.01` 看是否能把 arm 触发率拉高到"真正看到 mismatch 拒绝"。
2. 把硬门改成"尺寸调节器"：armed 时 1.0×，未 arm 时 0.5× —— 保留肥尾，压缩差入场的暴露。
3. Live 路径尚未接（与 `srb_cross_state_machine` 现状一致，仅事件回测）。

## 6. 关键文件

- `src/time_series_model/live/srb_staged_entry_2b.py`
- `scripts/event_backtest.py`（`_srb_staged_rt` / `match_arm` / `consume_arm` 接线）
- `config/strategies/srb/archetypes/execution.yaml`（`srb_staged_entry_2b` 块，默认关）
- `scripts/fast_ab_srb_staged_2b.sh`
- `scripts/summarize_fast_ab_srb_multi.py`（漏斗列）
- `scripts/_analyze_staged_2b_quality.py`（本次入场质量分析脚本，可复用）
- `tests/unit/test_srb_staged_entry_2b.py`
