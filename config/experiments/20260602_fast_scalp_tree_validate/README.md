# fast_scalp 树模型验证实验（2026-06-02）

| 字段 | 值 |
|------|-----|
| 策略模板 | **`config/strategies/tree_strategies/fast_scalp`**（唯一，6 币 pooled） |
| 快照 | `config_experiments/fast_scalp_alpha_G*_.../fast_scalp/` |
| 产物 | `results/rd_loop/fast_scalp_tree_validate/` |
| 训练流程 | **[`TRAINING.md`](TRAINING.md)** |
| Promote | [`LAYER_PROMOTION_CRITERIA.md`](../LAYER_PROMOTION_CRITERIA.md) |

## 验证目标（两轨）

| 轨道 | 模型 | Event 快照 | 对照 |
|------|------|------------|------|
| **A** | 双 binary head（long_win + short_win agreement） | **G7** | **G3** short+regime_off |
| **B** | execution-aligned g5-label + IC-prune adverse gate | **G14** / **G16** | **G5** H=3 score + G5 exec |

## 物料

| 文件 | 用途 |
|------|------|
| [`TRAINING.md`](TRAINING.md) | **训练流程与命令（canonical）** |
| [`cohorts.yaml`](cohorts.yaml) | pooled_6 币种列表 |
| [`overrides/`](overrides/) | label / features / dual_head 实验 override |
| [`rd_loop_track_a_dual_head.yaml`](rd_loop_track_a_dual_head.yaml) | 轨道 A 编排 |
| [`rd_loop_track_b_exec_aligned_gate.yaml`](rd_loop_track_b_exec_aligned_gate.yaml) | 轨道 B 编排 |
| [`segment_validate_dual_head.yaml`](segment_validate_dual_head.yaml) | A：四段 event（G7 vs G3） |
| [`segment_validate_exec_gate.yaml`](segment_validate_exec_gate.yaml) | B：四段 event（G14/G16 vs G5） |
| [`DECISION.md`](DECISION.md) | 两轨结论（已完成） |
| [`FEATURES_AND_OVERFITTING.md`](FEATURES_AND_OVERFITTING.md) | 特征选择与过拟合（树 vs 规则） |

## 一键跑法

```bash
cd /home/yin/trading/ml_trading_bot

# 轨道 A — 双 head
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260602_fast_scalp_tree_validate/rd_loop_track_a_dual_head.yaml

# 轨道 B — exec-aligned + gate
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260602_fast_scalp_tree_validate/rd_loop_track_b_exec_aligned_gate.yaml
```

## 前置

- Step 0 `take_profit.target_r` 已统一（见 `20260530_.../DECISION.md` §13）
- 重新 materialize 快照（单包 `fast_scalp/`）：  
  `PYTHONPATH=src:scripts python scripts/research/prepare_fast_scalp_alpha_snapshots.py`

## 历史实验

Phase 0–4 alpha rebuild 归档在 [`../20260530_fast_scalp_alts_majors/`](../20260530_fast_scalp_alts_majors/)。本目录为 **树模型两轨验证** 的独立实验入口。
