# 平坦高原优化工作流程

## 当前推荐入口（与仓库一致）

Gate 阈值「扫描 + 稳定高原 + 稳健性」的**可执行实现**在：

- **`scripts/optimize_gate_unified.py`**（`python scripts/optimize_gate_unified.py --help`）

方法论背景见 [PLATEAU_OPTIMIZATION_METHODOLOGY.md](./PLATEAU_OPTIMIZATION_METHODOLOGY.md)。

> **已知问题**：`mlbot optimize gate-plateau`、`mlbot optimize gate-plateau-all` 以及 `mlbot rule optimize-gate-plateau` 在 `src/cli/main.py` 中仍指向 **`scripts/optimize_gate_plateau.py`**，该文件**已缺失**，命令会直接失败。在代码修复前，**不要使用下文「历史 CLI」块作为操作步骤**。

---

## 概述

在**已冻结结构与世界观**的前提下，对少量连续阈值做扫描，寻找 **lift 稳定、pass rate 不过分塌、稳健性得分可接受** 的区间（plateau），并可选择写回 `gate.yaml`（`--promote`）。

---

## 优化目标（`optimize_gate_unified.py` 口径）

与旧版「按 archetype × vol 分桶 Sharpe」的叙述不同，统一脚本侧重：

- **Lift**：好样本 vs 坏样本通过率的相对提升（见脚本内注释与默认 `min_lift`）。
- **稳定高原**：在阈值扫描序列上寻找宽度与 pass rate 稳定性满足配置的区间（`find_stable_lift_plateau` 等）。
- **RobustnessScore**：对决策边界平缓性的综合分（`robustness_score.overall_score`）。
- **硬约束**：如 `min_pass_rate` / `max_pass_rate`、`min_combined_pass_rate`、组合 AND 通过率下限等，防止「全拒」或无效区间。

若仍需 **NNMULTIHEAD execution_archetypes** 那套「按规则名单 + gated/raw logs」工作流，需在仓库中**恢复** `optimize_gate_plateau.py` 或把 CLI 改指向统一脚本后再沿用旧文档字段。

---

## 工作流程（推荐）

### 1) 准备 logs

- 一份 Parquet：**含 gate 优化所需特征列** + 用于打标签的收益列（脚本会探测 `bpc_impulse_return_atr` / `forward_rr` / `rr` / `return_atr` 等以生成 `is_good`，或使用已有 `--label-col`）。

### 2) 运行统一优化

```bash
python scripts/optimize_gate_unified.py \
  --strategy bpc \
  --logs results/your_run/trade_logs.parquet \
  --output results/gate_opt/bpc_unified.json \
  --step 0.05 \
  --min-lift 1.0 \
  --min-pass-rate 0.20 \
  --max-pass-rate 0.80 \
  --min-plateau-width 0.05
```

常用可选参数：

- `--gate-path`：自定义 gate YAML（默认用 `config/strategies/<strategy>/archetypes/gate.yaml` 逻辑）。
- `--promote`：优化后将阈值写回 archetypes 下 gate（**谨慎**，先备份）。
- `--prefilter`：先按 YAML 规则过滤再扫 plateau（与生产分布对齐）。
- `--cutoff-date`：仅用该日期之前数据做 IS。
- `--write-back-intervals` / `--interval-method`：输出区间型阈值而非单点。

完整参数以 **`--help`** 为准。

### 3) 看结果与落地

- 控制台会汇总 `stable_plateau_found` / `no_stable_plateau` / skip 等计数。
- `--output` JSON：**按规则 `id` 为 key** 的字典；每条含 `status`、`recommended_threshold`、`plateau_start` / `plateau_end`（若适用）、`lift`、`pass_rate`、`robustness_score` 等（大字段 `scan_results` 默认不落盘）。

---

## 输出 JSON 形态（节选示例）

统一脚本写出的是 **`{ "<rule_id>": { ... } }`**，而非旧文档里「单条 archetype+rule_name」一层结构。示例（字段以实际运行为准）：

```json
{
  "gate_evt_var_99_evt_var_99": {
    "status": "stable_plateau_found",
    "recommended_threshold": 0.73,
    "plateau_start": 0.68,
    "plateau_end": 0.78,
    "lift_at_mid": 1.12,
    "pass_rate_at_mid": 0.31,
    "robustness_score": {
      "overall_score": 0.85
    }
  }
}
```

---

## 历史 CLI（当前不可用，仅保留字段记忆）

以下命令在实现仍指向缺失脚本之前**勿执行**：

```bash
# 以下依赖 scripts/optimize_gate_plateau.py（缺失）
mlbot optimize gate-plateau ...
mlbot optimize gate-plateau-all ...
mlbot rule optimize-gate-plateau ...
```

旧版参数语义（与统一脚本**不一一对应**）：

- `--min-trade-rate`：最小交易率
- `--min-trades-per-bucket`：分桶最小成交数
- `--min-sharpe-threshold`：分桶 Sharpe 下限
- `--threshold-step`：扫描步长（与统一脚本 `--step` 概念接近）

---

## 使用建议

1. **先冻结结构与世界观**，再跑统一脚本；同时动的连续阈值不要超过方法论中的 2～3 个维度（见 METHODOLOGY）。
2. **先备份** `gate.yaml` / `gate_draft.yaml`，再使用 `--promote`。
3. 优化后重新跑基线或事件回测，对比漏斗与尾部指标。
4. 将采纳的阈值与 TaskSpec / 实验记录一并存档，便于复盘。
