# fast_scalp 树模型 — 配置三层架构

> **原则：** `config/strategies/` 下 **只有 `fast_scalp` 一个模板**；六币 pooled 训练；币种/alt-major 差异只在 **实验层** 用 `symbols` / cohort 验证，不在 strategies 里 fork 多套树。

---

## 三层职责

| 层 | 路径 | 放什么 | 不放什么 |
|----|------|--------|----------|
| **① 策略模板** | `config/strategies/tree_strategies/fast_scalp/` | 唯一 yaml 族（6 币 meta、默认 label/features/archetypes） | 实验 label、IC 产物、gate 池、symbol-pack fork |
| **② 实验计划** | `config/experiments/20260602_fast_scalp_tree_validate/`（当前）；`20260530_.../`（归档） | rd_loop、grid、`overrides/`、`cohorts.yaml`、DECISION | 整棵策略树副本 |
| **③ 冻结快照** | `config_experiments/fast_scalp_alpha_G*_.../` | 仅含 **`fast_scalp/`** 一棵的 event 变体 | `fast_scalp_alts` / `fast_scalp_majors` 子目录 |

---

## ① 唯一模板

```
config/strategies/tree_strategies/
└── fast_scalp/          ← 仅此一个（meta 含 6 币）
```

**训练默认：** pooled 6 币（`BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT`）。

**币种差异去哪？**

| 需求 | 做法 | 不在 strategies 里做 |
|------|------|------------------------|
| 只在 4 alt 上跑 event | grid / CLI `symbols: [SOL,BNB,XRP,ADA]` | ~~fast_scalp_alts/~~ |
| 只在 BTC/ETH 上对比 | grid 一行 `symbols: [BTC,ETH]` | ~~fast_scalp_majors/~~ |
| alt vs major 是否有 edge 差 | `cohorts.yaml` + segment 矩阵多行 | 第二套 train slug |
| 分币 τ | snapshot `per_symbol_thresholds` 或实验 override | 整包复制策略树 |

见 [`cohorts.yaml`](cohorts.yaml)：`pooled_6` / `alts_4` / `majors_2` 是 **实验 cohort 名**，不是 strategy 名。

### 已废弃（strategies/ 下全部清掉或只留 README）

| 目录 | 原因 |
|------|------|
| `fast_scalp_alts` / `fast_scalp_majors` | symbol-pack fork → 实验 `symbols` |
| `fast_scalp_realized_g5/g10` | label override → `overrides/labels_*.yaml` |
| `fast_scalp_dual_head` | dual_head block → `overrides/direction_dual_head.yaml` |
| `fast_scalp_forward_rr_ic_small` 等 | IC 对照 → ic_prune writeback |

---

## ② 实验计划

```
config/experiments/20260602_fast_scalp_tree_validate/   ← canonical overrides + 两轨验证
├── TRAINING.md / DECISION.md / FEATURES_AND_OVERFITTING.md
├── cohorts.yaml
└── overrides/
    ├── labels_execution_aligned_g5.yaml
    ├── labels_dual_head_h3.yaml
    ├── features_gate_candidates.yaml
    └── direction_dual_head.yaml

config/experiments/20260530_fast_scalp_alts_majors/   ← Phase 0–4 归档
├── CONFIG_LAYOUT.md
├── TREE_TRAINING_PLAN.md
├── cohorts.yaml
├── rd_loop_track_dual_head.yaml
├── rd_loop_track_exec_aligned_gate.yaml
└── fast_scalp_*_validate.yaml   ← event grid（strategy: fast_scalp + symbols cohort）
```

训练命令 **始终**：

```bash
--config config/strategies/tree_strategies/fast_scalp
--symbol BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT   # pooled
# 或实验子集：--symbol SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT      # 仅 score/gate/event
```

产物：`results/train_final/fast_scalp/<run_id>/`

---

## ③ 冻结快照（单包）

```bash
PYTHONPATH=src:scripts python scripts/research/prepare_fast_scalp_alpha_snapshots.py
# → config_experiments/fast_scalp_alpha_G14_.../fast_scalp/   （只有这一棵）
```

Event grid 示例：

```yaml
strategy: fast_scalp
strategies_root: config/experiments/20260530_fast_scalp_alts_majors/variants/fast_scalp_alpha_G14_g5label_g5exec_strategies
symbols: [SOLUSDT, BNBUSDT, XRPUSDT, ADAUSDT]   # cohort 在实验层
```

**旧快照**（含 `fast_scalp_alts/` + `fast_scalp_majors/` 三棵）需 **重新 materialize** 后再跑 event。

Promote：优胜 snapshot 的 `fast_scalp/archetypes/*` → 复制回 deploy `fast_scalp/archetypes/`，打 `locked: true`。

---

## 与 TPC 对齐

| TPC | fast_scalp |
|-----|------------|
| 一个 `config/strategies/tpc/` | 一个 `config/strategies/tree_strategies/fast_scalp/` |
| 实验 override + rd_loop | `overrides/` + `rd_loop_track_*.yaml` |
| `config_experiments/*_strategies/` | `config_experiments/fast_scalp_alpha_G*_strategies/` |
