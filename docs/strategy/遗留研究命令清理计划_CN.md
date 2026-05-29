# 遗留研究命令清理计划

> **状态**：待新命令（`mlbot research` / `rd_loop` / calibrate / promote）在真实 parquet 上验收后再执行。  
> **关联**：[`R&D工具矩阵_CN.md`](R&D工具矩阵_CN.md) · [`迁移_旧meta到新rd_loop_CN.md`](迁移_旧meta到新rd_loop_CN.md) · [`研究工具重构计划_CN.md`](研究工具重构计划_CN.md)

---

## 1. 现状：重写 vs 抽出 vs 未动

当前实现 **不是整体重写**，而是三层结构：

```
┌─────────────────────────────────────────────────────────┐
│  新入口：mlbot research * / rd_loop / calibrate / promote │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  scripts/research/*   quick_layer_scan.py   optimize_*_unified.py
  (薄 CLI)             (老 scan CLI)          (老 optimize CLI + HTML/promote)
        │                   │                   │
        └───────────────────┴───────────────────┘
                            ▼
              src/research/stat_kernels/*  +  gate_when.py  +  expr.py
              （共享数学/解析内核）
```

### 各块对照

| 能力 | 做法 | 新命令 | 老管线 |
|------|------|--------|--------|
| **Gate lift/plateau** | ✅ 抽出内核 | `plateau --kpi lift` → `gate_lift_scan` / `gate_plateau_scan` → `gate_optimize.py` | `optimize_gate_unified.py` import 同一 kernel |
| **Gate when 解析/写回** | ✅ 新模块 | `calibrate` → `gate_when.py` | kernel 间接共用 |
| **Scan（condition-set / feature-plateau / pair-scan）** | ⚠️ 未重写 | `mlbot research scan` **直接调** `quick_layer_scan` | 仍可单独跑 `quick_layer_scan.py` |
| **IC decay** | ⚠️ 未重写 | `mlbot research ic` → `quick_layer_scan.mode_ic_decay` | 同上 |
| **Label plateau** | ⚠️ 未重写 | `plateau --kpi label` → `quick_layer_scan.feature_plateau_payload` | 同上 |
| **Entry snotio** | ⚠️ 半抽出 | `plateau --kpi snotio` + `entry_plateau_scan.py` | `optimize_entry_filter_plateau.py`；pipeline subprocess |
| **calibrate / promote** | ✅ 全新 | 仅新路径 | 老管线靠 `--promote` / adopt |
| **Meta 特征发现** | ❌ 未动 | `research fit` 仅 audit | `analyze_archetype_feature_stratification.py` + `meta_algorithm_unified.py` |
| **locked prefilter 数值** | ❌ 未动 | `rd_loop locked-prefilter-tune` | `tune_locked_prefilter_thresholds.py` / pipeline auto-tune |
| **bundle 编排** | ❌ 未动 | 不用 | `auto_research_pipeline.py` 串老 optimize / meta |

---

## 2. 删除风险分级

### 现在不能删（删了会断）

| 老入口 | 仍被谁依赖 |
|--------|-----------|
| `quick_layer_scan.py` | `scripts/research/scan.py`、`ic.py`、`plateau.py`（label）直接 import |
| `optimize_gate_unified.py` | `auto_research_pipeline.py` subprocess；HTML 报告；`export_lightgbm_rules_to_readme.py` 等 |
| `optimize_entry_filter_plateau.py` | `auto_research_pipeline.py` entry 分支 |
| `analyze_archetype_feature_stratification.py` + `meta_algorithm_unified.py` | `research_roll.features_on.yaml`（`meta_algorithm: true`） |
| `auto_research_pipeline.py` + pipeline yaml | `calibrate_roll` / `pre_deploy_replay`（③ 监控，非 discovery） |

### 已知 refactor 后遗症（清理前需修）

- `tests/unit/test_compute_lift_for_threshold.py` 仍从 `optimize_gate_unified` import `compute_lift_for_threshold`，符号已迁至 `src/research/stat_kernels/gate_lift.py`。

### 可延后删（低价值 wrapper）

- `optimize_all_archetypes_plateau.py` 等已 DEPRECATED 的薄 wrapper。

### 不要与 discovery 混删

- **`calibrate_roll` / `pre_deploy_replay` / `regime_watchdog`** 属于 ③ 监控，新 doctrine 弃用的是 `research_roll` **做发现**，不是弃用全部 `mlbot pipeline run`。

---

## 3. 推荐删除顺序（Phase 1–5）

```
Phase 1 — Discovery 入口切换（配置/文档）
  research_roll / validate_static 不再当默认入口
  → yaml 标记 ROUTINE_R&D_DEPRECATED；代码暂留

Phase 2 — Scan 去壳
  quick_layer_scan 逻辑迁入 src/research 或 scripts/research
  → 改 scan.py / ic.py / plateau.py 的 import
  → 删 quick_layer_scan.py
  → 修 test_scan_parity / test_compute_lift_for_threshold 等引用

Phase 3 — Optimize 去壳
  auto_research_pipeline gate/entry 分支改调 scripts/research/*
  → 删 optimize_gate_unified.py（或仅留 HTML 报告模块）
  → 删 optimize_entry_filter_plateau.py

Phase 4 — Meta 发现迁移
  research_roll 全部 meta_algorithm: false
  特征发现改 research fit + 人工审 SHAP
  → 删 meta_algorithm_unified 链（或移 bad-candidates）

Phase 5 — Bundle 瘦身
  auto_research_pipeline 只保留 train / calibrate_roll / pre_deploy
  → 删 discovery stage 与 validate_static 路径
```

---

## 4. 各 Phase 前置条件（checklist）

### Phase 2 前置

- [ ] `mlbot research scan|ic|plateau --kpi label` 在 ME/TPC 真实 parquet 验收通过
- [ ] `tests/research/test_scan_parity.py` 改指向新模块后仍绿
- [ ] 无外部脚本直接 `python scripts/quick_layer_scan.py`（文档除外）

### Phase 3 前置

- [ ] `rd_loop gate-plateau` + `calibrate` + `promote` 全链路在 TPC smoke / train_final 验收
- [ ] `auto_research_pipeline.py` 中 `gate_optimize` 分支改调 `gate_plateau_scan` 或 subprocess `mlbot research plateau --kpi lift`
- [ ] entry-plateau 与 `optimize_entry_filter_plateau` 输出 parity 对拍（可选）

### Phase 4 前置

- [ ] 各策略 `research_roll.features_on.yaml` 中 `meta_algorithm: false` 或 yaml 归档
- [ ] prefilter 新候选流程文档化：`research fit` → 人审 → variant-grid

### Phase 5 前置

- [ ] ③ calibrate_roll / pre_deploy 独立跑通，不依赖 discovery stage
- [ ] CI / 单测全绿；历史 results 路径无硬编码老脚本名

---

## 5. 验收后再清理的硬指标

1. TPC / ME 至少各跑通一次：`rd_loop` → `calibrate` → `promote --dry-run` → `event_backtest --variant-grid`
2. Gate lift parity：`gate_optimize` kernel 与 legacy CLI 委托同一对象（见 `test_gate_kernels_parity.py`）
3. pre_deploy `cross_regime_evidence` + `plateau_stability` 在 staging 无 BLOCKED
4. 全量 `pytest tests/research tests/unit/test_rd_loop.py -q` 绿

---

## 6. 参考：canonical 内核位置

| 内核 | 路径 |
|------|------|
| Gate 优化 | `src/research/stat_kernels/gate_optimize.py` |
| Gate lift 扫描 | `src/research/stat_kernels/gate_lift.py` |
| Gate when | `src/research/gate_when.py` |
| Plateau 检测 | `src/research/stat_kernels/plateau.py` |
| Snotio | `src/research/stat_kernels/snotio_calc.py` |
| IC | `src/research/stat_kernels/ic.py` |
| 条件 DSL | `src/research/expr.py` |
| Entry batch | `scripts/research/entry_plateau_scan.py` |
| Gate batch | `scripts/research/gate_plateau_scan.py` |

Legacy CLI 应逐步变为 **零数学、仅报告/兼容壳**；数学只存在于 `src/research/`。
