# 遗留研究命令清理计划

> **状态**：待新命令（`mlbot research` / `rd_loop` / calibrate / promote）在真实 parquet 上验收后再执行。  
> **关联**：[`R&D工具矩阵_CN.md`](R&D工具矩阵_CN.md) · [`迁移_旧meta到新rd_loop_CN.md`](迁移_旧meta到新rd_loop_CN.md) · [`研究工具重构计划_CN.md`](研究工具重构计划_CN.md) · [`完整命令速查表.md`](../完整命令速查表.md)（废弃命令一览）

---

## 0. 文档分工

| 文档 | 内容 |
|------|------|
| [`完整命令速查表.md`](../完整命令速查表.md) | 全仓库 CLI + **过时标记表**（§⚠️）+ `mlbot research` 子命令示例 |
| [`R&D工具矩阵_CN.md`](R&D工具矩阵_CN.md) | ABC 三阶段、TPC/树/chop grid 端到端、calibrate skip / promote |
| **本文** | 代码层依赖、删除顺序、Phase checklist、待删文件清单 |

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

---

## 7. 老命令 → 新命令对照表（清理验收用）

> 与 [`完整命令速查表.md` §⚠️](../完整命令速查表.md#-过时命令标记) 同步。验收通过后可按 Phase 删「老入口」列。

### 7.1 假设筛查（①）

| 老入口 | 新入口 | 可删时机 |
|--------|--------|----------|
| `python scripts/quick_layer_scan.py condition-set ...` | `mlbot research scan condition-set ...` | Phase 2（内联 scan 内核后） |
| `quick_layer_scan.py feature-plateau` | `mlbot research scan feature-plateau` 或 `plateau --kpi label` | Phase 2 |
| `quick_layer_scan.py pair-scan` | `mlbot research scan pair-scan` | Phase 2 |
| `quick_layer_scan.py ic-decay` | `mlbot research ic` 或 `mlbot analyze factor-eval` | Phase 2 |
| `mlbot pipeline run` + `research_roll.features_on.yaml`（discovery） | `scripts/rd_loop.py` + `mlbot research *` | Phase 1（yaml 标记）→ Phase 5（删 stage） |
| `analyze_archetype_feature_stratification.py --promote` | `mlbot research fit` + 人审 + variant-grid | Phase 4 |

### 7.2 Gate / Entry 精标（②b）

| 老入口 | 新入口 | 可删时机 |
|--------|--------|----------|
| `python scripts/optimize_gate_unified.py --strategy tpc ...` | `mlbot research plateau --kpi lift` / `rd_loop gate-plateau` | Phase 3 |
| `mlbot optimize gate-plateau` / `gate-plateau-all` | 同上（B 系统）；nnmh 链路或归档 | Phase 3 或单独归档 nnmh |
| `python scripts/optimize_entry_filter_plateau.py ...` | `mlbot research plateau --kpi snotio` / `rd_loop entry-plateau` | Phase 3 |
| `optimize_entry_filter_snotio.py` | `research plateau --kpi snotio` | Phase 3 |
| `locked_prefilter_parquet_tune.py` | `rd_loop locked-prefilter-tune` | Phase 3（库函数可保留） |
| `tune_locked_prefilter_thresholds.py` | `research plateau --layer prefilter` + calibrate | Phase 3–4 |
| `--promote` / pipeline adopt 写生产 | `mlbot research calibrate` + `promote --yes` | Phase 1 起禁用 adopt |

### 7.3 因果验证（②a）

| 老入口 | 新入口 | 可删时机 |
|--------|--------|----------|
| `validate_static.full_study.yaml` 整段 | `event_backtest --variant-grid` + decision doc | Phase 1 yaml → Phase 5 |
| `validate_static.constrained.yaml` | variant-grid + 单层 research 精标 | 同上 |
| 手工 cp draft yaml 无 backup | `mlbot research promote`（locked merge + backup） | 已可用，无需等 Phase |

### 7.4 监控（③ — 不删）

| 入口 | 说明 |
|------|------|
| `calibrate_roll.default.yaml` | 月 replay；**保留** |
| `pre_deploy_replay.yaml` + `pre_deploy_contract_checks.py` | 上线 contract；**保留** |
| `regime_watchdog.py` / `regime_drift_monitor.py` | 周监控；**保留**（drift 可 emit rd_loop suggestions） |

---

## 8. 待删文件清单（按 Phase）

### Phase 2 目标

- [ ] `scripts/quick_layer_scan.py`（逻辑迁入 `src/research/` 或 `scripts/research/`）
- [ ] 更新 `scripts/research/scan.py`、`ic.py`、`plateau.py` 的 import
- [ ] 修 `tests/research/test_scan_parity.py`、`tests/unit/test_quick_layer_scan_modes.py`

### Phase 3 目标

- [ ] `scripts/optimize_gate_unified.py`（或仅保留 HTML 报告子模块）
- [ ] `scripts/optimize_entry_filter_plateau.py`
- [ ] `scripts/optimize_all_archetypes_plateau.py`
- [ ] `scripts/optimize_entry_filter_snotio.py`
- [ ] `scripts/run_gate_optimization_experiments.py`、`experiment_gate_plateau_optimization.py`
- [ ] `auto_research_pipeline.py` 内 gate/entry optimize subprocess 分支

### Phase 4 目标

- [ ] `scripts/meta_algorithm_unified.py`（或移 `bad-candidates/`）
- [ ] `analyze_archetype_feature_stratification.py` 内 meta 自动 promote 路径
- [ ] 各策略 `research_roll.features_on.yaml` 中 `meta_algorithm: true` → false 或归档

### Phase 5 目标

- [ ] `auto_research_pipeline.py` discovery stages（validate_static、research_roll discovery）
- [ ] `config/strategies/*/research/validate_static.*.yaml`（归档，不物理删亦可）
- [ ] `src/cli/main.py` 中 `mlbot optimize gate-plateau*` DEPRECATED 别名

### 清理前必修测试

- [ ] `tests/unit/test_compute_lift_for_threshold.py` → import `src.research.stat_kernels.gate_lift.compute_lift_for_threshold`
- [ ] `tests/research/stat_kernels/test_gate_kernels_parity.py` 保持「legacy 委托同一 kernel」断言

---

## 9. 新命令 canonical 清单（删老代码后的目标面）

```
mlbot research scan|ic|plateau|segment|fit|compare|robustness|calibrate|promote
scripts/rd_loop.py
scripts/research/gate_plateau_scan.py
scripts/research/entry_plateau_scan.py
scripts/research/drift_suggestions.py
python -m scripts.event_backtest --variant-grid
scripts/regime_watchdog.py
scripts/regime_drift_monitor.py
scripts/pre_deploy_contract_checks.py
mlbot pipeline run + calibrate_roll.default.yaml | pre_deploy_replay.yaml
src/research/stat_kernels/*
src/research/gate_when.py
```

**不在目标面（非 R&D discovery）**：`mlbot train final`、`mlbot pipeline run` 生产训练、`deploy_config_to_live.py`、live 运维脚本。

---

## 10. 验收 gate（启动 Phase 2 前）

与 §5 硬指标一致，另加：

- [ ] [`完整命令速查表.md`](../完整命令速查表.md) 与本文对照表无矛盾
- [ ] TPC + ME（或 SRB）真实 `features_labeled.parquet` 跑通 §R&D工具矩阵 端到端示例
- [ ] 团队默认不再新开 `research_roll` / `quick_layer_scan` 实验（仅 legacy 对拍）

验收负责人签字 / 日期：________________ （待填）
