# Layer Promotion Criteria (Gate + All Layers)

**Status**: Canonical rule for all future promote decisions (codified 2026-06-01 during TPC gate finalization).

**Applies to**: `gate`, `entry_filters`, `prefilter`, `regime`, `direction`, `execution`, and any other archetype layer that can be ablated via variant grids.

---

## The Rule (一句话版)

**只有在 `config/market_segment.yaml` 定义的三个真实市场阶段（或其 focused recent_6m_oos 子集）上，同时满足以下三条，才允许把规则写入生产 archetype YAML 并打 `locked: true`**：

1. **总 R-multiple 明显提升**（primary KPI，跨主要牛市 + 近期窗口都要看）
2. **maxDD 不恶化**（风险预算不被突破）
3. **逻辑可解释 + regime-aware**（不是纯数据挖掘的窄窗阈值）

IC、label scan（feature-plateau / condition-set / quick_layer_scan）、单特征 ICIR 等**只能用来生成假设**，**不能直接用来决定生产配置**。

---

## 为什么 IC / label scan “有用”的特征，完整回测经常失效？（用户原问题）

用户在 2026-06-01 问的原话核心：

> “为什么在 ic 和 labelscan 阶段有用的特征，最后回测比较没有用，是因为 oos 正好没用吗，应该在更加长期的数据上保留他们，还是干脆删了？”

**答案（已反复被 TPC gate 系列实验证伪）**：

- **目标函数完全不同**：label 用 success_no_rr_extreme 或 forward_rr，生产回测用 **total realized R + tail contribution + maxDD + 执行现实（PCM 槽位、加仓、滑点、regime 慢变量联合过滤）**。前者上升不等于后者上升，甚至经常反向。
- **多重检验 + 阈值挖掘幻觉**：扫几百个特征 × 几十个阈值，在有限 regime 窗口里极易找到“统计显著”的东西（尤其是中文特征）。这是数据挖掘，不是稳健 edge。
- **执行摩擦被完全忽略**：label 阶段看不到 gate 真正拦住的是哪些 bar（可能把大尾部赢家也干掉）、PCM 冲突、每日节流、symbol 级风控等。
- **Regime 非平稳**：bear_2022 / bull_2023_2024 里有效的规则，在 2025-2026 high-range chop + 转熊段经常失效或反向。长期数据只会把更多已 drift 的规则“平均”进来，反而更脏。
- **结论**：**干脆删**。留着 disabled 的历史规则只会让配置文件越来越难维护、越来越容易误用（项目风格明确反对 legacy shims）。

只有**跨三个真实阶段的 variant-grid 事件回测 + 总 R + 风险指标** 说话才算数。

---

## TPC Gate 系列的实际判决（2026-05/06 完整链路）

- 0530 deep_pullback ablation + 0531 gate_validate + 0601 regime_gate_extend + 0602 monotonic_validate + 最终 G0/G1 对比，**唯一通过上面三条杠的只有 G1**（两条 bull-only vol 中间带规则全部 disabled）。
- G2（关 chop）、G4/G5/G9（各种 vol_persist 形态）、G6（vol_lev 低尾）、G7（EVT 低尾）等全部在至少一个关键阶段 **总 R 下降或 maxDD 明显恶化**，或逻辑上不可持续。
- 最终形态：**chop gate + prefilter 承担主要过滤职责**，system_safety 不再增加任何 vol_* / EVT gate。所有已 disabled 的历史规则在最终 lock 时**物理删除**（不留包袱）。

这个过程直接催生了本准则，并要求所有后续层（entry / prefilter / regime 等）必须走同样流程。

---

## 操作落地（推荐 checklist）

1. 任何新规则先在 label/IC 阶段生成假设（rd_loop + condition-set / feature-plateau）。
2. 必须用 **segment_matrix + market_segment.yaml** 里定义的 canonical segments 做完整 variant-grid 事件回测（G0 基线 vs 新变体）。
3. 在对应 `config/experiments/<date>_<topic>/DECISION.md` 里用表格呈现每个 segment 的 Total R、maxDD、CAGR、胜率、tail contrib 等。
4. 只有同时满足“三条杠”的变体，才允许：
   - 写入 `config/strategies/<family>/archetypes/*.yaml`（+ live/highcap 同步）
   - 打 `locked: true` + `promote_never_disable: true`
   - 删除所有对应的 disabled 历史痕迹
5. 原则上每个 layer 最终只保留“当前已验证最好”的那套规则，历史实验留在 `config_experiments/` 快照里即可。
6. **Promote 后更新「平台基线」并 `git push`**（远程 drift 只读 git；**不要**上传 `train_final` parquet）。见下文 §4 与 [`docs/strategy/漂移监控_mlbot_monitor_CN.md`](../../docs/strategy/漂移监控_mlbot_monitor_CN.md) §10。

---

## 4. Post-promote「平台基线」与远程漂移（监控 bundle）

**Status**: Codified 2026-06-02. Complements the causal “三条杠” above; does **not** replace variant-grid evidence.

### 4.1 What「平台 baseline」means

| Term | Meaning |
|------|---------|
| **Platform baseline (reference)** | Stats frozen at promote/calibration on a **calibration window** (e.g. `recent_6m_oos`): bull_share, trigger_rates, IC signs, PSI reference distribution, regime **plateau bands** |
| **Current (near-term)** | Built **only on the prod host**: feature-bus export (~7d) + archive-batch (~6m) — **not** rsync’d from local `train_final` |
| **Monitoring contract features** | Columns that participate in drift = **production rule columns** + `psi_features` in manifest + features listed in `regime.yaml` `last_calibration.plateaus` — **not** every column in `features.yaml` |

You do **not** need a platform baseline row for every feature explored in research; only for **promoted, production-monitoring** contracts.

### 4.2 What to commit to git (remote drift reads this)

| Artifact | Commit? | Used by remote drift |
|----------|---------|----------------------|
| `config/strategies/<slug>/archetypes/*.yaml` (locked rules) | ✅ | Rule semantics; gate trigger parsing |
| `regime.yaml` → `last_calibration.plateaus` | ✅ | `regime_drift_monitor` plateau P50 |
| `config/monitoring/regime_watchdog_baseline.json` | ✅ per slug | bull_share, trigger_rate, PSI ref metadata |
| `config/monitoring/factor_ic_baseline_<slug>_*.json` | ✅ when IC monitoring applies | IC sign-flip |
| `config/market_segment.yaml` | ✅ when segments change | `archive-batch` window |
| `config/monitoring/*.yaml` manifests | ✅ | cron `mlbot monitor run` |
| `DECISION.md` / `docs/decisions/*.md` | ✅ | Human audit |
| `results/train_final/**/features_labeled.parquet` | ❌ | **Forbidden** as remote weekly current (C1) |

Remote also needs **local data only on the server**: archive bars, feature-bus, execution ledgers / rolling monthly reports.

### 4.3 Monitoring bundle checklist (after each B/C rule-stack promote)

On the **same calibration window** documented in `DECISION.md` (recommended: `recent_6m_oos`):

1. **Plateaus**: run `regime_threshold_calibrate` (or equivalent) → write `archetypes/regime.yaml` `last_calibration.plateaus`.
2. **Watchdog baseline**: on calibration-window parquet, update `config/monitoring/regime_watchdog_baseline.json` for `<slug>` (bull_share, trigger_rates). That parquet is **reference only**, not remote weekly current.
3. **IC baseline** (if gate IC monitoring applies): `mlbot research ic` → `config/monitoring/factor_ic_baseline_<slug>_<date>.json`; set `factor_ic_baseline_ref` in watchdog baseline.
4. **PSI contract**: list production gate/prefilter columns in `config/monitoring/weekly_<slug>.yaml` `psi_features` (default script list is only 3 features).
5. **Decision + reproduce**: finish `DECISION.md`; `git push` → remote `git pull` + deploy `live/highcap`.
6. **Do not ship**: full `results/train_final` to prod for cron; remote builds current via bus export + archive-batch (see drift doc §7).

Repeat 1–4 after ALERT-driven R&D **only if** a new promote changes thresholds or monitored features.

**Flow (full diagram)**: [`漂移监控_mlbot_monitor_CN.md`](../../docs/strategy/漂移监控_mlbot_monitor_CN.md) §10.2.

### 4.4 Does this repo’s R&D flow guarantee it today?

| Requirement | In flow? | Today |
|-------------|----------|-------|
| Pre-promote dual-segment variant-grid | ✅ §1–3 above | Enforced by doctrine |
| Post-promote monitoring bundle | ⚠️ §2.5 方法论 + this §4 | **Manual**; no CI/deploy gate (T11) |
| Per-strategy IC baseline | ⚠️ | Mostly TPC only (C3) |
| Remote current ≠ reference | ❌ | C1/C6 until T1 export + archive-batch |

**Bottom line**: Passing LAYER_PROMOTION **does not** auto-update drift baselines; treat §4.3 as mandatory human steps until T11 hard-checks land.

---

## 工具支撑

- `config/market_segment.yaml` + `scripts/event_backtest/market_segment.py`
- `variant_grid.py` 的 `segment_matrix` 支持（自动展开日期 + 输出子目录按 segment id 干净命名）
- 每个实验必须有 `README.md`（跑法 + 结果路径）和 `DECISION.md`（结果 + promote 结论）
- Post-promote drift baselines: [`docs/strategy/漂移监控_mlbot_monitor_CN.md`](../../docs/strategy/漂移监控_mlbot_monitor_CN.md) §10；迁移 TODO T11

---

**本文件即为全项目 layer promote 的最高优先级参考**。任何与本准则冲突的 promote 建议，原则上不被接受。

（TPC gate 最终 lock 实验即本准则的第一次完整应用。后续 short_term_swing、fast_scalp 等树的 entry / regime / filter 规则均应遵循。）
