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

## 标准 R&D 阶段（新特征 / 新周期 — 必须先走完 Phase 0–2 再跑 grid）

**禁止跳步**：未在 Phase 1/2 用 IC、label scan 或专用历史扫描 **证明特征有用并标定窗宽/阈值** 之前，不得把拍脑袋的 τ / lookback 写进 `variant_grid` 当 promote 依据。  
（反例：手选 `lookback=120`、`box_breakout>=0.5` 直接 segment grid — 只能当探索性 ablation，**不得**进 DECISION promote。）

| Phase | 名称 | 做什么 | 典型工具 / 产物 | 能否 promote？ |
|-------|------|--------|-----------------|----------------|
| **0** | 特征可算 | 新列进 `feature_dependencies` + 策略 `features*.yaml`；有 labeled parquet | `train_strategy_pipeline --prepare-only` → `features_labeled.parquet` | 否 |
| **1** | 假设扫描 | IC、label plateau、condition-set、pair-scan、snotio | **`mlbot research scan`** / **`mlbot research ic`** / **`mlbot research plateau`**；批量 → **`rd_loop_*.yaml`** | **否**（仅生成假设） |
| **2** | 定参 | 从 Phase 1 读 plateau / scan 报告，**人写** τ、lookback、组合逻辑；更新实验 `DECISION.md` 假设表 | `quick_scan/*.md`、`results/*/research/*_scan.json` | 否 |
| **3** | 因果复验 | `segment_matrix` + `market_segment.yaml` canonical 三阶段 event_backtest | `*_grid.yaml` + `scripts.event_backtest --variant-grid` | **仍否**（除非过下面三条杠） |
| **4** | 人审 | 胜出变体全窗 **trading map**（语义是否对齐，如 BPC 是否仍追高） | `run_trading_maps.sh` | 否 |
| **5** | Promote | 满足本文 **三条杠** → 写 prod archetype + `locked: true` + §4 监控 bundle | `LAYER_PROMOTION_CRITERIA` §4.3 | **是** |

**参考实验（顺序正确）**：

- TPC gate：`20260531_tpc_gate_validate/` — Phase 1 rd_loop only → Phase 2 grid  
- TPC 深回撤：`20260530_tpc_deep_pullback/` — Pass 1 plateau → Pass 2 condition-set → Phase 3 ablation  
- TPC macro 窗宽：`mlbot research scan` +（可选）`scan_tpc_pullback_lookback.py` 标定 binding → `20260610_tpc_macro_pullback_replace/` grid  
- BPC box×depth×lookback：`20260611_bpc_lookback_retest_validate/rd_loop_bpc_box_pullback_phase1.yaml`（**勿**写 `scan_bpc_*.py`）

**窗宽 / 多尺度**：若压缩区用 box N、回踩用 soft_phase L，须在 Phase 1 分别扫描或论证对齐；**不得**默认 `box_breakout@120` 与 `lookback_breakout@240` 混用且无文档。

**Phase 1 工具优先级（防 AI 忘）**：

1. **默认**：`mlbot research scan`（`feature-plateau` / `condition-set` / `pair-scan`）+ `rd_loop.py` 编排；输入 `features_labeled.parquet`。
2. **binding 窗宽**（`lookback_breakout` 等）：`box_pos_60/120/240` 等同 parquet 多列用 ①；改 binding 则 **N 次 `mlbot train final --prepare-only`**（各实验树）再 plateau。
3. **禁止**：为 B/C 新写 `scripts/research/scan_<topic>.py` 重复已有 scan 内核。
4. **例外**：`scripts/research/scan_tpc_pullback_lookback.py` — 在不重跑 N 次 prepare 时 OHLC 重算 TPC depth/macro binding；**不是** BPC 模板。

---

## 操作落地（推荐 checklist）

1. 任何新规则先在 **Phase 1** 用 **`mlbot research scan`**（或 `rd_loop_*.yaml` 批量）生成假设；**不要**新写 `scan_*.py` 除非符合上文「例外」。
2. **Phase 2**：在 `DECISION.md` 记录从扫描选定的 τ / lookback；再建本实验 `variants/*_strategies/` 静态树。
3. **Phase 3**：必须用 **segment_matrix + market_segment.yaml** 里定义的 canonical segments 做完整 variant-grid 事件回测（G0 基线 vs 新变体）。
4. 在对应 `config/experiments/<date>_<topic>/DECISION.md` 里用表格呈现每个 segment 的 Total R、maxDD、CAGR、胜率、tail contrib 等。
5. **Phase 4**：segment 胜出者跑 trading map，核对入场语义（尤其 prefilter 周期错配）。
6. 只有 **Phase 5** 同时满足“三条杠”的变体，才允许：
   - 写入 `config/strategies/<family>/archetypes/*.yaml`（+ live/highcap 同步）
   - 打 `locked: true` + `promote_never_disable: true`
   - 删除所有对应的 disabled 历史痕迹
7. 原则上每个 layer 最终只保留“当前已验证最好”的那套规则，历史实验留在对应 `config/experiments/<dir>/variants/` 快照里即可。
8. **Promote 后更新「平台基线」并 `git push`**（远程 drift 只读 git；**不要**上传 `train_final` parquet）。见下文 §4 与 [`docs/strategy/漂移监控_mlbot_monitor_CN.md`](../../docs/strategy/漂移监控_mlbot_monitor_CN.md) §10。

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

1. **Phase 1 draft**: `rd_loop` with `monitor_bundle.mode: draft` → `config/experiments/<topic>/monitor_bundle/bundle.json` (not in git).
2. **Phase 5 promote**: `mlbot research promote-baseline --experiment-dir config/experiments/<topic> --enable-drift-ready` writes:
   - **Labeled (TPC)**: `regime_shares` + `rules_hash` → `regime_watchdog_baseline.json` + `regime.yaml` `last_calibration.regime_shares` (**no** regime-layer feature plateaus for TPC).
   - **Legacy plateau regime**: `last_calibration.plateaus` on calibration parquet quantiles.
   - **PSI ref**: trimmed columns → `config/monitoring/reference/<slug>_psi_ref.parquet`.
   - **IC baseline** (if `forward_rr` in parquet): `config/monitoring/factor_ic_baseline_<slug>_*.json`; `factor_ic_baseline_ref` in watchdog baseline.
3. **PSI contract**: gate/prefilter columns in manifest `psi_features` (default 3 + `adx_50` for TPC).
4. **Decision + reproduce**: finish `DECISION.md`; `git push` → remote `git pull` + deploy `live/highcap`.
5. **Do not ship**: full `results/train_final` to prod for cron; remote builds current via bus export + archive-batch (see drift doc §7).

**One-shot migration** (no prior draft): `mlbot research promote-baseline --strategy tpc --parquet <calibration.parquet> --enable-drift-ready`.

Repeat after ALERT-driven R&D **only if** a new promote changes thresholds or monitored features.

See also: [`docs/strategy/研发与监控打通_CN.md`](../../docs/strategy/研发与监控打通_CN.md).

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
