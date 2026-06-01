# R&D 流程方法论（mlbot research / rd_loop → event_backtest → watchdog）

> 这是 [`WORKFLOW_整体架构与管线改进计划_CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md) 的**执行手册**：把"假设 → 实验 → 上线 → 监控"压成一条固定流程，不需要每次现编。
>
> **命令口径（以本文 + 工具矩阵为准，勿混用旧脚本名）**  
> | 层级 | 文档 | 角色 |
> |------|------|------|
> | ① 假设筛查 | **[`R&D工具矩阵_CN.md`](R&D工具矩阵_CN.md) §1** | `mlbot research scan\|ic\|plateau\|segment` 子命令全文 |
> | 流程步骤 | **本文** | 五步图 + 双段 backtest + promote / watchdog |
> | ABC / 数据三件套 | [`ABC统一研究框架_CN.md`](ABC统一研究框架_CN.md) | Phase 0–4 与 `features_labeled.parquet` |
> | 编排（多 scan 一键） | `config/experiments/rd_loop_*.yaml` + `scripts/rd_loop.py` | 内部调 `mlbot research`，见工具矩阵 §4 |
> | 遗留 | `scripts/quick_layer_scan.py` | 仍可用，打印 DEPRECATED；与 `mlbot research *` 对拍一致 |
>
> **顶层框架（ABC × 规则/树 统一视图）**：[`ABC统一研究框架_CN.md`](ABC统一研究框架_CN.md) —— 本文是它的"流程手册"（命令细节见上表）。  
> **工具矩阵与 pipeline yaml 弃用口径**：[`R&D工具矩阵_CN.md`](R&D工具矩阵_CN.md)（各层脚本 vs `rd_loop` 见 §4）。  
> **滚动 optimize 为何禁止、月监控 vs 季 R&D**：[`为何不做滚动调阈值_与研究节奏_CN.md`](为何不做滚动调阈值_与研究节奏_CN.md)。
>
> 适用范围：
> - **B 系统**（chop/box/EMA-band 趋势）— 本手册的主流程；
> - **A1 spot_accum_simple** — 规则化，不走 R&D 闭环；
> - **A2 spot_fattail**（规划中）— 若引入 live，按 [`WORKFLOW_..._CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md) §2.2.1 思路做"尾部代理 R&D"；
> - **C 系统**（chop_grid / trend_scalp）— **不走** 本手册的 SHAP/方向 label 工厂，但仍要做 [`WORKFLOW_..._CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md) §2.2.1 的 **C 语义代理 R&D** 季度循环（同工具：`mlbot research scan` + 多腿回测 + `_new_decision_doc.py`）。
>
> **Label scan 与 IC 的区别、1pp 含义、为何两者都要看**：见 [`label_scan_vs_IC_说明_CN.md`](label_scan_vs_IC_说明_CN.md)。
>
> **为何先时序过滤（TPC 思路）再横截面 rank、小资金 A 股买几只 / ETF vs 个股、买了以后日频信号怎么不调仓、周线环境 + 每月 rank、要不要做空对冲**：见 [`时序过滤与横截面排序_为何先TPC再Rank_CN.md`](时序过滤与横截面排序_为何先TPC再Rank_CN.md)（§8–§10）。
>
> 适配的 doctrine：
> - 慢变量（regime / prefilter） 只在 Q-级动 yaml，其他时间窗口冻结
> - SHAP / 优化器结果 **不 auto-promote**，全部走人审
> - 单段 walk-forward 不能拍板，必须做"近期 + 历史 bull"双段验证

## 0. 五分钟版

```
假设 (人脑/直觉/盘后复盘)
   │
   ▼
[1] mlbot research / rd_loop   ← 1-2 分钟，离线 label/桶诊断（或 hypothesis yaml 批量扫）
   │                           过滤显著: |z|>2 且 Δsucc>+0.5pp
   │
   ▼
[2] 准备 variant 配置          ← cp config_experiments/<base>_strategies → <new>_strategies
   │                            改 1-2 个 yaml
   │
   ▼
[3] event_backtest 两段        ← `python -m scripts.event_backtest --variant-grid`；2024 bull + recent
   │                            必看: trades, totR, win, maxDD, by-side breakdown
   │
   ▼
[4] cross-regime decision      ← Pareto 优于 baseline 两段 → promote；任一段恶化超阈值 → drop
   │
   ▼
[5] promote 到 config + live  ← cp gate.yaml / regime.yaml 到主 config + live/highcap/
   │                            写 docs/decisions/<topic>_<日期>.md
   │
   ▼
[6] 更新 watchdog baseline     ← regime_watchdog.py 重新算 bull_share / trigger_rate
   │
   ▼
[7] 周度 cron 监控             ← regime_watchdog + regime_drift_monitor
                                  alert → 触发新一轮 R&D
```

## 1. 工具链（保持稳定，不再每次重写）

| 工具 | 触发频率 | 输入 | 输出 | 改 yaml？ |
|---|---|---|---|---|
| **`mlbot research`** `scan` / `ic` / `plateau` / `segment` | 假设时 | `features_labeled.parquet`（`--prepare-only` 或 `--features-parquet`） | markdown + 可选 json | ❌ |
| `scripts/rd_loop.py` | 多步假设编排 | `config/experiments/rd_loop_*.yaml` | `results/rd_loop/<topic>/` + 各 scan md | ❌ |
| `python -m scripts.event_backtest` | 候选确定后 | `--variant-grid` 或 strategies_root + 日期窗 | trades csv + summary + EXPERIMENT_INDEX | ❌ |
| `scripts/regime_watchdog.py` | 周度 cron | recent features parquet + baseline + IC baseline | report.json（含 IC/PSI）+ summary.txt | ❌ |
| `scripts/_new_decision_doc.py` | promote 前 | EXPERIMENT_INDEX.json | `docs/decisions/<topic>_<date>.md` 骨架 | ❌ |
| `scripts/quick_layer_scan.py` | 遗留 | 同上 | 同上（DEPRECATED，对拍用） | ❌ |
| `scripts/_build_grid_segment_labels.py` | C 语义代理离线筛 | grid_segments.csv + features parquet | seg_labeled.parquet（`seg_*` KPI 列） | ❌ |
| `scripts/regime_drift_monitor.py` | 周/月度 cron | recent features parquet | drift report | ❌ |
| `scripts/deploy_config_to_live.py` | yaml change | config/ + live/highcap/ diff | 同步到 live + 重启 quant-feature-bus | ✅ live only |

**只有 deploy 这一步真正改 live yaml**。其它都只产出报告或临时实验目录。

## 2. 阶段细节

### 2.1 阶段 [1] 假设筛查：`mlbot research` 或 `rd_loop`

**首选**：单条命令走 `mlbot research`（子命令与旧 `quick_layer_scan` 模式一一对应，见 [`R&D工具矩阵_CN.md`](R&D工具矩阵_CN.md) §1）。  
**批量**：同一 topic 多条 scan + 可选 variant-grid → `config/experiments/rd_loop_<topic>.yaml` + `scripts/rd_loop.py`（例：`config/experiments/me/rd_loop_me_direction.yaml`）。

**数据准备**（按需一次）：
```bash
mlbot train final --no-docker --prepare-only \
  -c config/strategies/tpc \
  --output-dir results/train_final/tpc/<run_id>
# → features_labeled.parquet
```

三种 scan 模式：

**A. feature-plateau** — 单特征阈值扫描，看 plateau 是否真的存在
```bash
mlbot research scan feature-plateau \
  --strategy tpc --layer prefilter \
  --features-parquet results/<train_final>/tpc/features_labeled.parquet \
  --label success_no_rr_extreme \
  --feature tpc_pullback_depth --operator "<=" \
  --grid 0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95 \
  --subset "tpc_semantic_chop<=0.4 AND ema_1200_position>=0.10" \
  --calendar-window 2024-01-01,2025-01-01 \
  --output results/tpc/quick_scan/<topic>_<日期>.md
```

**判读**：
- `succ_hit ≈ succ_other` 且 `|z|<2` → 阈值在 plateau 上，**调它没用**
- `|z|>2` 且 succ_hit > succ_other → 阈值方向正确，可调严
- `|z|>2` 但 succ_hit < succ_other → 阈值方向**反了**（如本次 TPC pullback 深 vs 浅）

**B. condition-set** — 比较若干 regime 条件
```bash
mlbot research scan condition-set \
  --strategy tpc --layer regime \
  --features-parquet results/<train_final>/tpc/features_labeled.parquet \
  --label success_no_rr_extreme \
  --subset "tpc_semantic_chop<=0.4" \
  --condition "H: abs(ema_1200_position)>0.10" \
  --condition "F_prime: abs(ema_1200_position)>0.10 AND abs(ema_1200_slope_10)>0.002" \
  --output results/tpc/quick_scan/regime_candidates_<日期>.md
```

**判读**：
- Δpp 是 succ_in − base_success（与 base mask 比，不是与 succ_out 比）
- `|z|<2` → 候选与 base 没显著差，跳过
- `|z|>2` + Δpp ≥ +0.5pp → 拉 event_backtest 验证 R-multiple

**C. pair-scan** — 二维 deny 表（如 vp × vla 联合 deny）
```bash
mlbot research scan pair-scan --strategy tpc --layer gate \
  --features-parquet results/.../features_labeled.parquet \
  --pair-a 'vol_persistence:>:0.003,0.01,0.03' \
  --pair-b 'vpin:<=:0.5,0.7' \
  --output results/tpc/quick_scan/pair_<日期>.md
```

**D. IC decay**（树通道 / 方向特征对齐 H）
```bash
mlbot research ic --strategy tpc \
  --features-parquet results/.../features_labeled.parquet \
  --features pulse_z,macd_atr --horizons 1,3,5,10,20,50 \
  --target forward_rr --subset "tpc_semantic_chop<=0.4" \
  --output results/tpc/quick_scan/ic_<日期>.md
```

**E. rd_loop 编排**（多 scan + 双段 grid + decision doc 骨架）

**TPC Gate lift 精调**（`gate-plateau` 模式，替代 `optimize_gate_unified`）：
```bash
python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/tpc/rd_loop_tpc_gate_plateau.yaml
```

**ME Entry / Direction**：
```bash
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/me/rd_loop_me_direction.yaml
```

> **阈值约定**：rd_loop 里 condition-set 的 q50/q90 是 **探测假设**，不是生产 τ；生产阈值来自 `research plateau`（snotio/lift）→ `calibrate` → 人审 → `promote`。详见 [`研究工具重构计划_CN.md`](研究工具重构计划_CN.md) §14.3。

> 遗留：`PYTHONPATH=src:scripts python scripts/quick_layer_scan.py <mode> ...` 与上面对拍；新实验不要从它写起。

### 2.2 阶段 [2] 准备 variant

固定 pattern：

```bash
cp -r config_experiments/<base>_strategies config_experiments/<new>_strategies
# 只改要变的 yaml；保留完整策略树（bpc/me/srb/...），否则 event_backtest 会减半
```

**踩过的坑**：
1. **`--strategies-root` 必须含完整策略目录**（bpc/me/srb/chop_grid/...），仅放 tpc/ 会让 timeline 减半，trades 数减半，与 baseline 不可比
2. 改 regime.yaml 用 list of rules（每条 rule 是 AND 连接），不要嵌套 `all_of`/`any_of` 到深层
3. `gate.yaml` 里加 `ema_1200_position` 等条件可以做 regime-conditional gate，比运行时切换更干净（见 variant H）

### 2.3 阶段 [3] event_backtest 双段验证

**强制规则**：候选必须**两段都跑**才能拍板：
- recent (2025-04 → 2026-04)：当前市场
- bull (2024-01 → 2025-01)：calendar bull market

```bash
# 推荐：variant-grid 一次跑多 variant × 多日期窗（见 config/experiments/<grid>.yaml）
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/<your_grid>.yaml \
  --quiet-signal-logs

# 单 variant 单窗（调试）
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --strategy tpc \
  --strategies-root config_experiments/<new>_strategies \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --start-date <start> --end-date <end> \
  --trades-csv results/tpc/experiments/<run_name>/event_trades_tpc.csv \
  --capital-report results/tpc/experiments/<run_name>/ \
  --output results/tpc/experiments/<run_name>/summary.json \
  --quiet-signal-logs
```

约 30 分钟/段 × 6 symbols。可两段并行（占 ~2 core）。grid yaml 里写 recent + bull 两段，避免手抄日期。

**必查指标**：
| 维度 | 看什么 |
|---|---|
| 整体 R | totR, meanR, sharpe |
| 风险 | maxDD（绝对值百分比）, Ret/DD |
| 频率 | n_trades, trades/month |
| **by side** | LONG totR vs SHORT totR — **必须做**，否则会被"整体涨"骗（H 的 +12.8R 缺口 100% 来自 long） |
| funnel | total_signals_checked / reject_regime / reject_gate_deny — 验证 variant 真生效 |

### 2.4 阶段 [4] cross-regime decision

**Pareto rule**：候选必须**两段都不劣于 baseline 显著**才能 promote。

| 情况 | 决策 |
|---|---|
| 两段都比 baseline 好 | **Promote** |
| 一段好、一段持平（≤2pp DD 恶化、≤5R totR 损失） | 可 promote，记录权衡 |
| 一段好、一段显著恶化 | **Drop** 或做 regime-conditional（如 H 来自 B） |
| 两段持平 | Drop（不冒上线风险换 noise） |

**反例**：B 在 recent +60R 看着是赢家，但 2024 bull DD -13.5% vs baseline -8.6% → 不能直接上 B，必须做 H 这种 regime-conditional。

### 2.5 阶段 [5-6] promote + watchdog

**Promote 后必须更新「平台基线」并 push git**（远程 drift 只读这些；**不要**上传 train_final parquet）。完整 checklist 见 **[`漂移监控_mlbot_monitor_CN.md`](漂移监控_mlbot_monitor_CN.md) §10.6**。

```bash
# 同步配置
cp config_experiments/<new>_strategies/tpc/archetypes/gate.yaml \
   config/strategies/tpc/archetypes/gate.yaml
cp config/strategies/tpc/archetypes/gate.yaml \
   live/highcap/config/strategies/tpc/archetypes/gate.yaml

# 监控 bundle（标定窗 parquet 仅用于本地统计 reference，非远程 current）
PARQ=$(ls -t results/train_final/tpc/train_final_*/tpc/features_labeled.parquet | head -1)
# 1) regime.yaml last_calibration.plateaus（regime_threshold_calibrate）
# 2) 更新 config/monitoring/regime_watchdog_baseline.json
# 3) 可选 mlbot research ic → config/monitoring/factor_ic_baseline_<slug>_<date>.json
PYTHONPATH=src:scripts python scripts/regime_watchdog.py \
  --strategies tpc --window-parquet "$PARQ" \
  --baseline-json config/monitoring/regime_watchdog_baseline.json
git add config/strategies config/monitoring docs/decisions
```

**写决策文档** `docs/decisions/<topic>_<日期>.md`，最低要求：
- 变体定义表
- 双段结果表（含 by-side）
- 决策理由（为什么这个比其他变体好）
- 复现命令
- 已知坑

### 2.6 阶段 [7] 周度监控

权威命令见 **[`漂移监控_mlbot_monitor_CN.md`](漂移监控_mlbot_monitor_CN.md)**。本地与远程同一套 `mlbot monitor`；远程需显式设置 `WATCHDOG_PARQUET`。

```bash
# 周日 cron 或 systemd（etc/systemd/mlbot-weekly-watchdog.timer）
export WATCHDOG_PARQUET=results/<recent>/features_labeled.parquet
mlbot monitor weekly
# 或拆开：
mlbot monitor watchdog --window-parquet "$WATCHDOG_PARQUET" \
  --baseline-json config/monitoring/regime_watchdog_baseline.json
mlbot monitor drift --window-parquet "$WATCHDOG_PARQUET" --emit-rd-loop-suggestions
```

**任一 ALERT（exit 1）→ 人审后新一轮 R&D**（回到阶段 [1]）；monitor **不改** yaml。

## 3. ABC × 层 × 节奏 × 命令 速查

> 把 [`WORKFLOW_..._CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md) §2.2 + §3 + §8 的"哪些层活、哪些层冻、用什么算法"压成 4 张可直接复制的命令表。  
> **任何命令都不改 live yaml**；任何 promote 都走 §2.5 的 `_new_decision_doc.py` + 人审。

### 3.1 A1 系统（spot_accum_simple，规则化，几乎不动）

| 层 | 现状 | 触发 | 命令 / 产物 | 改 yaml？ |
|---|---|---|---|---|
| Regime | 周 EMA200 死区（10 年一更）| 极端事件 | 人脑复盘（无 R&D 工具） | 年度 / 触发 |
| Direction | 周 EMA200 上/下（粗）| 同上 | 同上 | 同上 |
| Execution | 阶梯卖出（5×）| Q | `mlbot backtest --strategy spot_accum_simple ...` | Q / 触发 |

> A2 spot_fattail 还在 bad-candidates 设计稿，需要时再单独建条流程（OI/funding/链上 → 离线打分 → shadow → 人审）。

### 3.2 B 系统（BPC / TPC / ME / SRB，本手册主线）

| 层 | 现状 | 触发 | 命令（可直接 copy-paste） | 产物 | 改 yaml？ |
|---|---|---|---|---|---|
| Regime | EMA1200 dead zone + chop（季度） | Q / drift | `mlbot research scan condition-set --strategy <slug> --layer regime --features-parquet ... --condition "H: abs(ema_1200_position)>0.10" --output results/<策略>/quick_scan/regime_<日期>.md` | regime_<日期>.md | 否 |
| Prefilter | 4 archetype 形态（locked） | 半年 / drift | `mlbot research scan feature-plateau --strategy tpc --layer prefilter --feature tpc_pullback_depth --operator "<=" --grid 0.5,0.6,...,0.95 --output ...` | plateau 报告 | 否（仅复核） |
| Direction | 公式（locked） | 年度 | — | — | 否 |
| Gate | tail veto（小帽子树 / 单 τ）| Q | 先 `mlbot research scan condition-set`（或 `rd_loop`），再 `event_backtest --variant-grid` 双段验 | EXPERIMENT_INDEX.json + `_new_decision_doc.py` | 是（人审 → archetypes/gate.yaml）|
| Entry | OR rules（locked）| Q | `mlbot research plateau` 或 `rd_loop` mode `entry-plateau`（子样本加 regime+prefilter subset）| top-3 候选报告 | 是（人审 → entry_filters.yaml）|
| Execution | 紧 SL + 较快兑现 | 年度 | `execution_opt grid`（不自动 promote）| — | 否（默认）|
| **监控** | bull_share / trigger_rate / IC / PSI | W cron | `mlbot monitor watchdog` / `mlbot monitor weekly`（见 [`漂移监控_mlbot_monitor_CN.md`](漂移监控_mlbot_monitor_CN.md)） | report.json + exit=1 | 否 |
| 漂移触发 | plateau drift / IC sign-flip | W cron | `mlbot monitor drift`（`--emit-rd-loop-suggestions` 可选） | drift_report.json | 否 |

### 3.3 C 系统（chop_grid / trend_scalp，多腿）

> 与 B 关键差异：**用策略相关 KPI**（grid 段内 maker 回合期望 / `adverse_break` 率 / 5/5 period；trend_scalp 段级 `total_r/maxDD/费用占比`）做语义代理对照（[`WORKFLOW_..._CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md) §2.2.1）。

| 层 | 现状 | 触发 | 命令 | 产物 | 改 yaml？ |
|---|---|---|---|---|---|
| Regime（`semantic_chop` 0.50/0.32 双阈）| 半年看 | 半年 / drift | `mlbot research scan feature-plateau --feature bpc_semantic_chop --operator ">="` | plateau 报告 | 否（仅复核） |
| **语义代理选择**（§2.2.1）| 季度 R&D | Q | `mlbot research scan condition-set --label <grid KPI 列>` + `event_backtest --variant-grid config/experiments/chop_grid_proxy_<日期>.yaml` | `results/<slug>/semantic_proxy_scan/<日期>.md` + 多腿回测 | 是（人审 → `entry_feature` / `max_semantic_chop_*`）|
| Prefilter（路由）| locked | 半年 | 同 B Prefilter（仅复核） | — | 否 |
| Execution（grid spacing / fee-aware TP）| 年度 | 触发 | 多腿回测 grid | — | 否（默认）|

### 3.4 树通道（独立 slug，与 B/C 并行；详见 [`短期树独立策略_设计与落地_CN.md`](短期树独立策略_设计与落地_CN.md)）

| 层 | 现状 | 触发 | 命令 | 产物 | 改 yaml？ |
|---|---|---|---|---|---|
| Regime（共享 B 的 EMA1200 / chop）| 季度 | Q | 同 §3.2 Regime | — | 否 |
| **特征 IC 对齐 H** | 每次重训前 | Q | `mlbot analyze factor-eval --ic-decay-lags 1,3,5,10,20,50` → 取 best_lag | IC 衰减曲线 + best_lag 列表 | 否 |
| **训练单棵树**（直接给方向）| 季度 | Q | `mlbot train final -c config/strategies/<fast_scalp\|short_term_swing>` | `predictions.parquet`（含 score） | 否 |
| **τ plateau 标定** | 季度 | Q | holdout 上分位扫描（`scripts/regime_threshold_calibrate.py` 同套统计）| τ 区间 + plateau 中心 | 是（写 `backtest.yaml` 的 `long/short_entry_threshold`）|
| **回测确认** | promote 前 | Q | `event_backtest --variant-grid ...` 双段 | EXPERIMENT_INDEX | 否 |
| Execution（独立 SL/TP/trail）| 年度 | 触发 | — | — | 否（默认）|

### 3.5 一句话总结

| 系统 | 主要 R&D 节奏 | 用什么命令开局 |
|---|---|---|
| A1 | 几乎不动 | — |
| A2（未上 live） | Q | 占位（尾部代理） |
| B | **Q：gate / entry / regime**；W：watchdog | `mlbot research` / `rd_loop` → `event_backtest --variant-grid` |
| C | **Q：语义代理 + 阈值**；半年：regime 复核 | 同 B，但 KPI 换多腿指标 |
| Tree | **Q：IC 对齐 → 训树 → τ plateau**；不与 B 合并仓位 | `mlbot research ic` / `factor-eval` → `mlbot train final` → 回测验证 |

---

## 4. 反模式（不要做）

| 反模式 | 为什么 | 应该 |
|---|---|---|
| 跳过 ① 假设筛查，直接跑 backtest | 假设可能压根不成立，浪费 30min | 先 `mlbot research scan`（1-2 分钟）|
| 新实验仍从 `quick_layer_scan.py` 写起 | 与工具矩阵口径分裂 | `mlbot research *` 或 `rd_loop` |
| 单段 walk-forward 决策 | 时段相关，B/H 案例已证 | 强制双段 |
| 看 label success rate 等于看 R-multiple | label 高 ≠ 总 R 高（vol gate 案例） | 两者都看 |
| 自动 promote（含 SHAP / optimizer 提议） | 人审才能抓住 cross-regime bug | 人审 + 决策文档 |
| 直接改 live yaml 不更新 baseline | watchdog 失去意义 | promote 时同步重算 baseline |
| `cp -r` 覆盖了实验目录的修改没注意 | 见 H 第一次跑挂的真实事故 | 改完 md5 / diff 校验 |

## 5. 案例索引（学这条流程怎么用）

| 案例 | 入口文档 | 触发的工具链 |
|---|---|---|
| TPC vol gate ABH 实验 | [`docs/decisions/tpc_gate_vol_ABH_experiment_20260526.md`](../decisions/tpc_gate_vol_ABH_experiment_20260526.md) | research scan + 7 variants × 2 periods event_backtest + regime_watchdog |

## 6. 与 ML4T 工作流对应

| 阶段 | ML4T 标准 | 我们的实现 |
|---|---|---|
| Hypothesis | 章节 4：因子构造直觉 | `mlbot research` / `rd_loop` + 人脑 |
| Backtest | 章节 8：vectorbt walk-forward | `event_backtest`（订单流级，比 ML4T 标准更细） |
| Cross-validation | 章节 7：Purged K-fold | **不用 K-fold**：`mlbot train` 用 causal `TimeSeriesSplit`；promote 用 **多窗口 walk-forward 共识**（recent + bull + 可选第三段），见 [Wave 3 §3](../experiments/z实验_001_bpc/wave3/02_meta_findings_on_meta_algo.md) |
| Live → Monitoring | 章节 23 | `mlbot monitor`（watchdog + drift，含 PSI / IC drift）— [`漂移监控_mlbot_monitor_CN.md`](漂移监控_mlbot_monitor_CN.md) |

## 7. 后续 roadmap

| 优先级 | 项 | 备注 |
|---|---|---|
| P1 | 多窗口 walk-forward 共识（≥3 段） | 替代 Purged/CPCV；crypto regime 短 → 更需要 **因果** 多段，不是 K-fold 平均 |
| ~~P2~~ | ~~quick_layer_scan `--bucket-by`~~ | ✅ 已落地（`ema` / `calendar` / `feature_quantile`） |
| ~~P2~~ | ~~`scripts/rd_loop.py` 一键 driver~~ | ✅ `mlbot research` scan → variant-grid → decision doc |
| ~~P2~~ | ~~regime_watchdog 加 PSI / IC drift~~ | ✅ 已落地 |
| ~~P3~~ | ~~event_backtest 加 `--variant-grid`~~ | ✅ 已落地 |
| ~~P3~~ | ~~自动生成决策文档骨架~~ | ✅ `_new_decision_doc.py` |
| — | ~~Combinatorial Purged CV~~ | **永久搁置**（Wave 3 §3：非因果、不适合生产 promote） |
