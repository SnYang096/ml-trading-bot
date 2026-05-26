# R&D 流程方法论（quick_layer_scan → event_backtest → watchdog）

> 这是 [`WORKFLOW_整体架构与管线改进计划_CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md) 的**执行手册**：把"假设 → 实验 → 上线 → 监控"压成一条固定流程，不需要每次现编。
>
> 适用范围：
> - **B 系统**（chop/box/EMA-band 趋势）— 本手册的主流程；
> - **A1 spot_accum_simple** — 规则化，不走 R&D 闭环；
> - **A2 spot_fattail**（规划中）— 若引入 live，按 [`WORKFLOW_..._CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md) §2.2.1 思路做"尾部代理 R&D"；
> - **C 系统**（chop_grid / trend_scalp）— **不走** 本手册的 SHAP/方向 label 工厂，但仍要做 [`WORKFLOW_..._CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md) §2.2.1 的 **C 语义代理 R&D** 季度循环（同工具：`quick_layer_scan` + 多腿回测 + `_new_decision_doc.py`）。
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
[1] quick_layer_scan.py      ← 1-2 分钟，离线 label/桶诊断
   │                           过滤显著: |z|>2 且 Δsucc>+0.5pp
   │
   ▼
[2] 准备 variant 配置          ← cp config_experiments/<base>_strategies → <new>_strategies
   │                            改 1-2 个 yaml
   │
   ▼
[3] event_backtest 两段        ← 2024 bull + 2025-2026 recent, ~30 min/段
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
| `scripts/quick_layer_scan.py` | 假设时 | features_labeled.parquet | markdown 报告（含 `ic-decay`） | ❌ |
| `scripts/event_backtest.py` | 候选确定后 | strategies_root + symbols + 日期；或 `--variant-grid` | trades csv + summary + capital report + EXPERIMENT_INDEX | ❌ |
| `scripts/regime_watchdog.py` | 周度 cron | recent features parquet + baseline + IC baseline | report.json（含 IC/PSI）+ summary.txt | ❌ |
| `scripts/_new_decision_doc.py` | promote 前 | EXPERIMENT_INDEX.json | `docs/decisions/<topic>_<date>.md` 骨架 | ❌ |
| `scripts/regime_drift_monitor.py` | 周/月度 cron | recent features parquet | drift report | ❌ |
| `scripts/deploy_config_to_live.py` | yaml change | config/ + live/highcap/ diff | 同步到 live + 重启 quant-feature-bus | ✅ live only |

**只有 deploy 这一步真正改 live yaml**。其它都只产出报告或临时实验目录。

## 2. 阶段细节

### 2.1 阶段 [1] quick_layer_scan：把"我猜某层某特征该调"变成 1 分钟扫描

三种模式：

**A. feature-plateau** — 单特征阈值扫描，看 plateau 是否真的存在
```bash
PYTHONPATH=src:scripts python scripts/quick_layer_scan.py feature-plateau \
  --features-parquet results/<train_final>/tpc/features_labeled.parquet \
  --label success_no_rr_extreme \
  --feature tpc_pullback_depth --operator "<=" \
  --grid 0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95 \
  --filter "tpc_semantic_chop<=0.4" "ema_1200_position>=0.10" \
  --calendar-window 2024-01-01,2025-01-01 \
  --out results/tpc/quick_scan/<topic>_<日期>.md
```

**判读**：
- `succ_hit ≈ succ_other` 且 `|z|<2` → 阈值在 plateau 上，**调它没用**
- `|z|>2` 且 succ_hit > succ_other → 阈值方向正确，可调严
- `|z|>2` 但 succ_hit < succ_other → 阈值方向**反了**（如本次 TPC pullback 深 vs 浅）

**B. condition-set** — 比较若干 regime 条件
```bash
PYTHONPATH=src:scripts python scripts/quick_layer_scan.py condition-set \
  --features-parquet results/<train_final>/tpc/features_labeled.parquet \
  --label success_no_rr_extreme \
  --filter "tpc_semantic_chop<=0.4" \
  --condition "H: abs(ema_1200_position)>0.10" \
  --condition "F': abs(ema_1200_position)>0.10 AND abs(ema_1200_slope_10)>0.002" \
  --out results/tpc/quick_scan/regime_candidates_<日期>.md
```

**判读**：
- Δpp 是 succ_in − base_success（与 base mask 比，不是与 succ_out 比）
- `|z|<2` → 候选与 base 没显著差，跳过
- `|z|>2` + Δpp ≥ +0.5pp → 拉 event_backtest 验证 R-multiple

**C. pair-scan** — 二维 deny 表（如 vp × vla 联合 deny）

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
PYTHONPATH=src:scripts python scripts/event_backtest.py \
  --strategy tpc \
  --strategies-root config_experiments/<new>_strategies \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --start-date <start> --end-date <end> \
  --trades-csv results/tpc/experiments/<run_name>/event_trades_tpc.csv \
  --capital-report results/tpc/experiments/<run_name>/ \
  --output results/tpc/experiments/<run_name>/summary.json \
  --quiet-signal-logs
```

约 30 分钟/段 × 6 symbols。可两段并行（占 ~2 core）。

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

```bash
# 同步配置
cp config_experiments/<new>_strategies/tpc/archetypes/gate.yaml \
   config/strategies/tpc/archetypes/gate.yaml
cp config/strategies/tpc/archetypes/gate.yaml \
   live/highcap/config/strategies/tpc/archetypes/gate.yaml

# 重新计算 watchdog baseline
PARQ=$(ls -t results/train_final/tpc/train_final_*/tpc/features_labeled.parquet | head -1)
# 修改 config/monitoring/regime_watchdog_baseline.json
# 然后 smoke test:
PYTHONPATH=src:scripts python scripts/regime_watchdog.py \
  --strategies tpc --window-parquet "$PARQ" \
  --baseline-json config/monitoring/regime_watchdog_baseline.json
```

**写决策文档** `docs/decisions/<topic>_<日期>.md`，最低要求：
- 变体定义表
- 双段结果表（含 by-side）
- 决策理由（为什么这个比其他变体好）
- 复现命令
- 已知坑

### 2.6 阶段 [7] 周度监控

加 cron（建议每周一早上）：

```bash
0 8 * * 1 cd /home/yin/trading/ml_trading_bot && \
  PYTHONPATH=src:scripts python scripts/regime_watchdog.py \
    --strategies tpc \
    --window-parquet results/<recent>/features.parquet \
    --baseline-json config/monitoring/regime_watchdog_baseline.json \
    || /usr/bin/notify-send "regime_watchdog ALERT"

0 8 * * 1 cd /home/yin/trading/ml_trading_bot && \
  PYTHONPATH=src:scripts python scripts/regime_drift_monitor.py \
    --strategies tpc,bpc,me,srb \
    --window-parquet results/<recent>/features.parquet
```

**任一 alert 触发新一轮 R&D**（回到阶段 [1]）。

## 3. 反模式（不要做）

| 反模式 | 为什么 | 应该 |
|---|---|---|
| 跳过 quick_layer_scan，直接跑 backtest | 假设可能压根不成立，浪费 30min | 先 1-2 分钟扫 label |
| 单段 walk-forward 决策 | 时段相关，B/H 案例已证 | 强制双段 |
| 看 label success rate 等于看 R-multiple | label 高 ≠ 总 R 高（vol gate 案例） | 两者都看 |
| 自动 promote（含 SHAP / optimizer 提议） | 人审才能抓住 cross-regime bug | 人审 + 决策文档 |
| 直接改 live yaml 不更新 baseline | watchdog 失去意义 | promote 时同步重算 baseline |
| `cp -r` 覆盖了实验目录的修改没注意 | 见 H 第一次跑挂的真实事故 | 改完 md5 / diff 校验 |

## 4. 案例索引（学这条流程怎么用）

| 案例 | 入口文档 | 触发的工具链 |
|---|---|---|
| TPC vol gate ABH 实验 | [`docs/decisions/tpc_gate_vol_ABH_experiment_20260526.md`](../decisions/tpc_gate_vol_ABH_experiment_20260526.md) | quick_layer_scan + 7 variants × 2 periods event_backtest + regime_watchdog |

## 5. 与 ML4T 工作流对应

| 阶段 | ML4T 标准 | 我们的实现 |
|---|---|---|
| Hypothesis | 章节 4：因子构造直觉 | `quick_layer_scan` + 人脑 |
| Backtest | 章节 8：vectorbt walk-forward | `event_backtest`（订单流级，比 ML4T 标准更细） |
| Cross-validation | 章节 7：Purged K-fold | **缺**：当前是双段对照，**未实现 Combinatorial Purged CV**（roadmap） |
| Live → Monitoring | 章节 23 | `regime_watchdog` + `regime_drift_monitor`（待加 PSI / IC alert） |

## 6. 后续 roadmap

| 优先级 | 项 | 备注 |
|---|---|---|
| P1 | Combinatorial Purged CV | ML4T 的硬缺项，crypto 特别需要因为 regime 短 |
| P2 | quick_layer_scan 加 `--bucket-by`（自动按 ema/calendar 分桶） | 减少手写 filter |
| ~~P2~~ | ~~regime_watchdog 加 PSI / IC drift~~ | ✅ 已落地 |
| ~~P3~~ | ~~event_backtest 加 `--variant-grid`~~ | ✅ 已落地 |
| ~~P3~~ | ~~自动生成决策文档骨架~~ | ✅ `_new_decision_doc.py` |
