# ABC 统一研究框架（规则栈 + 树通道）

> **定位**：把 A1 / A2 / B（BPC/TPC/ME/SRB）/ C（chop_grid/trend_scalp）/ 树（fast_scalp/short_term_swing）所有策略类型放在 **同一张研究 → 验证 → 上线 → 监控** 图里。
>
> 配套阅读：
> - [`R&D工具矩阵_CN.md`](R&D工具矩阵_CN.md)（各层脚本能力 + pipeline yaml 弃用口径）
> - [`方法论_R_and_D流程_CN.md`](方法论_R_and_D流程_CN.md)（执行手册：命令 / 模板 / 反模式）
> - [`WORKFLOW_整体架构与管线改进计划_CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md)（架构背景与里程碑）
> - [`短期树独立策略_设计与落地_CN.md`](短期树独立策略_设计与落地_CN.md)（树通道的独立 slug 与上线流程）
> - [`label_scan_vs_IC_说明_CN.md`](label_scan_vs_IC_说明_CN.md)（label scan / IC 含义与互补）

---

## 0. 一句话核心 doctrine

| 行为 | 是否自动化 | 谁触发 |
|------|-----------|--------|
| **发现**（特征 / 阈值假设） | 工具固定，**假设由人定** | R&D 人 |
| **验证因果**（双段 backtest） | 实验流程固定，**变体由人定** | R&D 人 |
| **监控**（drift / 缺勤 / IC / PSI） | **cron 全自动** | watchdog |
| **改 yaml** | 必须人审 + decision doc | R&D 人 |

**绝对禁止**：在同一趟运行里 ① 找特征 ② 调阈值 ③ 评分定上线 三件事一起做并 `--adopt`。

---

## 1. 三阶段 × 策略类型矩阵

```
┌── Phase 0 ──┐ ┌──── Phase 1 ────┐ ┌── Phase 2 ──┐ ┌── Phase 3 ──┐ ┌── Phase 4 ──┐
│ 设计/数据    │ │ ① 假设 / 筛特征  │ │ ② 验因果    │ │  人审 promote │ │ ③ 监控/上线  │
│ 人脑 + meta  │ │ scan + IC + 树    │ │ variant-grid│ │  cp + 决策文档 │ │ cron + 告警  │
└─────────────┘ └───────────────────┘ └─────────────┘ └──────────────┘ └─────────────┘
   不改 yaml      不改 yaml             不改生产 yaml    改生产 yaml      不改 yaml
```

| | 规则栈（archetypes/*.yaml） | 树通道（独立 slug） |
|--|----------------------------|--------------------|
| **A1** `spot_accum_simple` | 周 EMA200 死区，年级；几乎不动 | — |
| **A2** `spot_fattail`（规划中） | OI/funding/链上分位 → 离线打分 | optional：尾部代理树 |
| **B** BPC/TPC/ME/SRB | 本框架主线 | 可挂方向树 / gate 树 |
| **C** chop_grid/trend_scalp | 用策略相关 KPI 做语义代理 | — |
| **D** fast_scalp/short_term_swing | — | 主线（独立 slug） |

> A1 不走本闭环；A2 未上 live。B / C / D 全部走下方 §3 同一流程。

---

## 2. 数据三件套（所有策略共享）

| 文件 | 产生方式 | 包含 | 用途 |
|------|---------|------|------|
| `features_labeled.parquet` | `mlbot train final --prepare-only -c config/strategies/<slug>` | 特征 + label（`success_no_rr_extreme`、`forward_rr` 等）+ OHLC + atr | **① 假设阶段唯一输入**；optimize_* / quick_layer_scan / factor-eval 都在此跑 |
| `predictions.parquet` | `mlbot train final -c config/strategies/<slug>`（含树训练） | 上面所有 + 模型 score + gate_decision | 树通道必需；规则栈仅 ②b 时偶用 |
| `trades.csv` / `summary.json` | `event_backtest --variant-grid` | 1m 重放 R-multiple / by-side / DD | **② 验因果唯一可信指标** |

**规则栈 ① 不需要 predictions.parquet** —— 这是 doctrine 的关键改动（详见 §5）。

---

## 3. 统一研究/上线流程（B 与 D 共通）

### 3.1 时序图

```
[Phase 0] 设计 + 数据
   人脑 + meta.yaml + features.yaml
   mlbot train final --prepare-only → features_labeled.parquet
        │
        ├──── 规则栈分支（B / C） ────┐         ┌──── 树通道分支（D / A2） ────┐
        ▼                              │         ▼                                 │
[Phase 1] ① 假设                       │  [Phase 1] ① 假设
  quick_layer_scan feature-plateau     │    mlbot analyze factor-eval --ic-decay
  quick_layer_scan condition-set       │    quick_layer_scan ic-decay → 选 H
  quick_layer_scan ic-decay            │    mlbot train final → predictions.parquet
  posthoc_layer_effectiveness          │    regime_threshold_calibrate（τ plateau）
        │                              │         │
  ②b 数值精标（可选）：                  │  ②b τ plateau 精标：
  optimize_gate_unified                 │    （同名 calibrate 脚本即可）
  optimize_entry_filter_plateau         │
  locked_prefilter_parquet_tune         │
        │                              │         │
        └──────────────┬───────────────┘         └──────────────┬──────────────────┘
                       ▼                                         ▼
[Phase 2] ② 因果验证（统一）
   event_backtest --variant-grid（recent + bull，两段都看 by-side R-multiple）
        │
        ▼  Pareto OK
[Phase 3] 人审 promote
   cp variant → config/strategies/<slug>/...
   docs/decisions/<slug>_<topic>_<YYYYMMDD>.md  （强制：变体表 + 双段 + by-side + 复现命令）
   mlbot monitor watchdog --baseline-refresh    （刷 baseline）
        │
        ▼
[Phase 4] 上线（统一）
   pre_deploy_replay.yaml 或 pre_deploy_contract_checks.py（无 BLOCKED）
   deploy_config_to_live.py（同步到 live/highcap/，人 confirm）
        │
        ▼
[Phase 4-持续] ③ 监控（cron）
   weekly  : regime_watchdog + regime_drift_monitor → result.json + heartbeat → CMS
   monthly : calibrate_roll rolling_sim 或 cron 跑固定 config event_backtest
   缺勤 / ALERT → Telegram
```

### 3.2 阶段命令速查

| 阶段 | 命令 | 适用 |
|------|------|------|
| 0 数据 | `mlbot train final --no-docker --prepare-only -c config/strategies/<slug> --output-dir results/train_final/<slug>/<run_id>` | 全部 |
| ① 规则 plateau | `python scripts/quick_layer_scan.py feature-plateau --features-parquet ... --feature <name> --operator "<=" --grid ...` | B / C 规则栈 |
| ① 规则 condition | `python scripts/quick_layer_scan.py condition-set --features-parquet ... --label success_no_rr_extreme --condition "H: ..."` | B / C 规则栈 |
| ① IC / lag | `python scripts/quick_layer_scan.py ic-decay --features-parquet ... --feature <name> --target forward_rr --lags 1,3,5,10,20` | 全部 |
| ① 树特征池 | `mlbot analyze factor-eval --ic-decay-lags 1,3,5,10,20,50` | 树通道 |
| ①→② 规则数值精标 | `python scripts/optimize_gate_unified.py --logs <features_labeled.parquet> --strategy bpc --output ...`<br/>`python scripts/optimize_entry_filter_plateau.py --logs <features_labeled.parquet> --strategy bpc` | B 规则栈（路线 B：现在可直接吃 features_labeled） |
| ① 树训练 + τ | `mlbot train final -c config/strategies/<fast_scalp|short_term_swing>` → `python scripts/regime_threshold_calibrate.py ...` | 树通道 |
| ② 因果 | `python scripts/event_backtest.py --variant-grid config/experiments/<grid>.yaml` | 全部 |
| Phase 3 决策 | `python scripts/_new_decision_doc.py --topic-template default ...` | 全部 |
| 上线 contract | `mlbot pipeline run --all --config config/strategies/<slug>/research/pre_deploy_replay.yaml --stage rolling_sim --skip-shap` | 全部 |
| 上线 | `python scripts/deploy_config_to_live.py` | 全部 |
| ③ 周 | `python scripts/regime_watchdog.py --strategies bpc,tpc,me,srb --window-parquet ... --baseline-json config/monitoring/regime_watchdog_baseline.json` | 全部 |
| ③ 月 | `mlbot pipeline run --all --config config/strategies/<slug>/research/calibrate_roll.default.yaml --stage rolling_sim --skip-shap` | 全部 |

> 任何 `--variant-grid` / `quick_layer_scan` / `optimize_*` / `factor-eval` 命令**都不动生产 yaml**。只有 Phase 3 的 `cp` 和 Phase 4 的 `deploy_config_to_live.py` 改 yaml，且都有人审与 decision doc 留痕。

---

## 4. 树通道在框架里的位置

> **共用前端、独立后端** —— 树不是新流程，只是把"假设来源"从手写规则换成 LightGBM。

### 4.1 共用前端（① 假设 / 选特征）

```
features_labeled.parquet
   │
   ├─ ic-decay              → 找出 best_lag ≈ H 的候选特征池
   ├─ condition-set         → 验证规则方向（树也用得到，避免拍脑袋）
   └─ feature-plateau       → 验证候选特征的连续阈值是否有 plateau
```

### 4.2 独立后端（两棵树，对应两种语义）

| 树 | 目标 | 训练标签 | 产物 | 阈值标定 | 落地 |
|----|------|---------|------|----------|------|
| **方向树** A | 近期收益方向（最近窗口能赚） | `forward_rr` 二分类 / 回归 | `predictions.parquet` 的 `score` | `regime_threshold_calibrate` τ plateau | 写 `backtest.yaml` 的 `long/short_entry_threshold` |
| **Gate 树** B | 避免大回撤（避开 -0.8R 以下） | `success_no_rr_extreme` | `gate_score` | 同上，单 τ deny | 写 `gate.yaml` 的 `gate_score >= τ` 规则 |

> 这两棵树**与 B 规则栈不合并仓位**（独立 slug），但**共用 features_labeled.parquet 的发现工具**。详见 [`短期树独立策略_设计与落地_CN.md`](短期树独立策略_设计与落地_CN.md)。

### 4.3 树 vs 规则：什么时候选谁

| 场景 | 选 | 原因 |
|------|----|------|
| 单条件解释清楚（如 `chop<=0.4` AND `pullback<=0.7`） | 规则 | 可审计、可手动锁、便于决策文档 |
| 多维非线性组合明显（≥3 个特征互相加成） | 树 | 规则容量不够，树能压住组合 |
| 数据少（<5k bar） | 规则 | 树 overfit 风险高 |
| 数据多 + 想自动适配多 symbol | 树 | 共享一个模型 |

**doctrine**：先用 ① 工具确认"非树规则方案不够"，再上树；不是默认上树。

---

## 5. 单层精标脚本的路线 B 改造（已落地）

> **背景**：旧管线把 `optimize_gate_unified` / `optimize_entry_filter_plateau` 绑在 `predictions.parquet` 上。但这两个脚本的统计内核（lift / plateau / robustness / snotio）**根本不需要模型 score**，只需要 features + label + OHLC + 方向列。
>
> 路线 B = 不重构内核，**只让它们也能吃 `features_labeled.parquet`**，从而脱离 pipeline 依赖。

| 脚本 | 改动 | 现在可以这样跑 |
|------|------|---------------|
| `optimize_gate_unified.py` | label_col 自动从 `forward_rr` 派生 `is_good`（已存在）；docstring 与示例补 `features_labeled.parquet` 入口；启动时打印输入类型 | `python scripts/optimize_gate_unified.py --logs results/train_final/bpc/<run>/features_labeled.parquet --strategy bpc --output results/bpc/gate_scan.json` |
| `optimize_entry_filter_plateau.py` | 缺 `gate_decision` 时打 INFO 并继续（features_labeled 没有此列，等同"无 gate 预过滤"）；docstring 与示例补入口 | `python scripts/optimize_entry_filter_plateau.py --logs results/train_final/bpc/<run>/features_labeled.parquet --strategy bpc` |

**意义**：

1. **R&D ①→②b 全程不需要 pipeline run**。从 `prepare-only` 出 `features_labeled.parquet` 后，scan + 数值精标 + variant-grid 都能离线连跑。
2. **树成为可选增强**，不是必经一步。规则栈的 gate / entry 数值精标不再被树训练时间拖慢。
3. **bundle yaml 历史价值仍在**（contract checks / monthly replay），但**日常 R&D 完全脱钩**。

---

## 6. 监控架构（cron + 远端 CMS + 缺勤告警）

### 6.1 本地节奏

```
本地机（systemd timer 优于 cron：断电补跑，且 OnFailure 触发告警）：

[Unit]
Description=Weekly regime watchdog
[Service]
Type=oneshot
WorkingDirectory=/home/yin/trading/ml_trading_bot
ExecStart=/usr/bin/env bash scripts/monitoring/run_weekly.sh
OnFailure=notify-telegram@%n.service
[Install]
WantedBy=multi-user.target

# Timer：周日 08:00 + Persistent=true 保证错过会补跑
[Timer]
OnCalendar=Sun 08:00
Persistent=true
```

`run_weekly.sh` 三件事：

```bash
#!/usr/bin/env bash
set -euo pipefail
RUN_TS=$(date -u +%Y%m%d_%H%M)
OUTDIR=results/monitoring/weekly_watchdog/${RUN_TS}
mkdir -p "$OUTDIR"

# 1) 跑 watchdog
PYTHONPATH=src:scripts python scripts/regime_watchdog.py \
  --strategies bpc,tpc,me,srb \
  --window-parquet results/<recent_window>/features_labeled.parquet \
  --baseline-json config/monitoring/regime_watchdog_baseline.json \
  --output "$OUTDIR/result.json"

# 2) 写心跳（必须，独立于 result.json）
cat > "$OUTDIR/heartbeat.json" <<EOF
{"task": "weekly_watchdog", "ts": "$(date -u --iso-8601=seconds)", "status": "OK"}
EOF

# 3) 推到远端（git or s3 都行）
git add results/monitoring/weekly_watchdog/${RUN_TS}/
git commit -m "monitor: weekly watchdog ${RUN_TS}"
git push origin monitoring   # 单独 branch，不污染主 trunk
```

### 6.2 远端 CMS 三个职责

| 职责 | 实现要点 |
|------|---------|
| **心跳缺勤告警** | 每个 task 注册 `expected_period_seconds`；CMS 定时扫 `heartbeat.json`，若 `last_ts + 1.2 * expected_period < now` → 触发"缺勤告警"（与脚本失败告警是两条独立通道） |
| **结果展示** | 读 `result.json` summary（KB 级），渲染卡片：sharpe / DD / IC drift / PSI；ALERT / FAIL 标红 |
| **决策入口** | 点 ALERT → 自动链到 `docs/decisions/<topic>_<date>.md` 模板（用 `_new_decision_doc.py`）+ 推荐 `rd_loop` 命令 |

### 6.3 告警 dedupe / 风暴控制

| 问题 | 做法 |
|------|------|
| 同一 ALERT 反复触发 | CMS 端按 (task, alert_signature) 24h 内 dedupe |
| 脚本失败 vs 心跳缺勤 | 走**不同 Telegram channel**，避免互相淹没 |
| `result.json` 太大 | 只 commit summary；详情走对象存储 / 仅本地 |
| 网络中断时本地丢数据 | `run_weekly.sh` 总写本地 `OUTDIR`，push 失败时 systemd `OnFailure` 单独通知，下次自动补 push |

### 6.4 监控任务清单（落地后）

| 频率 | 任务 | 失败/缺勤动作 |
|------|------|--------------|
| 日 | live ledger 健康（已有 quant-feature-bus 日志监控） | Telegram |
| 周日 08:00 | `regime_watchdog` + `regime_drift_monitor` | 心跳 → CMS；缺勤 24h → Telegram |
| 月 1 号 02:00 | `calibrate_roll rolling_sim` 或 cron 跑 `event_backtest` 固定 window | 同上 |
| 每次 deploy 前 | `pre_deploy_contract_checks.py`（人触发，但产物入 CMS 历史） | BLOCKED → 阻止 deploy |

---

## 7. 反模式（统一禁止）

| 反模式 | 应该 |
|--------|------|
| 在 ① 假设阶段就改生产 yaml | 等 ② Pareto OK + 人审 |
| 把 SHAP / `optimize_*` 输出当 promote 决策 | 必须 ② variant-grid 双段验证 |
| 跳过 by-side R 看整体 totR | 强制看 LONG / SHORT 分桶 |
| 单段 walk-forward 拍板 | 强制双段（recent + bull） |
| 把 `optimize_gate_unified` 包进月度 cron | 那是 ②b 工具，不是监控 |
| 把树 score 当 regime 切换器 | regime 永远只用慢变量（EMA1200 / chop） |
| 删掉 `optimize_*` 脚本"因为有 quick_layer_scan" | 它们是 ②b 精标武器，rd_loop 替不了 |
| 删掉 bundle yaml "因为弃用" | 它们仍是 contract checks / monthly replay 的入口，只是不再做 R&D 发现 |

---

## 8. 与历史文档的关系

| 文档 | 关系 |
|------|------|
| [`R&D工具矩阵_CN.md`](R&D工具矩阵_CN.md) | **本文 §3 的工具单**；本文给框架图，工具矩阵给逐工具能力对比 |
| [`方法论_R_and_D流程_CN.md`](方法论_R_and_D流程_CN.md) | **本文的执行手册**；本文给"why + 谁配谁"，方法论给"具体命令 + 反例" |
| [`WORKFLOW_整体架构与管线改进计划_CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md) | 架构背景与里程碑（M1/M4 等） |
| [`短期树独立策略_设计与落地_CN.md`](短期树独立策略_设计与落地_CN.md) | 本文 §4 的详细版（树通道独立 slug 上线流程） |
| [`label_scan_vs_IC_说明_CN.md`](label_scan_vs_IC_说明_CN.md) | label scan 与 IC 在 ① 假设阶段的互补关系 |
| [`时序过滤与横截面排序_为何先TPC再Rank_CN.md`](时序过滤与横截面排序_为何先TPC再Rank_CN.md) | A 股 / ETF 小资金场景的 TPC 在前 + 周线 rank 在后 |
| `research_roll.features_on.yaml` / `validate_static.*.yaml` | 标记 **ROUTINE_R&D_DEPRECATED**，详见 [`R&D工具矩阵_CN.md`](R&D工具矩阵_CN.md) §2 |

---

## 9. 落地路线图（已完成 + 待办）

| 状态 | 任务 |
|------|------|
| ✅ | `quick_layer_scan` 三模式 + `--bucket-by` |
| ✅ | `rd_loop.py` 编排 |
| ✅ | `event_backtest --variant-grid` |
| ✅ | `regime_watchdog` 加 PSI / IC drift |
| ✅ | `_new_decision_doc.py` 模板 |
| ✅ | 本文（统一框架）+ `R&D工具矩阵_CN.md`（弃用口径） |
| ✅ | 路线 B：`optimize_gate_unified` / `optimize_entry_filter_plateau` 支持 `features_labeled.parquet` |
| ⏳ | systemd timer + heartbeat upload 脚本（§6.1 落地） |
| ⏳ | CMS 心跳缺勤告警 endpoint + dedupe |
| ⏳ | 路线 A（远期）：把 lift/plateau/robustness 抽到 `src/research/stat_kernels.py`，scan 与 optimize 共用内核 |
