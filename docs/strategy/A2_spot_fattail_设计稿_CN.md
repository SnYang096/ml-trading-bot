# A2 spot_fattail — 设计稿（未上 live，默认不建）

> **状态**：`config/strategies/bad-candidates/spot_accum/` 下仅有讨论稿；**当前决策是深化 A1 `spot_accum_simple`，不单独上 spot_fattail**（见 [spot_fattail 讨论](config/strategies/bad-candidates/spot_accum/spot_fattail.md) 与 portfolio 分工：A=cycle inventory，B=trend alpha，C=micro alpha）。
>
> 本文档描述：**若未来**要验证「尾部事件驱动的现货加仓」假设，应如何走与 C 语义代理同构的 R&D 环（[`WORKFLOW_整体架构与管线改进计划_CN.md`](WORKFLOW_整体架构与管线改进计划_CN.md) §2.2 A2 行），**不动 live**。

---

## 1. 与 A1 的分工

| | A1 spot_accum_simple | A2 spot_fattail（假设） |
|---|---|---|
| 目标 | 熊市周期 inventory（周线 EMA200 下 deploy） | 极端尾部事件后的**有限次**加仓窗口 |
| 触发 | 规则化、极低频 | OI surge / funding 极端 / 链上巨鲸等 **尾部代理** |
| 与 B/C | 互补（非 trend timing） | 若做成「宽 stop 趋势」会与 **BPC 重叠** — 故默认不做 |
| R&D | 几乎无 | 尾部代理 R&D（Q，若启用） |

---

## 2. 尾部代理 KPI（验收指标）

与 B 的 `success_no_rr_extreme` 不同，A2 应用 **事件后 N 日** 的现货侧指标：

| KPI 列名（建议） | 含义 |
|---|---|
| `tail_event_forward_return_7d` | 事件 bar 后 7 日 log return |
| `tail_deploy_success` | deploy 后未触发 cycle-death 且 30d return > 0 |
| `max_adverse_7d` | 事件后 7 日内最大回撤（现货口径） |
| `tail_z_oi_change` | OI 变化 z-score（事件定义用） |

离线筛：候选特征与上述 KPI 的 Spearman / 分桶 lift（`quick_layer_scan`，连续 label 时用 `seg_*` 风格或改 label 为 bool `tail_deploy_success`）。

---

## 3. 候选特征池（命名语义）

从 `features.yaml` / 衍生品 / 链上扩展，示例：

- `oi_change_zscore`, `funding_rate_extreme`, `liquidation_cluster_score`
- `whale_inflow_z`, `exchange_netflow_spike`
- `evt_var_99`, `vol_leverage_asymmetry`（与 B gate 共用列但 **验收 KPI 不同**）

---

## 4. R&D 闭环（若启用）

```text
定义尾部事件 + KPI
  → quick_layer_scan（feature-plateau / condition-set / ic-decay）
  → shadow 回测（规则阈值，非 mlbot train 主路径）
  → 人审 promote（仅 config/strategies/spot_fattail/，非 live）
  → deploy_config_to_live.py（显式开关）
```

命令模板（占位路径）：

```bash
PYTHONPATH=src:scripts python scripts/quick_layer_scan.py condition-set \
  --features-parquet results/spot_fattail/features_labeled.parquet \
  --label tail_deploy_success \
  --condition "oi_z: oi_change_zscore>=2.0" \
  --out results/spot_fattail/quick_scan/tail_proxy_<日期>.md

PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/spot_fattail_hypothesis.yaml
```

---

## 5. 明确不做

- ❌ 用 spot_fattail 替代或复制 BPC 突破/宽 stop 趋势逻辑
- ❌ 默认 rolling turbo / SHAP 写回 live
- ❌ 在未证明与 A1 互补前上 PCM 槽位

---

## 6. 何时重新打开此设计稿

- 有 **独立** 尾部事件样本 ≥ 200 次（跨 2+ 周期）
- A1 spot_accum 的 cycle inventory 指标已稳定（deploy 曲线 + bull exposure 可复盘）
- 团队明确需要「事件驱动 deploy」且与 B 相关性 < 0.5（危机段）

在此之前：**继续 A1**，见 [`ABC新流程验证checklist_CN.md`](ABC新流程验证checklist_CN.md) §A。
