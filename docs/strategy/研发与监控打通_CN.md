# 研发与监控打通

> **Schema**: [`config/monitoring/monitor_bundle_schema.yaml`](../../config/monitoring/monitor_bundle_schema.yaml)  
> **Promote 准则**: [`config/experiments/LAYER_PROMOTION_CRITERIA.md`](../../config/experiments/LAYER_PROMOTION_CRITERIA.md) §4.3  
> **漂移消费端**: [`漂移监控_BC统一设计_CN.md`](漂移监控_BC统一设计_CN.md)

## 三条命令

| 时机 | 命令 |
|------|------|
| 脚手架 | `mlbot research init <topic> --strategy tpc --segment recent_6m_oos` |
| Phase 1 draft | `PYTHONPATH=src:scripts python scripts/rd_loop.py --hypothesis-yaml config/experiments/<topic>/rd_loop_<topic>_phase1.yaml` |
| Phase 5 promote | `mlbot research promote-baseline --experiment-dir config/experiments/<topic> --enable-drift-ready` |

一次性迁移（无 prior draft）：

```bash
mlbot research promote-baseline \
  --strategy tpc --layer regime \
  --parquet results/monitoring/tier0/tpc_*/features_labeled.parquet \
  --enable-drift-ready
```

## TPC labeled regime：regime_shares + PSI（不做 regime 层 plateau）

| 检测 | TPC regime | 说明 |
|------|------------|------|
| `regime_shares` | ✅ | 与 `RegimeConfig.classify()` 同源 |
| feature plateau (adx/ema P50 带) | ❌ | 与 shares 冗余/打架 |
| PSI (watchdog) | ✅ | 合同列分布形变 |

## draft vs promote

| | draft | promote |
|--|-------|---------|
| 阶段 | Phase 1 末（rd_loop `monitor_bundle.mode: draft`） | Phase 5（`promote-baseline`） |
| 输出 | `config/experiments/<topic>/monitor_bundle/` | git: `regime_watchdog_baseline.json`, `reference/*_psi_ref.parquet`, `regime.yaml` `last_calibration` |
| 进 git | 否 | 是 |

## Bundle 消费链（promote 后现网已接线）

- **drift**: [`scripts/regime_drift_monitor.py`](../../scripts/regime_drift_monitor.py) → `regime_watchdog_baseline.json` → `regime_shares`
- **watchdog PSI**: `factor_ic_baseline_ref` → IC JSON → `source_parquet` → [`evaluate_psi_features`](../../src/research/stat_kernels/drift.py)
- manifest: [`config/monitoring/weekly_rule_stack.yaml`](../../config/monitoring/weekly_rule_stack.yaml)

## PSI ref 存储

- 路径: `config/monitoring/reference/<slug>_psi_ref.parquet`
- 仅 PSI 合同列 + `forward_rr`；v1 **进 git**，不用 git-lfs
- `bundle.json` 含 `psi.sha256`

## rules_hash

export 时对 `allowed_regimes` 算 SHA1；promote 写入 baseline。规则变更后 smoke 报 `RULES_STALE`，需人工重跑 promote。

## rd_loop yaml 片段

```yaml
features_parquet: &parquet results/.../features_labeled.parquet
quick_layer_scans: [...]
monitor_bundle:
  strategy: tpc
  layer: regime
  parquet: *parquet
  out_dir: monitor_bundle
  mode: draft   # promote 请用 mlbot research promote-baseline
```

## 内部实现

[`scripts/monitoring/export_monitor_bundle.py`](../../scripts/monitoring/export_monitor_bundle.py) — rd_loop 与 `promote-baseline` 共用。
