# Regime 层（慢变量数据空间）

Regime 在 **Prefilter 之前**评估，约束「这段行情是否允许本策略类交易」，与 archetype 入场形态（prefilter）解耦。

## 运行时顺序

```text
regime → prefilter → direction → gate → entry → evidence → execution
```

实现：`generic_live_strategy.decide()`；向量研究路径：`backtest_execution_layer._apply_regime_vectorized()`。

## `regime.yaml` 当前生效字段

| 字段 | 运行时 |
|------|--------|
| `rules:` | **是** — 与 prefilter 同 schema（`feature` / `operator` / `value`，或 `any_of`） |
| `allowed_sides:` | **是** — direction 算出 ±1 之后掩码 long/short |
| `allowed_regimes:` | **否** — schema 占位；牛熊分桶见 `regime_ablation_report.py` 的 `ema_1200_position` 阈值 |

### TPC（20260522 / 20260526 更新）

仅 **EMA1200 宏观死区**：`ema_1200_position >= 0.10` OR `<= -0.10`（等价 `|pos|>0.10`）。chop 在 **`gate.yaml`**（`gate_tpc_semantic_chop_high`）；box 已移除。

#### Gate vol-bull-conditional（variant H，20260526）

`gate.yaml` 中两条 vol gate 改为 **regime-conditional**：仅当 `ema_1200_position > 0.10`（强多头侧）时才 deny。

| 规则 ID | 触发条件 |
|---------|---------|
| `gate_vol_persistence_vol_persistence_bull_only` | `vp ∈ (0.0029, 0.0616) AND ema_1200_position > 0.10` |
| `gate_tpc_vol_leverage_asymmetry_mid_bull_only` | `vla ∈ (0.0558, 0.1482) AND ema_1200_position > 0.10` |

**动机**：cross-regime event_backtest 显示

- 取消两条 vol gate（variant B）：recent 2025-2026 +17R 但 2024 牛市 maxDD 从 -8.64% 恶化到 **-13.52%**；
- 保留两条 vol gate（baseline A）：2024 牛市 maxDD -8.64%，recent 2025-2026 仅 +43R；
- **保留 bull 段、放开 bear 段（variant H）**：2024 牛市 maxDD **-7.57%**（比 baseline 还低），recent 2025-2026 +47R（比 baseline 改进，仅次于 B 的 +60R）。

H 是 Pareto 强于 baseline 的稳健折中：bull-DD 保护 + recent 上行均得。

**监控**：每周运行 `scripts/regime_watchdog.py` 检查 `ema_1200_position` 分布与 vol gate 实际触发率是否漂离 `config/monitoring/regime_watchdog_baseline.json`。

#### 实验配置归档

- `config_experiments/B_gate_only_chop_strategies/`（两条 vol gate 全部 disable）— **不上 live**
- `config_experiments/H_bull_conditional_vol_strategies/`（bull-conditional vol gate）— **已 promote 到 `config/strategies/tpc/archetypes/gate.yaml`**

### BPC / ME / SRB（研究仓，未改）

仍可为 chop + box（见各策略 `archetypes/regime.yaml`）。

### EMA1200 与 direction

- **TPC**：宏观带在 **`regime.yaml`**；**`direction.yaml`** 仅 MACD sign。
- **BPC/ME/SRB**：多数仍用 **`direction.yaml`** 的 `signal_match_position_band` + `ema_1200_position`。

## 特征依赖

| 列 | 来源 |
|----|------|
| `ema_1200_position` | `ema_1200_position_f`（TPC regime） |
| `tpc_semantic_chop`, `box_pos_*` | BPC/ME/SRB regime；TPC 已迁出 |

实盘 Feature Bus 通过 `extract_features_from_archetypes()` 扫描 `regime.yaml` 自动拉取 `box_structure_f` 节点，**无需**在 live 包内复制 `features.yaml`。

## Deploy 与验收

```bash
python scripts/deploy_config_to_live.py --diff -s tpc bpc me srb
python scripts/deploy_config_to_live.py --deploy -s tpc bpc me srb --yes
# 重启 quant-feature-bus、quant-trend-fattail
```

验收：

- `live/highcap/config/strategies/tpc/archetypes/regime.yaml` 存在
- live `gate.yaml` 含 `gate_tpc_semantic_chop_high`（chop）；regime **无** chop/box
- trend 日志：`N regime rules (empty=False, ...)`
- `posthoc_layer_effectiveness.py --strict-locked-features --strategies tpc` 退出码 0

## Tier-0 运维

- 校准：`python scripts/regime_threshold_calibrate.py --strategies tpc --dry-run`
- 漂移：`python scripts/regime_drift_monitor.py`
- 边际：`python scripts/regime_ablation_report.py --strategies tpc`

## Pre-deploy 合约

`config/strategies/*/research/pre_deploy_replay.yaml` 中 `contract_checks` 块；`fast_month` / `rolling_sim` 结束时若 pipeline YAML 含该块，会写 `contract_checks.json`（见 `scripts/pre_deploy_contract_checks.py`）。
