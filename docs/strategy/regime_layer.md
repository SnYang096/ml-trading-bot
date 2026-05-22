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

### TPC（20260522）

仅 **EMA1200 宏观死区**：`ema_1200_position >= 0.10` OR `<= -0.10`（等价 `|pos|>0.10`）。chop 在 **`gate.yaml`**（`gate_tpc_semantic_chop_high`）；box 已移除。

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
