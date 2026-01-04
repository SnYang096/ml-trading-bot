## nnmultihead 常用命令速查（CLI）

这份文档的目标：把 **nn 多头（Path Primitives）** 相关命令集中到一个地方，方便日常训练/评估/产物定位。

### 约定：核心配置目录

默认配置示例：
- `config/nnmultihead/path_primitives_4h_80h_min/`

### 1) 训练：`mlbot nnmultihead train`

训练并自动生成训练报告（`report.html` + `metrics_summary.md`）。

```bash
mlbot nnmultihead train \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --features-store-root feature_store \
  --features-store-layer features_83f12ecc5e \
  --output-dir results/nnmultihead/my_run_name
```

产物（典型）：
- `results/nnmultihead/<run>/model.pt`
- `results/nnmultihead/<run>/meta.json`
- `results/nnmultihead/<run>/metrics.json`
- `results/nnmultihead/<run>/report.html`
- `results/nnmultihead/<run>/metrics_summary.md`

### 2) 预测：`mlbot nnmultihead predict`

对指定数据段产出 primitives 预测（parquet/csv 等）。

```bash
mlbot nnmultihead predict \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2025-07-01 \
  --end-date 2025-12-31 \
  --features-store-root feature_store \
  --features-store-layer features_83f12ecc5e \
  --model results/nnmultihead/my_run_name/model.pt \
  --output-dir results/nnmultihead/my_run_name/preds_2025H2
```

### 3) 评估：`mlbot nnmultihead eval`

对某段数据进行评估并生成报告（用于 OOS 复核、回归测试）。

```bash
mlbot nnmultihead eval \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2025-07-01 \
  --end-date 2025-12-31 \
  --data-path data/parquet_data \
  --model results/nnmultihead/my_run_name/model.pt \
  --output-dir results/nnmultihead/my_run_name/eval_2025H2
```

### 4) 重新渲染报告：`mlbot nnmultihead render-report`

适用于：你更新了报告模板/summary 逻辑，但不想重训。

```bash
mlbot nnmultihead render-report \
  --run-dir results/nnmultihead/my_run_name
```

### 5) primitives 因子筛选（Pool B）：`mlbot nnmultihead factor-eval`

对候选特征做 “primitives 目标” 的单因子稳定性评估，并导出 Pool B YAML（后续用于快速收敛特征集）。

```bash
mlbot nnmultihead factor-eval --no-docker \
  --config-dir config/nnmultihead/path_primitives_4h_80h_min \
  --candidates-yaml config/strategies/sr_reversal_rr_reg_long/features_all.yaml \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --features-store-root feature_store \
  --features-store-layer features_83f12ecc5e \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --min-samples-per-group 120
```

默认输出目录（约定）：
- `results/pools/path_primitives_4h_80h_min/pool_b_primitives/`
  - `primitives_factor_eval_metrics.csv`
  - `primitives_factor_eval_summary.json`
  - `features_pool_b_primitives.yaml`

下一步（推荐）：参考 `docs/strategies/NNMULTIHEAD_FEATURE_SEARCH_PLAYBOOK_CN.md` 把 Pool B 变成可迭代的 `features.yaml`（required/optional_blocks）并做 multi-run 对比。


