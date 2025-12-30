## NN 多头基座 → Rule(3-action) → RL/BC e2e（长文档）

本文件收录了从 `README_CN.md` 抽离出来的“NN 多头 + 3-action + RL/BC”端到端命令解释与长段落说明，避免 README 过长。

建议先读：
- `docs/architecture/EXPERIMENT_LOOP_ARCHITECTURE.md`
- `docs/时序模型/架构：NN多头路径原语（Path Primitives）+Router解耦升级.md`

---

## 目标与产物

- **目标**：训练/推理 NN 多头（path primitives）→ 生成 `mode`（NO_TRADE/MEAN/TREND）→ 组装 RL/BC logs（含 `ret_mean/ret_trend`）→ 一键跑 shadow/counterfactual/fsm。
- **核心产物**：
  - `preds_*.parquet`：包含 `pred_dir_prob/pred_mfe_atr/pred_mae_atr/pred_t_to_mfe`
  - `mode_3action.parquet`：包含 `mode`（NO_TRADE/MEAN/TREND）
  - `logs_3action.parquet`：包含 `symbol,timestamp,mode,head_*,drawdown,ret_mean,ret_trend`
  - `results/rl/e2e/*`：shadow report / counterfactual report / fsm decision

---

## 最小可跑命令（示例）

### 0)（推荐）先构建 FeatureStore 宽表库

```bash
mlbot feature-store build \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --data-path data/parquet_data \
  --root feature_store \
  --layer AUTO \
  --no-docker
```

### 1) 训练 NN 多头

```bash
mlbot nnmultihead train \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --data-path data/parquet_data \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --feature-store-root feature_store \
  --feature-store-layer AUTO \
  --epochs 10 \
  --output-dir results/nnmultihead \
  --no-docker
```

### 2) NN 多头推理（多 symbol 输出目录）

```bash
mlbot nnmultihead predict \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --data-path data/parquet_data \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --feature-store-root feature_store \
  --feature-store-layer AUTO \
  --model results/nnmultihead/.../model.pt \
  --output results/nnmultihead/preds_multi \
  --no-docker
```

### 3) 纯规则 Router：从 heads 生成 3-action `mode`

```bash
mlbot rule mode-3action \
  --preds results/nnmultihead/preds_multi \
  --model results/nnmultihead/.../model.pt \
  --output results/rule/mode_3action.parquet \
  --no-docker
```

### 4) 组装 RL/BC logs（把 close 转成 ret_mean/ret_trend，并合并 heads + mode）

```bash
mlbot rl build-logs-3action \
  --preds results/nnmultihead/preds_multi \
  --mode results/rule/mode_3action.parquet \
  --model results/nnmultihead/.../model.pt \
  --data-path data/parquet_data \
  --timeframe 240T \
  --output results/rl/logs_3action.parquet \
  --no-docker
```

### 5) 一键跑 RL(e2e)：shadow → counterfactual → fsm

```bash
mlbot rl run-e2e-3action \
  --logs results/rl/logs_3action.parquet \
  --out results/rl/e2e \
  --no-docker
```


