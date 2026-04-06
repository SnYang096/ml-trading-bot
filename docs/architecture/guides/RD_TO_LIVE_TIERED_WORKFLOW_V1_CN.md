# 研发→上线分层工作流（V1）：Tier（特征）× Universe（symbols）× TaskSpec（合同）

本文件是“命令导向”的最短路径：解决你现在的核心痛点——**特征太多算不完、symbols 太多跑不通、研究结果无法平滑走到上线**。

建议先读：
- `docs/architecture/guides/FEATURE_COMPLEXITY_LAYERS_CN.md`（理念与分层定义）
- `docs/archive/guides/tree/DEPLOYMENT_MVP_WORKFLOW_CN.md`（上线闭环与验收范式）
- `docs/architecture/ARCH_UPGRADE_TASKSPEC_CONSTITUTION_V1_CN.md`（TaskSpec+Constitution+PCM 总体架构）

---

## 0) 一句话原则

- **先跑通主链路**（训练→预测→Router→E2E/报告），再解锁重特征/大 universe。  
- **每次只解锁一个自由度**（一个 Tier 或一个 optional block 或一个 universe 扩容），否则不可归因。  

---

## 1) Universe 分层（U1→U3）：先缩 symbols 再谈 heavy features

- **U1 Core2（必须先跑通）**：`BTCUSDT` + 1 个代表（例如 `ETHUSDT`）
- **U2 HighCap6（收敛特征）**：BTC/ETH/BNB/SOL/XRP/ADA（或你定义的 HighCap6）
- **U3 Top9/Top10（最后扩）**：在 U2 走通后再逐步加

---

## 2) 特征分层（Tier0→Tier3）：先便宜后昂贵

约定（强烈建议）：  
- Tier0/1 放 `required`（主链路必需、可稳定复现）  
- Tier2/3 放 `optional_blocks`（按阶段解锁、可快速回退）  

你可以直接复用：`docs/architecture/guides/FEATURE_COMPLEXITY_LAYERS_CN.md` 的 Tier 定义。

---

## 3) nnmultihead（3-action）主链路：从 U1/Tier0 开始

> 下面命令以 `--no-docker` 为例。  
> 你可以用 `TaskSpec v1` 作为“运行合同”，把 symbols/timeframe/windows/layer 名称都固定下来（便于复盘）。

### 3.1 FeatureStore（先 Tier0/1）

```bash
mlbot nnmultihead build-feature-store --no-docker \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --data-path data/parquet_data \
  --feature-store-root feature_store \
  --layer nnmh_u1_tier01_240T_v1 \
  --start-date 2023-01-01 --end-date 2024-12-31 \
  --warmup-bars 512 --warmup-months 1 \
  --feature-monthly-workers 2 --feature-monthly-backend process
```

### 3.2 Train（A-layer head eval 产物）

```bash
mlbot nnmultihead train --no-docker \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2023-01-01 --end-date 2024-12-31 \
  --features-store-root feature_store \
  --features-store-layer nnmh_u1_tier01_240T_v1
```

### 3.3 Predict → Router → Build-logs → E2E（系统层验收）

推荐直接用一键：

```bash
mlbot nnmultihead pipeline-3action-e2e --no-docker \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --start-date 2025-05-01 --end-date 2025-10-31 \
  --features-store-root feature_store \
  --features-store-layer nnmh_u1_tier01_240T_v1 \
  --router-mfe-min 0.1259 \
  --router-eff-min 1.0434 \
  --router-dir-conf-trend-min 0.0642
```

---

## 4) 解锁路径（标准迭代节奏）

### 4.1 U1/Tier0→Tier1
- 只引入 Tier1（压缩/SR质量/语义评分），不要同时扩 symbols。

### 4.2 U1→U2（HighCap6）
- 在 U1 稳定后扩 universe。\n
### 4.3 解锁 Tier2（DTW/Spectrum/WPT/Hilbert）
- 一次只解锁一个 block。\n
### 4.4 解锁 Tier3（orderflow/ticks）
- 建议启用：\n
```bash
--feature-monthly-workers 4 --feature-monthly-backend process --fast-features
```

---

## 5) Tree（策略侧）如何“正确参与”：只做 Gate/Detector（不直接 alpha->trade）

- Tree 的位置：**Router 下游**，做 veto/throttle/permission，并支持规则导出（可审计）  
- 具体思想见：`docs/architecture/树模型在多头模型下游的角色.md`

---

## 6) 你应该如何用 TaskSpec/Constitution 把流程“锁死”

- 用 `config/tasks/task_spec.yaml` 固定：窗口、universe、feature tier、FeatureStore layer、router 阈值来源、执行假设\n
- 用 `config/constitution/constitution.yaml` 固定：kill-switch、slots、加仓合法方式、跃迁条款\n

任何一次训练/调参/上线，都必须把 TaskSpec ID 写入产物目录与报告。\n

