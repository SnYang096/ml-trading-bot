# IC 剪枝命令重构计划

> **状态**：**M1–M4 已落地**（`mlbot research ic-prune` + `stat_kernels/ic_prune.py` + rd_loop subprocess）。M5 文档 / M6 清理进行中。  
> **关联**：[`研究工具重构计划_CN.md`](研究工具重构计划_CN.md) · [`遗留研究命令清理计划_CN.md`](遗留研究命令清理计划_CN.md) · [`R&D工具矩阵_CN.md`](R&D工具矩阵_CN.md) · [`树模型方法论演进与短期树重建指南_CN.md`](树模型方法论演进与短期树重建指南_CN.md) · [`config/experiments/README.md`](../../config/experiments/README.md)

---

## 0. 已实现（2026-05）

| 组件 | 路径 |
|------|------|
| 内核 | [`src/research/stat_kernels/ic_prune.py`](../../src/research/stat_kernels/ic_prune.py) |
| CLI | `mlbot research ic-prune` → [`scripts/research/ic_prune.py`](../../scripts/research/ic_prune.py) |
| rd_loop | `tree_steps: ic-prune` → subprocess `mlbot research ic-prune` |

**Target**：仅 `forward_rr`（与训练 label 对齐；已删除 ATR-forward fallback）。

**IC 符号（方案 B）**：

- json/md **始终**记录 per-node `ic_sign`（`+`/`-`）
- `invert_features` **默认关闭**（`--invert-mode none`）：对纯 LightGBM 无 `monotone_constraints` 时乘 -1 是 no-op；负 IC 因子可直接进树
- `--invert-mode auto`：给线性/NN 消费者写 `invert_features`
- `--emit-monotone-constraints`：产出 review-only 单调约束提示（对树真正有用的符号产物）

---

## 1. 问题

树通道（`fast_scalp` / `short_term_swing`）的 **IC 剪枝**（holdout 上扫全列 → `best_lag` 阈值 → 聚合为 `_f` 节点 → 写 `features.yaml`）目前走 **`scripts/research/fast_scalp_ic_prune.py`**，与文档里的官方研究面 **`mlbot research ic`** / Makefile **`ts-factor-eval`** 是 **三条平行线**：

| 能力 | 官方 `research ic` | 老 `factor-eval` | 当前 `fast_scalp_ic_prune` |
|------|---------------------|-------------------|----------------------------|
| 入口 | `mlbot research ic` | `make ts-factor-eval` | `rd_loop tree_steps: ic-prune` |
| 特征来源 | 手填 `--features` | 策略 `features.yaml` 或 `--factors` | parquet **全数值列** |
| 标签 / target | parquet 列 `forward_rr` + horizon shift | 策略 label 的 `target_col`，再 `shift(-lag)` | **现场重算** `(close[t+h]-close)/atr` |
| holdout 窗口 | 无（全 parquet 或 filter mask） | train 日期范围（非 holdout 语义） | **`holdout_start/end` 硬切** |
| lag 过滤 | 仅输出 decay 表 | `--filter-by-best-lag` + `--target-lag` | **`max_lag` + `min_ic`** |
| 节点聚合 | 无 | export 到 Pool-B（列级 + invert） | **`feature_dependencies.yaml` → `_f` 节点** |
| top-N / 写回 | 无 | `--export-yaml`（Pool-B 路径） | **`top_n_nodes` + 直接写策略 `features.yaml`** |
| 多币 pooled | `--parquet` 支持 | 默认 **单 symbol** | **pooled parquet**（Phase 1/2 实测） |

**rd_loop 因此出现「双 IC 人格」**：

- `research_scans` / `quick_layer_scans` 里 `mode: ic-decay` → 调 **`mlbot research ic`**（诊断、手选特征）
- `tree_steps` 里 `mode: ic-prune` → 调 **`fast_scalp_ic_prune.run_ic_prune`**（真正驱动 top-35 训练）

Phase 1/2 结论依赖后者；前者在 yaml 里更多是占位/对照，**不能替代 ic-prune**。

---

## 2. 命令清单（Makefile + CLI + 脚本）

### 2.1 Makefile（研究向）

| Make 目标 | 底层 | 用途 | 与树 IC 剪枝关系 |
|-----------|------|------|------------------|
| **`ts-factor-eval`** | `mlbot analyze factor-eval` | 单资产因子 IC / 分位 win-rate / **IC decay HTML**；可选 `--filter-by-best-lag`、`--remove-correlated`、`--export-yaml` | **树方法论文档 Step 0 推荐路径**（`树模型方法论演进` §2.2.1）；产出 Pool-B yaml，**非 holdout 全列剪枝** |
| **`ts-feature-eval`** | `python -m time_series_model.pipeline.training.feature_type_evaluator` | 按 feature-type 算 IC → `top_factors.json` | 更老的 Pool-B 入口；`feature-eval` 已 alias 到此 |
| **`ts-timeframe-forward-report`** | timeframe vs forward-bar 相关 | 选 horizon 的辅助报告 | 不剪枝 |
| **`ts-strategy-feature-compare`** | 多 `features.yaml` ablation | 训练对比 | 剪枝**之后**的验收 |

**`ts-factor-eval` 常用变量**（`Makefile` L391–444）：

```bash
TS_FACTOR_STRATEGY=config/strategies/tree_strategies/fast_scalp
TS_FACTOR_TIMEFRAME=120T
TS_FACTOR_IC_DECAY_LAGS=1,3,5,8,12,20
TS_FACTOR_FILTER_BY_BEST_LAG=1
TS_FACTOR_TARGET_LAG=5          # 可选；不设则 infer 自 label max_holding_bars
TS_FACTOR_LAG_TOLERANCE=5
TS_FACTOR_REMOVE_CORRELATED=1
TS_FACTOR_CORRELATION_THRESHOLD=0.9
# 注意：无 TS_FACTOR_HOLDOUT_*；日期是 TS_FACTOR_START / TS_FACTOR_END
```

### 2.2 `mlbot` CLI

| 命令 | 实现 | 树 IC 角色 |
|------|------|------------|
| **`mlbot research ic`** | `scripts/research/ic.py` → `quick_layer_scan.mode_ic_decay` + `src/research/stat_kernels/ic.py` | **诊断**：指定列 + decay 表；**不能** auto-prune / 写 features |
| **`mlbot analyze factor-eval`** | `src/time_series_model/diagnostics/factor_ts_eval.py` | **老官方树 IC**（docstring 已注明 decay 内核迁至 `research ic`）；重算特征、HTML、Pool-B export |
| **`mlbot analyze feature-eval`** | `feature_type_evaluator` | 同 `ts-feature-eval` |
| **`mlbot search tree`** | `scripts/run_poolb_semantic_search.py` → factor-eval + **feature-group-search** + writeback | **长周期树一键流**；方法论已明确 fast_scalp **不走 FGS** |
| **`mlbot diagnose feature-group-search`** | greedy/halving/beam 搜 semantic groups | Pool-B 之后的历史路径；与 ic-prune **目标不同** |

### 2.3 脚本 / rd_loop

| 路径 | 调用方 | 作用 |
|------|--------|------|
| **`scripts/research/fast_scalp_ic_prune.py`** | `rd_loop tree_steps ic-prune`、CLI `__main__` | **当前 fast_scalp 剪枝唯一真源** |
| **`scripts/research/ic.py`** | `mlbot research ic`、`rd_loop ic-decay` | 诊断 decay |
| **`scripts/quick_layer_scan.py`** | 被 `ic.py` import | 遗留壳；Phase 2 清理计划待迁 |
| **`scripts/rd_loop.py`** | `tree_steps` / `research_scans` | 编排；ic-prune **未**走 mlbot |
| **`scripts/run_poolb_semantic_search.py`** | `mlbot search tree` | factor-eval 包装 + FGS 阶段 |

### 2.4 实验 yaml 锚点

- Phase 1：`config/experiments/20260529_fast_scalp/rd_loop_fast_scalp_ic_plateau.yaml` → `tree_steps[ic-prune]`
- Phase 2：`config/experiments/20260530_fast_scalp_alts_majors/rd_loop_fast_scalp_alts_majors.yaml`
- 产物：`results/rd_loop/fast_scalp_ic_plateau/ic_prune_h5/`

---

## 3. 三条 IC 路径的行为差异（重构必须对齐）

### 3.1 Target 计算（**最高风险**）

```
research ic:     rank_ic(feat, forward_rr shifted by h)     # parquet 已有 label 列
factor-eval:     rank_ic(feat, target_col.shift(-h))       # 策略 label 配置
ic_prune (现):   rank_ic(feat, (close[t+h]-close)/atr)    # 现场 OHLC merge + ATR
```

fast_scalp 训练标签是 **`forward_rr`（barrier/horizon 语义）**；ic_prune 用的是 **纯价格 forward/ATR**。Phase 1/2 能 work，但和 `research ic` / 训练 label **不可直接对拍**。重构时必须先 **选定单一 target 内核**（建议：`src/research/stat_kernels/labels.py` 或 label 列 + 与 `research ic` 相同的 per-symbol shift）。

### 3.2 特征 universe

| 路径 | Universe |
|------|----------|
| factor-eval | `requested_features` 展开或 `--factors` |
| research ic | CLI 显式列表 |
| ic_prune | parquet 全部数值列 − skip set |

树剪枝需要 **「store 全量列 → holdout IC → 节点级 requested_features」**；factor-eval 的 universe 偏 **「已有 yaml 内因子」**，两者互补但不等价。

### 3.3 输出形态

| 路径 | 输出 |
|------|------|
| factor-eval | HTML + `features_pool_b.yaml`（列 + invert_features） |
| research ic | `ic_decay.md` / `ic_decay.json` |
| ic_prune | `ic_prune_holdout.json/md` + **策略 `features.yaml`（仅 requested_features 块）** |

树训练要的是 **节点列表**（`_f` 后缀），不是 Pool-B 的 invert 列集合；export 格式应对齐 **`feature_pipeline.requested_features`**，不是 Pool-B 路径惯例。

---

## 4. 目标架构

### 4.1 动词拆分（推荐）

保持 **`research ic`** = 诊断；新增 **`research ic-prune`** = 树通道剪枝 + 可选写回。

```
mlbot research ic          # 已有：手选 features，decay 报告
mlbot research ic-prune    # 新增：holdout 全列/池化 IC → 节点 prune → yaml
```

**理由**：`研究工具重构计划` 里 `ic` 已定义为 Q2 decay；把 prune / top-N / writeback 塞进同一 verb 会撑爆 CLI 且与 TPC gate 扫描参数正交。

### 4.2 内核落位

```
src/research/stat_kernels/ic_prune.py   # 新：screen_features, column→node, trim, export
scripts/research/ic_prune.py            # 薄 CLI（mirrors ic.py 结构）
scripts/research/fast_scalp_ic_prune.py # 过渡期：import run_ic_prune from kernel；最后删或仅 re-export
```

**与现有模块关系**：

- `rank_ic` / `ic_decay_rows`：继续用 `src/research/stat_kernels/ic.py`
- target：复用 `research ic` 的 `--target` + horizon shift（修齐 multi-symbol）
- 节点映射：读 `config/feature_dependencies.yaml`（现 logic 保留）
- OHLC merge：仅在 target=`forward_price_atr` 或 parquet 缺 OHLC 时启用（逐步废弃专用路径）

### 4.3 `ic-prune` CLI 草案

```bash
mlbot research ic-prune --no-docker \
  --strategy tree_strategies/fast_scalp \
  --parquet results/train_final/.../features_labeled.parquet \
  --holdout-start 2025-10-01 --holdout-end 2026-04-01 \
  --horizons 1,2,3,4,5 \
  --max-lag 5 --min-ic 0.02 --min-n 200 \
  --top-n-nodes 35 \
  --always-include atr_f \
  --intersect-features-yaml config/strategies/tree_strategies/fast_scalp/features.yaml \
  --write-features-yaml config/strategies/tree_strategies/fast_scalp/features_ic_top35.yaml \
  --output results/rd_loop/fast_scalp_ic_plateau/ic_prune_h5
```

| 参数 | 对应现 `run_ic_prune` | 说明 |
|------|----------------------|------|
| `--holdout-start/end` | ✓ | 树验收窗口 |
| `--max-lag` / `--min-ic` | ✓ | 替代 factor-eval 的 filter-by-best-lag（语义：best_lag ≤ max_lag） |
| `--top-n-nodes` | ✓ | Step 1 冻结 30–50 节点 |
| `--intersect-features-yaml` | ✓ | 与现有 pool 求交 |
| `--write-features-yaml` | ✓ | 写回；默认 dry（只 json/md） |
| `--target` | **新增对齐** | 默认 `forward_rr`；可选显式 override |
| `--scan all\|yaml\|list` | **新增** | `all`=现行为；`yaml`=factor-eval 式 universe |

### 4.4 rd_loop 收敛

```yaml
# tree_steps — 目标态
- mode: ic-prune
  cmd: mlbot research ic-prune   # subprocess，与 ic-decay 对称
  holdout_start: ...
  # 或保留 dict，由 rd_loop 拼 CLI（与 research_scans 一致）
```

`research_scans` 里 **`ic-decay` 保留**为人工挑特征前的诊断；**训练流水线只认 `ic-prune`**。

### 4.5 Makefile

| 现状 | 目标 |
|------|------|
| `ts-factor-eval` | **保留**给「单币 + 策略 yaml + HTML + Pool-B」探索；文档注明 **不等于** holdout ic-prune |
| （无） | 可选 **`ts-ic-prune`**：薄包装 `mlbot research ic-prune`，变量 `TS_IC_PRUNE_*`  mirror `TS_FACTOR_*` 里与 holdout 相关的部分 |

不建议把 `ts-factor-eval` 直接改成 ic-prune：factor-eval 仍服务 sr_reversal / Pool-B / `search tree`。

---

## 5. 迁移表

| 阶段 | 动作 | 验收 |
|------|------|------|
| **M0（现状）** | `fast_scalp_ic_prune` + `tree_steps` | Phase 1/2 DECISION 已归档 |
| **M1 内核抽取** | `run_ic_prune` → `src/research/stat_kernels/ic_prune.py`；sidecar 变 wrapper | 单测：`ic_prune` json 与 `ic_prune_h5/` 对拍 |
| **M2 target 对齐** | 默认 `--target forward_rr`；OHLC/ATR 路径仅 fallback | 同一 parquet：`research ic` 与 `ic-prune` 同 feature 的 best_lag 一致 |
| **M3 CLI** | `scripts/research/ic_prune.py` + `mlbot research ic-prune` | 手工 CLI 复现 Phase 1 top-35 |
| **M4 rd_loop** | `tree_steps ic-prune` → subprocess mlbot | `test_rd_loop.py` patch `research ic-prune` 而非 sidecar |
| **M5 文档** | 更新 `R&D工具矩阵`、`树模型方法论` Step 0、`config/experiments/README` | 树 Step 0 主命令改为 ic-prune；factor-eval 降为「单币探针」 |
| **M6 清理** | 删 `fast_scalp_ic_prune.py` 或留 DEPRECATED 一行 re-export | grep 无直接 import |

**与 [`遗留研究命令清理计划`](遗留研究命令清理计划_CN.md) 的关系**：

- Phase 2（迁 `quick_layer_scan`）与 M2 **可并行**，但 ic-prune **不应**再 import `quick_layer_scan`
- `analyze factor-eval`：**不删**；在 docstring + 树文档中降级为 Pool-B / 单币 HTML
- `mlbot search tree`：树 short-term 文档继续 **不推荐**；与 ic-prune 无合并计划

---

## 6. 命令对照速查（重构前）

```bash
# A. 诊断 decay（官方 research 面）
mlbot research ic --strategy fast_scalp --parquet "$PARQ" \
  --features pulse_z,macd_atr,bb_width_normalized_pct \
  --horizons 1,3,5,10,20 --target forward_rr

# B. 老 Makefile / 单币 Pool-B（方法论 Step 0 文档示例）
make ts-factor-eval \
  TS_FACTOR_STRATEGY=config/strategies/tree_strategies/bpc \
  TS_FACTOR_TIMEFRAME=120T \
  TS_FACTOR_IC_DECAY_LAGS=1,3,5,8,12,20 \
  TS_FACTOR_FILTER_BY_BEST_LAG=1 \
  TS_FACTOR_REMOVE_CORRELATED=1

# C. 当前 fast_scalp 剪枝（sidecar）
PYTHONPATH=src:scripts python scripts/research/fast_scalp_ic_prune.py \
  --parquet "$PARQ" \
  --output-dir results/rd_loop/fast_scalp_ic_plateau/ic_prune_h5 \
  --holdout-start 2025-10-01 --holdout-end 2026-04-01 \
  --max-lag 5 --min-ic 0.02 --top-n-nodes 35

# D. rd_loop 编排（ic-prune 在 tree_steps 内）
PYTHONPATH=src:scripts python scripts/rd_loop.py \
  --hypothesis-yaml config/experiments/20260529_fast_scalp/rd_loop_fast_scalp_ic_plateau.yaml

# E. 历史一键树（不推荐 fast_scalp）
mlbot search tree --strategies fast_scalp --symbol BTCUSDT ...
```

**重构后目标**：**C → `mlbot research ic-prune`（B 仍独立；A 不变；D 调 C 的新 CLI；E 不变）**

---

## 7. 待决事项

1. **Target 默认值**：是否在 M2 切到 `forward_rr` 后重跑 Phase 1 对拍（可能轻微改 node 列表）？
2. **写回策略**：`--write-features-yaml` 覆盖 live yaml vs 写 `_ic_top35.yaml` 再人工 promote（Phase 1 是前者）？
3. **factor-eval 的 `--remove-correlated`**：是否纳入 ic-prune（列级）还是在节点聚合后再做（现未做）？
4. **Makefile 是否新增 `ts-ic-prune`**：取决于是否仍常用 Docker make 入口跑 fast_scalp。

---

## 8. 建议实施顺序（1–2 天量级）

1. M1 + M2（内核 + target 对齐 + 单测）  
2. M3 CLI（可手工跑通）  
3. M4 rd_loop + 实验 yaml 注释更新  
4. M5 文档；M6 视 M4 稳定后再删 sidecar  

完成 M4 后，[`config/experiments/README.md`](../../config/experiments/README.md) 中 `ic-prune` 一行应从 sidecar 脚本改为 **`mlbot research ic-prune`**。
