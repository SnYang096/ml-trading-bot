# 管线与文档索引（INDEX）

> 用途：把「研究 / 快管线模拟 / 上线验收 / 发布与实盘」对应到哪份文档、缺什么自己补哪里，一眼可查。  
> 维护：改 `rolling.windows.calibration_months`、发布路径或阶段名时，同步改本节 **配置真值** 与 **发布 checklist**。

---

## 1. 快速导航

| 我想做的事 | 优先读/用 | 说明 |
|------------|-----------|------|
| 数据、Feature Store、全量研究 `pipeline run` | [A快速启动命令.md](./A快速启动命令.md) | 多策略、SHAP、完整研究链 |
| 2024 趋势窗 + BPC turbo 快检 / `rolling_sim` | [实施文档_01_2024牛市_5x趋势骑乘.md](./实施文档_01_2024牛市_5x趋势骑乘.md) | 与 `prod_train_pipeline_2h_turbo_2024bull_thresholds_only_bpc_only.yaml` 最贴近 |
| 2025 震荡 + FER、strict/turbo 对照 | [实施文档_02_2025震荡_控风险止盈_FER接入.md](./实施文档_02_2025震荡_控风险止盈_FER接入.md) | 配置示例文件名可能与当前 BPC-only 不同，复制思路即可 |
| 高原准则、跨环境验收、回滚顺序 | [实施文档_03_特征高原与执行参数高原_上线准则.md](./实施文档_03_特征高原与执行参数高原_上线准则.md) | 「什么算能上线」，不是逐步运维手册 |
| **发布物从哪来、是否进仓库 config** | **本文 §3～§4** | 原四份文档未单独写死 |
| **实盘监控、纸/实一致性** | **本文 §5** | 待你自建 runbook |

---

## 2. 四条管线在说什么（避免混用）

### 2.1 研究管线（慢、全量）

- **目标**：新特征、新结构、SHAP、树训练、大范围验证。  
- **入口**：[A快速启动命令.md](./A快速启动命令.md) 第四节「Research Pipeline」。  
- **产出**：模型、特征层、候选规则；**不等于**下月直接用来实盘的 `archetypes` 定稿。

### 2.2 快管线 / 模拟未来（turbo、固定特征）

- **目标**：在**特征集冻结**前提下，按月（或按窗）重标定 prefilter / gate / entry / direction 等阈值，事件回测看「类似实盘滚动」的表现。  
- **配置示例**：`config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_bpc_only.yaml`（`rolling.mode: turbo_fixed_features`）。  
- **命令**：`mlbot pipeline run --all --config <yaml> --stage fast_month --month YYYY-MM` 或 `--stage rolling_sim`。  
- **文档**：[实施文档_01](./实施文档_01_2024牛市_5x趋势骑乘.md)（趋势快检）、[实施文档_02](./实施文档_02_2025震荡_控风险止盈_FER接入.md)（震荡与 FER）。  
- **注意**：`rolling_sim` 下每月的 **`strategies_calibrated` 在结果目录**，不是默认写回仓库 `config/strategies/...`（详见 01 文档注释）。模拟赚钱 ≠ 仓库里的 yaml 已自动更新，**发布要按 §3 操作**。

### 2.3 上线前总验收（高原 + 跨年份）

- **目标**：执行高原、slot、慢变量等与 **2024 + 2025** 跨环境稳定性；**不依赖单月爆发**。  
- **文档**：[实施文档_03](./实施文档_03_特征高原与执行参数高原_上线准则.md)。  
- **与快管线关系**：03 里的 `rolling_sim` 总验收，可用 **strict / turbo 各一份配置** 对照；BPC-only 日常快检仍可用 01 的配置。

### 2.4 实盘运维（代码外流程）

- **目标**：部署、监控、回滚、与回测路径一致。  
- **文档**：**本文 §5** + 你方内部 runbook（本仓库 INDEX 只列提纲）。

---

## 3. 配置真值（文档与 YAML 冲突时以谁为准）

- **`rolling.windows.calibration_months`**：以当前使用的 **prod yaml** 为准。  
  - 例如 BPC-only turbo：`config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_bpc_only.yaml` 内现为 **6**（若日后改动，改 YAML 后在此 INDEX 更新一句即可）。  
- **`dates.holdout_months` / `validation_months`**：主要服务 **整段 holdout 切块与 `rolling_sim` 月份遍历**；**`fast_month` 单月**的标定窗由 **`rolling.windows.calibration_months` + `--month`** 决定，与「15 个月 OOS」横幅不是同一套语义（详见此前讨论）。  
- **实施文档 02 / 03** 若仍写「月度 3 个月校准」，视为**叙述模板**；**数值以仓库 yaml 为准**。

---

## 4. 发布物与上线 Checklist（补原四份文档缺口）

### 4.1 两类「真相源」

| 路径含义 | 典型位置 | 用途 |
|----------|----------|------|
| **仓库冻结配置** | `config/strategies/<strategy>/archetypes/*.yaml` | Git 版本化、实盘/CI 默认可指向 |
| **单次滚动实验产出** | `results/.../turbo-rolling-sim/_rolling_sim/<run_id>/fast_month_*/strategies_calibrated/` | 该月/该次模拟的标定结果 |

**原则**：快管线 **`rolling_sim` / `fast_month` 默认不把 archetypes 写进仓库**（除非你有单独 adopt/拷贝流程）。上线前必须明确：**采纳哪一次 run 的哪几个文件**。

### 4.2 建议发布 Checklist（按需勾选）

1. **特征与数据**  
   - Feature Store 层与训练/回测一致；若刚跑完老管线全量出特征，记录 **layer id / manifest**。  
2. **采纳范围**  
   - 列出要合并的文件：`gate.yaml`、`entry_filters.yaml`、`prefilter.yaml`、`direction` 相关、`execution.yaml`（若本配置关闭了 `execution_opt`，则可能仍以仓库旧版为准）。  
3. **从实验目录 → 仓库**  
   - 人工 `cp` / PR：从选定 `strategies_calibrated/.../archetypes/` 拷到 `config/strategies/bpc/...`（或你们约定的 mono-branch）。  
4. **Diff 审查**  
   - `locked: false` 的 gate 是否在 promote 中被 **移除**（而非仅 `disabled`）；与预期是否一致。  
   - entry：`promote_never_disable` 条目的 `enabled` 与阈值是否符合 OR 宽度预期。  
5. **回归**  
   - 对采纳后的仓库 config 再跑 **一次** `event_backtest` 或 `fast_month` 冒烟（可选但推荐）。  
6. **Git**  
   - tag / 发布说明中写清 **run_id** 与 **日期范围**（便于回滚）。

### 4.3 回滚顺序（与 03 对齐）

与 [实施文档_03 §6](./实施文档_03_特征高原与执行参数高原_上线准则.md#6-回滚顺序) 一致时：**execution → slot case → 慢变量结构**；若本次仅改阈值，通常回滚 **gate / entry / prefilter** 的上一 Git 提交即可。

---

## 5. 实盘运维提纲（自建 runbook 时照抄扩展）

以下内容 **INDEX 只列标题**，细节在运维文档或监控系统中维护：

- **监控**：订单拒单率、滑点、与 `gate_decision` / 信号一致性抽样对比。  
- **数据**：实盘特征是否落后、Feature Store 与 live 计算路径是否同版本。  
- **Regime**：月度重跑与主观牛熊/VWAP1200 切换的 **触发关系**（何时加密跑、何时只观察）。  
- **统计**：重大决策参考 **连续多个月** 的 `pcm_candidates` / ledger，避免单月 `n_trades` 过小下的误判（与 03「不依赖单月爆发」一致）。

---

## 6. 推荐阅读顺序（新人）

1. 本文 **§2 + §3**（管线边界 + 以谁为准）。  
2. [实施文档_03](./实施文档_03_特征高原与执行参数高原_上线准则.md)（上线哲学与验收）。  
3. 按场景二选一：[实施文档_01](./实施文档_01_2024牛市_5x趋势骑乘.md) 或 [实施文档_02](./实施文档_02_2025震荡_控风险止盈_FER接入.md)。  
4. [A快速启动命令.md](./A快速启动命令.md)（需要全量研究时再深入）。  
5. 发布前过 **本文 §4 Checklist**。

---

## 7. 文件清单（本目录相关）

| 文件 | 角色 |
|------|------|
| [INDEX_管线与文档导航.md](./INDEX_管线与文档导航.md) | 本索引 + 发布/运维补全 |
| [A快速启动命令.md](./A快速启动命令.md) | 研究管线主手册 |
| [实施文档_01_2024牛市_5x趋势骑乘.md](./实施文档_01_2024牛市_5x趋势骑乘.md) | 趋势窗 + BPC turbo 快检 |
| [实施文档_02_2025震荡_控风险止盈_FER接入.md](./实施文档_02_2025震荡_控风险止盈_FER接入.md) | 震荡 + FER |
| [实施文档_03_特征高原与执行参数高原_上线准则.md](./实施文档_03_特征高原与执行参数高原_上线准则.md) | 高原与上线准则 |
