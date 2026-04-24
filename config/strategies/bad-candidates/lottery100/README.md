# Lottery100（L3 高杠杆彩票）— **bad-candidates 归档**

> **状态**：已从 `config/strategies/lottery100` 迁入本目录；**不参与** `constitution.yaml` 的 PCM 白名单与实盘编排。保留代码与 YAML 仅为复现实验；在固定风险 + ATR 止损口径下与顺势主策略（如 ME）边界重叠，原「极小本金全仓极限杠杆」设想亦不在此仓库的执行模型内实现。

本目录存放 **三层仓位架构里 L3「高杠杆彩票」** 实验的可版本化参数快照，与 ME/BPC/TPC 等 **L2 合约系统** 分开。策略目录名 **`lottery100`** 表示历史「百倍增益侧」研究族。

## 研究方法（杠杆容量统计 v1–v4）

完整方法说明（与 NN 训练管线的关系、脚本与报告索引）：

- **`docs/architecture/strategies/lottery100_research_methodology.md`**
- **v4 宏观门 YAML**：`leverage_capacity_v4.yaml`（容量研究）
- **事件回测主线**：`event_backtest.py`（交易地图 + 盈亏统计；主管线默认）
- **B+ 回测 YAML**：`backtest_bplus.yaml`（离线容量研究辅助）
- **与 BPC prod 形状对齐的研究壳**：`config/prod_train_pipeline_2h_lottery100.yaml`（日期 / `strategies` / `kpi_gates` / `event_backtest` 块）
- **Archetypes（与 BPC 同契约）**：`archetypes/*.yaml` — 供 **`event_backtest` / `GenericLiveStrategy`** 与 **`--adopt`** 使用；说明见 **`archetypes/README.md`**
- **串联执行（主）**：`python scripts/auto_research_pipeline.py --strategy lottery100 --config config/prod_train_pipeline_2h_lottery100.yaml`（识别 **`strategy_family: lottery100`**，Feature Store → event_backtest → 写入实验目录 `execution.yaml`；产物在 **`results/research_history/.../results/lottery100_event/`**）
- **串联执行（离线研究）**：`python scripts/run_lottery_research_bundle.py --config config/prod_train_pipeline_2h_lottery100.yaml` → **`results/lottery100_bundle/`**
- **实验笔记归档**：`docs/z实验_008_lottery/README.md`

### 是否需要接 `research_pipeline`？

已在 **`auto_research_pipeline.py`** 接入 **`strategy_family: lottery100`**（无 NN / SHAP）：与 ME/BPC **同一实验目录与 adopt 路径**。研究主评估为 **event_backtest**；B+ 仅做容量研究辅助，不参与主决策。

## 仓库里其它配置在哪

| 内容 | 路径 |
|------|------|
| ME / BPC / TPC / SRB / FBF 等 | `config/strategies/<name>/` |
| **Lottery100（本目录，归档）** | `config/strategies/bad-candidates/lottery100/` |
| LV（清算脆弱性，快进快出，≠ 彩票蓄力位） | `live/highcap/config/strategies/lv/` |
| 各 archetype 仓位预算（含 LV） | `config/constitution/constitution.yaml` → `risk_budget.archetypes` |

说明：

- **PCM 仲裁**：默认 **不** 启用 lottery100；若需与 bpc/me 等并列实验，须在 **`config/constitution/constitution.yaml`** 的 `resource_allocation.enabled_archetypes`（及优先级、`per_strategy_limits`）中自行加入 lottery100。
- `config/research_pipeline.yaml` 里写了 `lv: config: config/strategies/lv`，但 **根目录下并不存在 `config/strategies/lv`**；LV 的可运行配置目前在 **`live/highcap/config/strategies/lv/`**。若要统一路径，需要后续把 LV 迁入 `config/strategies/lv` 或改掉 pipeline 引用。

## 与本目录对应的分析脚本

- `scripts/analyze_leverage_capacity_v2.py` — 特征库 + MAE/MFE + OOS 树（无宏观牛熊门）
- `scripts/analyze_leverage_capacity_v3.py` — 在 v2 基础上增加 **BTC 周线趋势型 bull regime**，可选只保留 `bull_regime=True` 的样本再统计 / 建树

实盘接入时：`gate_draft.yaml` 里的阈值应与 v3/v4 报告对齐后固化，再由 live 策略加载（具体加载方式待接到 `generic_live_strategy` 或专用 runner）。

---

## 与统一研究管线（`research_pipeline.yaml`）的关系

- **YAML 分层 KPI**：在 prod yaml / `config/kpi_gates/lottery100.yaml` 定义各层语义与阈值；管线内 **`strategies.lottery100.kpi_gates`** 与之对齐（可只改数值）。
- **执行器**：`strategy_family: lottery100` 时走 **event_backtest + KPI 校验**，并把事件指标写入 **`archetypes/execution.yaml`**；向量回测（`logs_gated`）不适用，与其它非 NN 策略处理方式一致。
- **事件回测**：需完整 **`archetypes/`**（尤其 **`direction.yaml`**）；PCM 是否在模拟中启用取决于 constitution（见上表）。
