# 杠杆容量 v3 — BTC 周线 bull regime 门 + 配置落点

## 1. 彩票（L3）配置在哪里？

**原先不在 `config/strategies/` 里。** 文档里的「三层仓位 / L3 高倍彩票」是架构说明，没有对应 ME/BPC 那种完整策略目录。

本次新增（**草案，可版本化**）：

| 路径 | 内容 |
|------|------|
| `config/strategies/bad-candidates/lottery100/README.md` | L3 与 LV、流水线路径对照 |
| `config/strategies/bad-candidates/lottery100/gate_draft.yaml` | 宏观门（BTC 周线）+ 特征门阈值（与 v2 §6 对齐） |

**不要和 LV 混淆**：LV（清算脆弱性 / 快进快出）的可运行 YAML 在 **`live/highcap/config/strategies/lv/`**，不是彩票蓄力位。

**陷阱**：`config/research_pipeline.yaml` 写了 `lv: config: config/strategies/lv`，但仓库根目录 **没有** `config/strategies/lv/`；若以 pipeline 为准，需把路径改成 `live/highcap/config/strategies/lv` 或把 LV 迁入 `config/strategies/lv`。

---

## 2. v3 相对 v2 改了什么？

脚本：`scripts/analyze_leverage_capacity_v3.py`

- 用 **BTC 周线收盘 vs 周线 EMA(50)**（`W-FRI` last），**滞后一周**再展开到 120T bar（避免偷看当周收盘）。
- 给每根 bar 打上 `bull_regime`；ETH 与 BTC **共用同一锚**（BTC-led regime）。
- `--bull-only`：只在 `bull_regime=True` 的样本上做 bucket / subset lift / feature lift / 决策树（文件名带 `_bull_only`）。

默认全样本约 **79.2%** bar 落在 bull_regime（2022-08 以来周线多在 EMA50 上方），门控偏宽；若要更「下一牛可部署」的严门，可在 v4 加 **6M 收益、或周线 EMA 斜率** 等（`gate_draft.yaml` 已预留说明位）。

产物目录示例：

- `reports/leverage_capacity_v3_all/` — 不打掉非牛样本，但 parquet 带 `bull_regime` 列  
- `reports/leverage_capacity_v3_bull_only/` — 仅牛样本  

---

## 3. 关键数值（H=120 long，`--bull-only`，train 2022-08~2023-09 → test 2023-10~2024-03）

与 v2 **同一段 test**、同一决策树设定，差异来自 **训练集只含周线牛样本**：

| 指标 | v2（全样本 train） | v3 bull_only train |
|------|---------------------|---------------------|
| Test top 1% prec（long） | 16.3% | **32.6%** |
| Test top 1% lift（long） | 7.57× | **15.14×** |

即：**在声明为「周线牛」的子样本上学规则，牛市 test 窗口里 top 1% 命中率约翻倍**。代价是 threshold 模型在「全样本 train」下的可比性变差——部署时应固定 **macro 门 + 特征门** 再校准。

**OOS（2024-04~2026-02）long**：v2 top 1% 约 prec 2.5%、lift 0.69；v3 bull_only top 1% 约 **prec 9.9%、lift 3.05**（仍显著低于牛市 test，但比无门控时可用得多）。

---

## 4. 建议的下一步（v4）

- [ ] 收紧 `bull_regime`（例如再加 **BTC 6M rolling return > 0** 或 **周线 EMA 斜率 > 0**），目标是把 bar 覆盖率从 ~80% 压到 ~40–55%，避免「弱牛 also-ran」污染彩票分布。  
- [ ] 让 `analyze_leverage_capacity_v3.py`（或 live）**读取 `gate_draft.yaml`**，避免脚本与 YAML 双源。  
- [ ] 把决策树导出为 `joblib` + 在线打分 hook，接到 paper/live（另开任务）。
