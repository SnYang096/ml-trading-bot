# Lottery100（原 L3 彩票 / 杠杆容量）研究方法论 — 与训练管线的关系

本文固化 **v1 → v4** 杠杆容量统计与 **B+ 回测** 的入口与关系，便于复现与评审；**不等同于** ME/BPC/TPC 的 NN + gate 训练管线。

**策略配置目录**：`config/strategies/bad-candidates/lottery100/`（曾用名 `l3_lottery`）。  
**与 BPC prod 对齐的「壳」**：`config/prod_train_pipeline_2h_lottery100.yaml`；串联执行见 **`scripts/run_lottery_research_bundle.py`** → 产出 **`results/lottery100_bundle/`**。

---

## 1. 要不要把 Lottery100 接进 `research_pipeline`？

**结论：研究阶段不必；实盘若要自动点火，再接一条「薄管线」。**

| 目标 | 建议 |
|------|------|
| **回答「历史上哪里能扛 100x / 特征长什么样」** | 保持现有 **`analyze_leverage_capacity_*.py` 离线统计** 即可；这是 ** MAE/MFE + 分桶 + lift + 浅树 **，与监督学习管线目标不同。 |
| **版本化阈值与复盘** | 已有 **`config/strategies/bad-candidates/lottery100/gate_draft.yaml`**；后续应让脚本 **读取同一 YAML**（避免双源）。 |
| **像 ME 一样每月产模型 artifact** | **一般不必要**：彩票层样本稀、体制切换强，浅规则/小树 + **宏观门**比大规模 NN 更合适。 |
| **实盘自动执行 / 纸交易** | 可增 **单独入口**：定时拉 feature store → 算 gate + 排序分 → 写意图队列；**不必**塞进 `train_strategy_pipeline` 的全套 SHAP/rolling。 |

**不要混淆两条「管线」：**

- **Archetype 管线**：特征 → 标签 → prefilter/gate/direction → 模型 → 回测（`research_pipeline.yaml` / prod train yaml）。
- **Lottery100 研究管线**：OHLCV 或 FS → 前向窗口 MAE/MFE → `L_max` 分桶 →（可选 funding）→ 特征 lift / 决策树 / OOS。

---

## 2. 方法版本与脚本对应

| 版本 | 脚本 | 输入 | 核心增量 |
|------|------|------|----------|
| **v1** | `scripts/analyze_leverage_capacity.py` | `DataHandler` OHLCV，12 个轻量特征 | 多 horizon、多空、杠杆档；特征 lift；浅树 |
| **v2** | `scripts/analyze_leverage_capacity_v2.py` | Feature store（默认 `features_me_120T_*`） | CVD/VPIN/OI/funding/BPC/ME 语义；funding 调整 MAE；≥100x 高原去首；train/test/oos 树；Top-K 精度 |
| **v3** | `scripts/analyze_leverage_capacity_v3.py` | 同上 + **BTC 周线 close > EMA(50)**（滞后一周） | 每 bar `bull_regime`；`--bull-only` 子样本重跑全套统计 |
| **v4** | `scripts/analyze_leverage_capacity_v4.py` | 同上；**宏观门完全由 YAML 驱动**（默认：周线牛 ∧ **月线 6M 收益 > min** ∧ 可选周线 EMA 斜率） | `config/strategies/bad-candidates/lottery100/leverage_capacity_v4.yaml` |
| **B+** | `scripts/lottery_backtest_bplus.py` | 读 v* 产出的 **parquet** + FS 收盘价；**H 根后平仓**；扣 funding 与往返费用；**trades + equity** | `config/strategies/bad-candidates/lottery100/backtest_bplus.yaml` |

**报告（实验笔记）：**

- `docs/z实验_008_lottery/`（归档 v1–v4 杠杆容量文档，见该目录 `README.md`）
- 亦保留于 `docs/z实验_007_lv/` 中 v1–v3 副本（与 008 同步复制）

**数值产物：** 历史运行仍可能在 `reports/leverage_capacity_v2/` 等旧路径；**bundle 默认输出**为 **`results/lottery100_bundle/`**（v4 子目录 + `bplus/` + `summary_bundle.json`）。

---

## 3. 统一定义（与代码一致）

- **维护保证金率 MMR**：默认 `0.004`（0.4%），用于 \(L_{\max} \approx (1-\mathrm{MMR})/\mathrm{MAE}\)。
- **MAE/MFE**：固定 horizon \(H\)（单位：根 120T bar），向前看路径极值；v2/v3 对 funding 做保守窗口累计。
- **≥100x 去重**：连续满足 \(L_{\max}\ge 100\) 的「高原」只保留 **首根 bar**，避免自相关夸大 base rate。
- **Bull regime（v3）**：BTC `W-FRI` 周线收盘 vs 周线 EMA50，**周线信号滞后一周**再展开到 120T（避免偷看当周收盘）。

---

## 4. 与代码库其它「L4」用语的区别

本仓库别处（如实盘监控、Gate 漏斗）常把 **第 4 层** 叫作 **L4 Gate**。  
**本文档中的 L3** 仍是 **加密内三层**里的彩票层；**跨市场 β 轮动**在 `实施文档_04` 里位于 **加密三层之外**，在 [l4_cross_market_beta.md](../portfolio/l4_cross_market_beta.md) 中单列为 **组合层 L4**，避免与「漏斗 L4」混淆。

---

## 5. B+ 回测（交易级 PnL / 资金曲线）

从杠杆容量 parquet 对齐 FS 收盘价，机械执行「开仓 bar 收盘 → 持有 H 根 → 平仓收盘」，输出 `trades.csv`、`equity_curve.csv`。  
脚本 **`scripts/lottery_backtest_bplus.py`**，默认配置 **`config/strategies/bad-candidates/lottery100/backtest_bplus.yaml`**；口径说明见 **`docs/z实验_008_lottery/B+回测说明.md`**。

## 6. 后续建议（维护者）

1. **`analyze_leverage_capacity_v4.py`**（若做）：从 `gate_draft.yaml` 读阈值；收紧 bull（如 6M 收益、斜率门）。  
2. **实盘**：独立子账户 + 仅消费 L2 利润 + `M_glob`/`M_sector` 门（见 `实施文档_04` §4.2）。  
3. **不把 Lottery100 纳入 archetype SHAP 流水线**，除非明确要训练「彩票专用」判别模型并单独评估 OOS。
