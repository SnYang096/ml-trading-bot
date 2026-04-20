# 实验 008 — L3 彩票 / 杠杆容量（文档归档）

本目录汇总 **彩票层（高杠杆容量）** 的历史实验笔记与脚本入口；与 `z实验_007_lv`（LV 清算语义）区分开：`007` 偏重 **Leverage Vulnerability archetype**，`008` 偏重 **L3 lottery + MAE/MFE 统计**。

## 文档索引（本目录）

| 文件 | 内容 |
|------|------|
| `杠杆容量统计_highcap_120T_v1.md` | v1：OHLCV-only，highcap 多品种 |
| `杠杆容量统计_BTC_ETH_120T_v2.md` | v2：feature store + funding + OOS |
| `杠杆容量统计_BTC_ETH_120T_v3.md` | v3：BTC 周线牛门 |
| `杠杆容量统计_BTC_ETH_120T_v4.md` | v4：YAML 驱动 + 更严牛门（周线 ∧ 6M 月收益） |

通用方法说明（是否与 NN 管线集成）：**`docs/architecture/strategies/lottery100_research_methodology.md`**

## 脚本与配置

| 脚本 | 配置 |
|------|------|
| `scripts/analyze_leverage_capacity.py` | CLI |
| `scripts/analyze_leverage_capacity_v2.py` | CLI |
| `scripts/analyze_leverage_capacity_v3.py` | CLI（周线牛） |
| `scripts/analyze_leverage_capacity_v4.py` | **`config/strategies/bad-candidates/lottery100/leverage_capacity_v4.yaml`** |
| **`scripts/lottery_backtest_bplus.py`**（**B+：交易表 + 资金曲线**） | **`config/strategies/bad-candidates/lottery100/backtest_bplus.yaml`** |
| 特征门草案 | `config/strategies/bad-candidates/lottery100/gate_draft.yaml` |

B+ 说明见 **`B+回测说明.md`**。

## 数值产物目录（仓库内，不复制进本文件夹）

- `reports/leverage_capacity_v2/`
- `reports/leverage_capacity_v3_all/`、`reports/leverage_capacity_v3_bull_only/`
- `reports/leverage_capacity_v4_bull_only/`（示例：`--bull-only` 跑 v4）

## 与 `z实验_007_lv` 的关系

v1–v3 报告最初写在 `z实验_007_lv/`；现已 **复制** 至本目录作为彩票专题归档。`007` 下原文件可保留或日后删除，以你的习惯为准。
