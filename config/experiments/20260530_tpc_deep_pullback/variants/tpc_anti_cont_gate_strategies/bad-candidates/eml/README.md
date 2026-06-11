# EML — EMA1200 Trend Umbrella（归档为 bad-candidate）

EML 已从 `config/strategies/eml` 移至本目录，与 `lottery100`、`crf` 等一致：**不作为主策略线推进**，仅保留配置以便对照实验与基线复跑。

## 为什么放在 bad-candidates

- **开仓偏多**：umbrella 语义偏宽（大结构 + 多路 entry filter 拼接），通过样本量明显高于 BPC / TPC / ME 等「单家族、强门控」策略；宽口径在震荡段容易反复触发，交易次数堆高。
- **单笔置信度低于 BPC / TPC / ME**：生产三族各自有完整的 prefilter / gate / 标签与家族语义铆钉；EML 是合并式实验壳，门控相对轻，**信号纯度与可解释性**不如三族，不适合与它们抢同一套资金与 slot。
- **定位只能是基线 / 研究层**：适合回答「在统一 EMA1200 大结构下，各子语义入口长什么样、漏斗如何漏」，以及和 BPC/TPC/ME 的 **R 与样本对照**；不是 BPC/TPC/ME 的替代。

若要跑管线，仍使用仓库根目录下的 `config/prod_train_pipeline_2h_turbo_eml_only.yaml` 等配置（其中 `strategies.eml.config` 已指向本路径）。
