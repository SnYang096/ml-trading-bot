# bad-candidates — 已废弃或实验失败的策略归档

本目录下的策略**不参与** `config/constitution/constitution.yaml` 的 `enabled_archetypes` 白名单，仅供复盘与对照实验。

| 子目录 | 说明 |
|--------|------|
| `lottery100/` | 高杠杆彩票研究（Lottery100） |
| `fer/` | 旧 FER 实验树 |
| `fbf_exp_fatter_tp/` | FBF 肥 TP 实验（不如基线） |
| `rmr/` | Range Mean Revert MVP — 慢滚验证为负期望，语义上缺强事件铆钉；裁决见 `docs/z实验_005_统一研究/FBF_RMR_HubRebound_verdict_20260420.md` |
| `eml/` | EMA1200 umbrella 研究壳 — 开仓偏密、单笔置信度低于 BPC/TPC/ME，仅作基线；见 `eml/README.md` |
| `srb/` | Structural Range Breakout — rolling 未达主腿预期，已自 `config/strategies/srb` 迁入；管线仍可用 `prod_train_pipeline_*_srb_only.yaml` |

若要重跑归档管线，使用各目录旁或 `meta.yaml` 中注释的 `prod_train_pipeline_*` 路径（RMR：`config/prod_train_pipeline_2h_slow_rmr_only.yaml`，其中 `strategies.rmr.config` 已指向本目录；EML：`config/prod_train_pipeline_2h_turbo_eml_only.yaml`；SRB：`config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_srb_only.yaml` / `config/prod_train_pipeline_2h_slow_srb_only.yaml`）。
