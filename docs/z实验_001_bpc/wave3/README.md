# Wave 3 — Label Refactor 实验归档

**时间**：2026-04-23
**范围**：BPC 策略 slow 管线的 label 语义改造尝试
**结论**：**方向证伪**。label 语义不是 BPC 退化根因。基础设施保留，label 改动全部回退。

---

## 背景一句话

Wave 1+2 封堵了 slow 管线若干过拟合自由度后，BPC 在 2024-04~06 仍从历史 +132R 退到 −67R。Wave 3 假设是 Prefilter / Gate / EF 的 label（统一用 `forward_rr >= -0.8`）和 BPC 策略语义错位，想通过三步改 label 测试能否修复。

## 主要产出

| 产出 | 类型 | 去向 |
|---|---|---|
| label 改造代码（Step 1/2） | **已回退** | 不进主线 |
| `run_step(env_extra=...)` 子进程环境变量注入 | **保留** | 未来对照实验用 |
| `scripts/compare_monthly_pnl.py` 月度对比工具 | **保留** | 通用基础设施 |
| baseline 锚点（6 个 fast_month timestamps） | **保留** | 下次诊断 D 方向免跑 baseline |
| 失败实验数据 + 根因分析 | **保留** | 本目录 |
| 三管线分工 + 慢管线改造设计 | **新产出** | 本目录 03/04 |

## 文件导航

| 文件 | 谁读 | 读多久 |
|---|---|---|
| [README.md](README.md)（本文件） | 所有人 | 2 分钟 |
| [00_why_and_plan.md](00_why_and_plan.md) | 想了解为什么做 Wave 3 | 5 分钟 |
| [01_experiments_and_findings.md](01_experiments_and_findings.md) | 想看 Step 0/1/2 实验细节 | 10 分钟 |
| [02_meta_findings_on_meta_algo.md](02_meta_findings_on_meta_algo.md) | **核心结论**，方法论层 | 15 分钟 |
| [03_three_pipeline_role_division.md](03_three_pipeline_role_division.md) | 未来运维参考（老/快/慢分工） | 10 分钟 |
| [04_slow_mode_redesign_candidate_discovery.md](04_slow_mode_redesign_candidate_discovery.md) | 准备改造慢管线前 | 15 分钟 |
| [baseline_bpc_wave2_runs.txt](baseline_bpc_wave2_runs.txt) | 做 D 方向对照实验前 | 1 分钟 |
| [baseline_vs_early_good_reference.md](baseline_vs_early_good_reference.md) | 看 baseline 和早期 20260413 的 diff | 5 分钟 |
| [step1_wave3a_bpc_monthly_diff.md](step1_wave3a_bpc_monthly_diff.md) | Step 1 no-op 的原始数据 | 3 分钟 |
| [step2_wave3b_bpc_monthly_diff.md](step2_wave3b_bpc_monthly_diff.md) | Step 2 FAIL 的原始数据 | 3 分钟 |

**如果只看一份** → [02_meta_findings_on_meta_algo.md](02_meta_findings_on_meta_algo.md)。

## 不要重复的坑

1. 不要再试 "改 label 语义让 meta-algo 学得更好" —— 01/02 已证伪。
2. 不要为慢管线加 Purged K-fold —— 02 里解释了为什么不适用金融时序生产场景。
3. 不要让慢管线 "自动落地" `gate_draft.yaml` —— 04 里给了候选发现器的改造方案。
4. 快管线 A/B 要跑 **全历史**（2024-01~今），不是 1~2 个月 rolling —— 03 里定义了验收规则。

## 下一步（D 方向，本次未做）

**聚焦 BPC execution 层与 Prefilter locked 阈值诊断**，具体：
- 2024-04 baseline 12 笔 / mean −3R/笔 → 看 SL 是否过紧
- Prefilter 3 条 locked 规则（`pullback_score ≤ 0.55`、`breakout ≥ 0.25`、`recovery ≥ 0.6`）在 2024-03 急涨 / 2025-11 死月是否需要 regime 化
- 直接用本目录 baseline runs 做对照，不再重跑 baseline

改动方案单独立文档，不放本目录。
