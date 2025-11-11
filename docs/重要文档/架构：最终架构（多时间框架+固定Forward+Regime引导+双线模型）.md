# 最终架构（多时间框架 + 固定 Forward + Regime 引导 + 双线模型）

本架构整合以下三份文档的核心原则并落地为可执行流水线：
- `docs/架构：timeframe（时间周期）到底什么时候定怎么定.md`
- `docs/架构：forwardbars如何确定.md`
- `docs/架构：根据交易策略训练模型而不是一个统一大模型.md`

目标：构建一个可实盘、可维护、可扩展的系统，统一方法论并映射到当前代码，实现“感知 → 判断 → 决策 → 执行 → 反馈”的闭环。

---

## 一、关键原则（来自三文档的一致结论）
- timeframe（输入采样频率）在最早的数据采样阶段确定；forward（预测跨度）在特征工程前、用信息效率/自相关等方法确定；二者均非“网格搜出来的超参”。
- 降维/特征筛选必须在“固定 forward”的任务定义下进行，否则会造成特征-目标错配。
- 多周期共振：特征可以多周期（5m/15m/1h/4h），但预测任务需统一语义（例如都预测“未来6h收益”，或都预测“未来N个有效周期”的收益）。
- 显式 Regime 引导：先识别市场状态，再训练/路由到相应专家模型（动量/反转/突破）；训练与实盘均避免“一个万能模型同时学习互斥逻辑”。
- 训练验证采用 purged / walk-forward CV，关注稳健指标与跨 Regime 表现，避免过拟合与信息泄露。

---

## 二、系统总览

```mermaid
graph TD
A[Market Data Feeds] --> B[数据采样/重采样<br/>timeframe 确定]
B --> C[Alpha Horizon 分析<br/>forward 决策(信息效率/ACF)]
C --> D[特征库/特征工程<br/>多周期特征, 无泄漏窗口]
D --> E[Regime Detection<br/>Rule/LightGBM + HMM 平滑]
E --> F{Regime 类型/概率}
F --> G1[时序专家: Momentum@1h]
F --> G2[时序专家: MeanReversion@15m]
F --> G3[时序专家: Breakout@1h/4h]
G1 --> H[Meta/Fusion 加权]
G2 --> H
G3 --> H
D --> X[横截面多因子管线<br/>Fama-MacBeth/IC 筛选/组合构建]
H --> I[仓位/风险控制<br/>ATR/VAR/冷却/滑点成本]
X --> I
I --> J[执行/订单路由]
J --> K[监控与反馈<br/>IC/IR/漂移/再训练触发]
K --> C
```

### 仓位与风险控制（链接与要点）
- 详细方案文档：[`仓位：最终仓位与风险控制方案（Regime感知 + 波动率目标 + 反马丁格尔）`](./仓位：最终仓位与风险控制方案（Regime感知 + 波动率目标 + 反马丁格尔）.md)
- 要点概要：
  - 基础仓位 s0：以校准后概率与期望收益缩放，`s0 = base_risk × (2p-1)_+ × clip(|r̂|/r_target, 0, m_exp)`
  - 波动率目标：`s1 = s0 × target_vol/σ̂`（设缩放上下限，避免极端）
  - Regime 软加权：牛市放大、震荡适中、崩塌收缩（clip 到 0.5–1.8）
  - 反马丁格尔：胜后阶梯加仓、失败冷却；限制 `max_adds / max_mult`
  - 风险模式：Aggressive/Normal/Defensive 动态切换；全局约束（总敞口/单品种/beta_cap/换手预算）
  - 止损与止盈：结构止损优先 + σ̂ 硬止损兜底；分段止盈 + 尾仓 trailing

---

## 三、timeframe 与 forward 决策

- timeframe 决策（输入侧）：
  - 策略逻辑先行：动量（5m–1h）、反转（1m–15m）、波段趋势（1h–4h）。
  - 数据驱动校验：ACF/Hurst/互信息 峰值/拐点，结合交易成本、执行能力选择合适采样周期。
- forward 决策（输出侧）：
  - 信息效率/IC vs horizon 曲线找拐点，或用“有效预测窗口”（ACF 显著非零的最大 lag）确定。
  - 在降维/筛选之前固定 forward，后续所有特征与训练都围绕该 forward 构建。
  - 进阶：multi-horizon 训练 + 动态权重（按信息效率进行融合）。

工程建议：
- 输出一个 `forward_selection_report.json` 保存每策略/标的/周期的 forward 决策结果，供训练流水线消费。

---

## 四、特征工程与特征库（Feature Store）

- 语义分组：Trend、Mean-Reversion、Volatility、Structure、Breadth/On-chain、Regime Flags。
- 多周期特征对齐统一预测目标（例如皆预测未来6h收益，15m/1h/4h特征一起输入）。
- 无泄漏窗口：特征窗口通常取 `≈ 2–4 × forward`，标签用 `shift(-forward)`。
- 统计稳健性：rolling IC、IC 波动、稳定性评分；按组选择 Top-K。

---

## 五、Regime 检测（显式 + 平滑）

- v1：规则系统（Hurst + 波动分位 + 压缩度等）
- v2：轻量监督模型（LightGBM 分类器，规则标签作弱监督）
- v3：HMM 平滑（对状态序列降抖动）
- 多周期合成：加权投票得到统一 Regime 序列与概率

代码映射：
- `src/regime_detection/detector.py`（规则版 + 多周期聚合/权重）
- `src/regime_detection/features.py`, `src/regime_detection/hmm_smoother.py`, `src/regime_detection/config.py`

---

## 六、时序模型流水线（单资产，多周期）

定位：面向单资产的时序信号（回归/分位/分类/波动）与多周期融合。

现有模块：
- `src/time_series_model/pipeline/multi_tf_pipeline.py`：多周期分位回归（q10/q50/q90）、分类、波动模型的训练与预测。`forward_bars` 由外部传入。
- `src/time_series_model/pipeline/training/*`：训练/滚动训练/预处理/标签工具。
- `src/time_series_model/models/*.py`：LightGBM 封装、可解释因子引擎等。
- `src/time_series_model/strategies/*`：量化策略封装。
- `src/time_series_model/pipeline/rolling/auto_rolling_update.py`：自动滚动更新。
- `src/time_series_model/pipeline/dimensionality/*`：特征工程与降维/报告。

建议增强（与三文档对齐）：
1) Forward 决策内置化：新增 `forward_selection` 模块（信息效率/ACF），在训练前固化每策略/周期的 forward，并记录报告。
2) Regime 引导训练：将 `regime_detection` 输出作为 gating/flag 注入：
   - 方案A：在特征中加入 `is_trending/is_range` 等 regime flags。
   - 方案B：按 Regime 分桶训练专家模型，推理时用概率加权融合。
3) Multi-horizon 输出：为关键模型并行预测多种 forward（如 3h/6h/12h），按效率动态加权。
4) Purged Walk-Forward CV + Embargo：在 `training/train.py` 引入无泄漏验证与稳健度指标（IC/IR/Calmar/回撤）。
5) 多周期融合策略：在 `multi_tf_pipeline` 增加按 Regime 与波动状态的自适应权重。
6) 监控与校准：引入线上校准（近期 OOS 表现加权）、模型漂移监控、再训练触发条件。

---

## 七、横截面多因子流水线（多资产，单时点回归）

定位：以因子暴露解释/预测横截面收益（Fama-MacBeth 风格），用于截面排序、组合构建与风险控制。

现有模块：
- `src/cross_sectional/panel_generation.py`, `panel.py`, `processing.py`：面板生成与处理。
- `src/cross_sectional/factor_catalog.py`, `crypto_factors.py`：因子目录与加密资产因子。
- `src/cross_sectional/factor_selection.py`：截面 IC 计算与基于 IC/IR 的筛选。
- `src/cross_sectional/model.py`：Fama-MacBeth 风格回归，Newey-West 标准误、IC 汇总、预测。
- `src/cross_sectional/boosting.py`, `report.py`：提升模型与报表。

建议增强：
1) 面板稳健化：按资产滚动标准化与 winsorize、流动性/交易成本/可借券过滤、分组中性化（市值/行业/风格中性）。
2) 因子治理：IC 稳定性、IC 衰减、相关性/聚类去重、multiple-testing 控制。
3) 组合构建：引入优化器（风险约束、换手/成本惩罚、净敞口/行业/风格约束），与交易成本模型耦合。
4) Regime 叠加：在组合权重上叠加 Regime 权重（趋势市提升动量类因子；震荡市提升反转/结构类因子）。
5) 标签与 forward：横截面目标也应统一 forward 定义（如未来 1–3 天/6–12 小时的截面收益），与时序侧保持一致的语义。
6) 回测与联动：与时序信号在仓位/风险层融合，避免冲突交易（信号一致时加权，冲突时降权/中性）。

---

## 八、训练与验证（共同）
- 验证策略：purged / walk-forward CV，时间切片稳健性评估，Regime 分层表现。
- 指标：IC/IR、分位回归覆盖率（pinball loss）、分层收益、Calmar、回撤、胜率/盈亏比、成本后收益。
- PBO/漂移检测：参数平稳区间优先（flat-plateau），分布漂移触发再训练与降权。

---

## 九、在线更新、监控与模型治理
- 在线校准：近期 OOS 表现加权、分层剔除衰退专家。
- 漂移监控：特征分布漂移、IC 漂移、Regime 占比变化报警。
- 模型治理：版本化（按 Regime/周期/资产）、影子交易/灰度上线、回滚流程。

---

## 十、与当前代码的模块映射

- Regime 检测：
  - `src/regime_detection/detector.py`：规则 + HMM（可作为 v1/v3）
  - 建议新增：`regime_detection/lgb_classifier.py`（v2 分类器）
- 时序模型（单资产）：
  - `src/time_series_model/pipeline/multi_tf_pipeline.py`：多周期分位/分类/波动模型
  - `src/time_series_model/pipeline/training/*`：训练/滚动/预处理/标签
  - 建议新增：
    - `time_series_model/pipeline/training/forward_selection.py`（信息效率/ACF）
    - `time_series_model/pipeline/training/regime_gating.py`（专家路由/加权）
    - `time_series_model/pipeline/training/walkforward.py`（purged WFCV + embargo）
- 横截面（多资产）：
  - `src/cross_sectional/*`：因子目录、面板、IC、Fama-MacBeth、报表
  - 建议新增：
    - `cross_sectional/portfolio.py`（优化器/约束/成本）
    - `cross_sectional/neutralization.py`（市值/行业/风格中性）
    - `cross_sectional/governance.py`（因子治理与多重检验控制）

---

## 十一、优先级路线图（实现顺序）

1) 固定 forward 的“前置分析模块”（信息效率/ACF）与报告产出；训练流程读取该结果。
2) 将 Regime 输出引入时序流水线（flags + 专家训练 + 概率加权）。
3) Walk-forward（purged + embargo）与稳健指标落地，替换单次切分。
4) 横截面面板稳健化与组合构建（成本与约束），并与时序信号在仓位层融合。
5) 监控/校准/治理闭环（IC 漂移、专家降权、自动再训练触发）。

---

## 十二、实施 TODO（按子系统）

时序模型（单资产）：
- 固化 forward 决策：新增 `forward_selection.py`，输出报告与配置。
- Regime 引导：在特征注入 regime flags，并新增 `regime_gating.py` 做专家路由与集成。
- Multi-horizon：为关键模型并行预测 3h/6h/12h，按信息效率加权。
- 无泄漏验证：新增 `walkforward.py`（purged + embargo）并接入 `training/train.py`。
- 多周期融合：在 `multi_tf_pipeline` 按 Regime 与波动状态自适应加权。
- 线上校准与监控：近期 OOS 绩效回灌，自动降权/再训练触发。

横截面（多资产）：
- 面板稳健化：滚动 zscore/winsorize、流动性与可交易性过滤。
- 中性化与去冗余：市值/行业/风格中性、因子相关性聚类与去重。
- 组合构建：新增 `portfolio.py` 做优化器（风险/成本/约束）与回测接口。
- Regime 叠加：不同 Regime 下的因子权重调度策略。
- 因子治理：IC 稳定性/衰减、多重检验控制与退场机制。

公共/治理：
- 统一配置与报告产出（forward/timeframe 决策、CV 报告、回测汇总）。
- 模型注册/版本化与影子交易/回滚流程。

---

## 十三、备注与度量
- Forward/Timeframe 决策与训练/特征工程的解耦是避免过拟合的关键。
- 以“统一预测语义”替代“强行统一时长”，必要时用 multi-horizon + 动态权重。
- 所有验证均以 purged WFCV + embargo 为准线；报告中强制输出 Regime 分层结果。

以上文档可作为项目级别的“总体架构说明”，配合下述待办列表（Backlog）推进实现。***

