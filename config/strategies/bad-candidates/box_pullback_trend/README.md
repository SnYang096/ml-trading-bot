# BPT — Box Pullback Trend（Bad Candidate）

在**已校验的盘整盒子（box/chop）窗口**内，只做与**宏观趋势一致**一侧的回调（pullback）：宏观向上只做下沿多、宏观向下只做上沿空、宏观横盘不做。盒子只负责**选窗**，不负责双向 fade。

该候选在完整 **turbo `rolling_sim` 事件回测**（多币、按月重标定）下表现为**成交极度稀疏**，无法作为独立生产腿推进，因此移动到 `bad-candidates`。

---

## 为什么降级（详细）

### 1. OOS 成交密度过低，无法支撑统计与容量

一次代表性跑法（与本仓库管线一致：`prod_train_pipeline_2h_turbo_box_pullback_trend_only.yaml`，`rolling_sim`，`turbo_fixed_features`，事件回测拼接，`starter_a` / `highcap` 多币）产出量级如下（run_id `20260426_212621`，结果见 `results/box_pullback_trend/turbo-rolling-sim/_rolling_sim/20260426_212621/`）：

| 指标 | 数值 |
|------|------|
| 覆盖月份 | **27**（约 2024-01～2026-03） |
| 拼接总 R（`stitched_total_r`） | **约 +9.82 R** |
| 拼接总成交笔数（`stitched_total_trades`） | **30** |
| 粗算均值 | 约 **+0.33 R/笔**（未在此拆胜率/分年/摩擦） |

折算频率：**27 个月 × 多币 universe 上合计约 30 笔** → 月均约 **1 笔量级**（不是「每币每月」，而是**全组合**仍极稀）。对任何需要样本外稳健性、容量或风控校准的策略而言，这一密度都**不足以**做可靠的推广推断，也不适合单独占一条实盘策略腿。

### 2. 方向层与「稀疏」同源：覆盖与验证经常过不去

管线中 **Direction Validate** 在多个月份表现为失败或勉强：宏观 + 盒子边缘条件同时满足时，**有方向的行占比极低**（例如单月校准窗内仅数十行量级、Phase 2 相对基线样本量不足）。这说明：

- 策略假设在实现上是**自洽**的（宁缺毋滥），但产物是**天然长尾**；
- 继续调参往往只是在「更少但更纯」与「略多但更噪」之间权衡，**无法在不改变假设的前提下**把频率抬到与其它 2H 腿（BPC/TPC 等）可比的数量级。

### 3. Gate 在稀疏腿上的边际贡献有限

在 `disable_model_training` 的 turbo 快环里，Gate 优化常见结局是：**hard gate 难稳定产出**，re-apply 后决策接近「全放行」或仅 guardrail。对 BPT 这类**方向行本身就是稀有事件**的策略，门控很难在保持频率的同时再切一刀；**质量压力主要集中在 prefilter + direction**，与「独立可交易产品」的期望不匹配。

### 4. 与 TPC / 其它趋势族的关系

BPT 本质是「**趋势回调**」在 **box/chop regime** 下的窄子集。若最终需要类似 exposure，更现实的路径通常是：

- 把「box 窗口 + 宏观对齐」吸收为 **TPC（或别的主流趋势腿）上的 regime / 特征 / gate**，而不是维护一条**极低频的平行策略**；
- 或与 **BPC** 等腿做显式重叠分析后合并，避免多腿之间**几乎不成交却占维护成本**。

---

## 原研究假设（英文摘要，便于检索）

CRF 作为**双向 box fade** 失败后，诊断仍提示：在宏观过滤下，**顺宏观的盒边回调**优于逆宏观摸边。BPT 将 box/chop 明确为 **window selector**，宏观方向来自轻量锚（管线 v1 为各 symbol `EMA1200` 状态/斜率等，见 `research.yaml`），与 CRF 的「盒内双向均值回归」不同。

---

## 诊断与对比脚本（仍可复现历史分析）

```text
scripts/diagnose_box_pullback_trend.py
scripts/compare_box_pullback_trend_overlap.py
```

默认输出目录已指向 `results/bad-candidates/box_pullback_trend/...`；历史 `rolling_sim` 仍可能留在 `results/box_pullback_trend/` 下，便于对照旧 run。

---

## 归档说明

- **归档日期**：2026-04-27  
- **原因摘要**：全段 OOS 事件回测**总 R 略正但样本极少**，方向与门控结构决定其**不适合作为独立生产候选**。  
- **特征注册**：`bpt_macro_box_direction_f` 等仍可在 `config/feature_dependencies.yaml` 中保留，供其它策略复用；本目录仅表示**该策略 SKU 不再推进**。
