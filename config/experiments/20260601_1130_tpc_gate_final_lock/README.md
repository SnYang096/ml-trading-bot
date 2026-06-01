# TPC Gate 最终锁定实验（20260601_1130）

**目的**：用 `config/market_segment.yaml` 定义的三个 canonical 市场阶段，彻底决定 TPC gate 层应该保留哪些规则（以及用什么形态），或者把剩余的 disabled 规则全部删除。

这是计划中“确定 TPC 配置”所剩的最后一个关键回测 grid。

## 实验设计原则（来自 2026-06-01 规划）

- 只用官方三个市场阶段（不再用手写窗口）
- 核心对比：G0（当前 prod） vs G1（已验证最好的方案：两条 bull vol 中间带全部 disabled）
- 只额外加一个“最后的机会”单调单边候选：vol_leverage 极低尾（`< 0.03` 且仅 bull）
  - 理由：这是目前唯一在 label scan 上方向正确（低尾更差）的单边写法
  - 其他单边（vol_persistence 低/高尾、EVT 低尾）已在 20260602 monotonic 实验中被 label 证伪，无需再浪费回测资源
- Chop gate 暂不触碰（G2 单独关掉的代价在之前实验已可见）

跑完本 grid 后，即可：
1. 确认 G1 是否在熊市、主牛、近期高位震荡转熊三个阶段都稳健
2. 决定是否把 vol_leverage 极低尾作为最终形态（概率较低）
3. 干净地删除 gate.yaml 里剩余的 disabled 规则（或保留作为历史）

## 如何运行

```bash
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260601_1130_tpc_gate_final_lock/tpc_gate_final_lock_grid.yaml \
  --quiet-signal-logs
```

## 后续清理（跑完后）

- 更新 `config/strategies/tpc/archetypes/gate.yaml`（删除或最终形态）
- 同步 `live/highcap/...`
- 更新主 DECISION 文档
- 可选：把旧的 `20260601_tpc_gate_validate` 和 `20260601_1126_tpc_gate_monotonic_validate` 目录按新命名规范重命名一次（见下文）

## 推荐的实验目录命名规范（从本实验开始严格执行）

为了让目录按时间自然排序（最新的永远在最下面），统一使用：

**`YYYYMMDD_HHMM_<简短主题>`**

示例：
- `20260601_1130_tpc_gate_final_lock`   ← 本实验（推荐）
- `20260531_0945_tpc_gate_validate`
- `20260601_1126_tpc_gate_monotonic_validate` （可改名为 20260601_1126_...）

这样 `ls` 或文件管理器排序时，时间顺序 = 字母顺序，最新的永远排在最后。

建议对之前两个命名不规范的目录做一次一次性重命名（不影响内容）：
- `20260601_tpc_gate_validate` → `20260531_1530_tpc_gate_validate`（或你实际开始时间）
- `20260601_1126_tpc_gate_monotonic_validate` 保持或微调为一致格式

本实验严格遵守新规范。

## 运行状态（实时更新）

- **启动时间**: 2026-06-01 ~03:38 UTC (北京时间 11:38)
- **命令**: `PYTHONPATH=src:scripts python -m scripts.event_backtest --variant-grid .../tpc_gate_final_lock_grid.yaml --quiet-signal-logs`
- **后台 PID**: 3035 (shell wrapper 393731)
- **当前阶段**: 数据加载 / 特征构建初始化（3 个市场段 × BTC+ETH × 3 variants）。这类 grid 早期通常 3-8 分钟无 stdout。
- **策略树准备**:
  - G0: `tpc_gate_ablate_G0_prod_strategies` (冻结快照)
  - G1: `tpc_gate_ablate_G1_no_bull_vol_strategies` (冻结快照，当前 prod 基线)
  - G10: `tpc_gate_G10_vla_lt003_bull_strategies` (已创建并 patch：vol_leverage 改成 `< 0.03` 单边 bull-only，已启用)
- **预期输出目录**: `results/tpc/experiments/gate_final_lock/{G0,G1,G1_vla_lt003}/...`

**监控方式**：我会在后台持续 poll，一旦出现 variant 启动、segment 完成、R-multiple 数字或错误，会立即分析并给出最终结论 + gate.yaml 清理建议。

跑完后下一步（自动执行）：
1. 汇总三个 segment × 3 variants 的总 R、maxDD、CAGR 等核心指标
2. 对比 G1 vs G0 是否在所有三个真实市场阶段都稳健领先
3. 判断 G10（vol_lev 极低尾）是否有任何价值
4. 给出明确的“删除所有 vol_* gate 规则”或“保留最终形态”的推荐
5. 生成 patch 直接应用到 `config/strategies/tpc/archetypes/gate.yaml` + live/highcap 同步
6. 更新主 DECISION 文档

目前正在全力跑，预计整体 20-60 分钟（取决于机器负载）。有任何输出我都会第一时间报告。

---

## 2026-06-01 实际运行结果（已结束）

**状态**：**未完整跑完，途中崩溃**（非正常结束）。

### 已完成的部分
- G0 / bear_2022：成功（30 trades, Total R = **-0.116 R**，CAGR ≈ -0.64%，maxDD 4.0%）
- G1 / bear_2022：成功（与 G0 **完全相同**结果）
- 原因：在 bear_2022 里 ema_1200_position 极少 >0.10，两条 bull-only vol gate 基本不触发，所以 G0 与 G1 表现一致（符合预期）。

### 崩溃点
- 尝试加载 **G10**（vol_leverage < 0.03 bull-only）时 YAML 解析失败：
  ```
  ScannerError: mapping values are not allowed here
  ... reason: Final monotonic candidate (20260601): vol_leverage ...
  ```
- 根本原因：`reason:` 和部分 `lock_reason:` 字段写了带英文冒号 `:` 的长描述字符串，未加引号，YAML parser 把 `:` 当成映射开始。
- 已修复：G10 的 gate.yaml 里所有长描述字段（reason / comment / lock_reason）全部加双引号。

### 当前结果目录
只存在：
- `results/tpc/experiments/gate_final_lock/G0/bear_2022/`
- `results/tpc/experiments/gate_final_lock/G1/bear_2022/`
（无 bull_2023_2024、无 recent_range_to_bear、无任何 G10 输出）

### 下一步建议（见用户对话）
见正文回答。推荐：**先把 G10 彻底放弃**，直接重跑一个**干净的只含 G0 vs G1 的三段 grid**，用 canonical 三个市场阶段把 TPC gate 这件事一次性锁死。

---

**教训（直接回答你的第二个问题）**：
IC / label scan 上“有用”的特征，在最终 segmented backtest 里经常失效，**不是单纯因为 OOS 运气差**，而是系统性原因（详见正文）。这正是我们坚持用 `market_segment.yaml` 三个真实阶段做最终 gate 判决的原因 —— 只有跨 regime 的总 R + 风险指标能说话，label/IC 只能用来生成假设，不能用来决定生产配置。


