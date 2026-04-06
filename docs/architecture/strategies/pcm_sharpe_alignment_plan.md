# PCM Sharpe 对齐方案：从研究到实盘的收益保真

> 创建: 2026-02-28
> 状态: P0 执行中

## 1. 问题定义

| 回测类型 | Sharpe(R) | 交易数 | 胜率 | slot 限制 |
|----------|-----------|--------|------|----------|
| 研究回测（向量，无slot） | ~0.43 (FER) | 159 | ~55% | 无 |
| 事件回测 slot=2 | 0.0046 | 1770 | 51.0% | 2 (全局) |
| 事件回测 slot=3 | 0.0096 | 2736 | 51.4% | 3 (全局) |

**核心矛盾**: 研究 Sharpe 虚高（无限仓位假设），但实盘只能开 2-3 slot。
**最终目标**: 在 2-3 slot 约束下，让实盘 Sharpe 也达到合理水平（>0.3）。

## 2. 根因分析

### 2.1 研究 Sharpe 为何虚高

研究回测对每个信号独立开仓，无资源竞争：
- 同一时刻 BTC/ETH/BNB 各有一个信号 → 全部开仓
- 无机会成本：所有好信号和坏信号都被执行
- Sharpe 反映的是 "如果能吃掉所有机会" 的理论上限

### 2.2 事件回测 Sharpe 为何崩塌

信号漏斗（slot=3 结果）:
```
total_signals_checked         : 52066
signals_generated             :  3065 (5.9%)
reject_pcm_slot_full          : 36669 (70.4%)  ← 主因
reject_gate_deny              : 12332 (23.7%)
reject_max_positions          :   329 (0.6%)
```

问题不是信号少，而是 **slot 满时不选信号质量，先到先得**：
1. BTC 来了一个弱信号（evidence=0.30），占了 1 slot
2. 10 分钟后 ETH 来了强信号（evidence=0.55），slot 满被拒
3. BTC 弱信号亏了 -1R，ETH 强信号本来能赚 +2R → 净损失 3R

### 2.3 差距的本质

| 维度 | 研究回测 | 事件回测 | 差距原因 |
|------|---------|---------|---------|
| 信号选择 | 所有通过 gate 的 | 先到先得 | 无质量排序 |
| 仓位替换 | 不需要 | 不支持 | 好机会被浪费 |
| 跨 symbol | 独立并行 | 全局竞争 | 资源分配不智能 |
| 持仓时间 | 无影响 | 占 slot 时间 | 拖尾亏损阻塞 slot |

## 3. 改进路线（按优先级）

### P0: 建立真实基线（当前阶段）

**目的**: 搞清楚在 slot 约束下，理论最优 Sharpe 是多少。

#### P0.1: 向量回测加 slot 限制
```bash
# slot=3 (constitution 当前值)
python scripts/backtest_execution_layer.py \
  --pcm bpc:results/train_final_20260228_154956_rr_extreme/bpc/logs_gated.parquet \
       fer:results/train_final_20260228_155621_rr_extreme/fer/logs_gated.parquet \
       me:results/train_final_20260228_160522_rr_extreme/me/logs_gated.parquet \
  --strategies-root config/strategies \
  --quantile-train-start 2025-02-01 --quantile-train-end 2025-08-01

# slot=2 (实盘配置对比)
python scripts/backtest_execution_layer.py \
  --pcm bpc:results/train_final_20260228_154956_rr_extreme/bpc/logs_gated.parquet \
       fer:results/train_final_20260228_155621_rr_extreme/fer/logs_gated.parquet \
       me:results/train_final_20260228_160522_rr_extreme/me/logs_gated.parquet \
  --strategies-root config/strategies \
  --quantile-train-start 2025-02-01 --quantile-train-end 2025-08-01 \
  --max-slots 2

# 无 slot 限制 (理论上限)
python scripts/backtest_execution_layer.py \
  --pcm bpc:results/train_final_20260228_154956_rr_extreme/bpc/logs_gated.parquet \
       fer:results/train_final_20260228_155621_rr_extreme/fer/logs_gated.parquet \
       me:results/train_final_20260228_160522_rr_extreme/me/logs_gated.parquet \
  --strategies-root config/strategies \
  --quantile-train-start 2025-02-01 --quantile-train-end 2025-08-01 \
  --max-slots 999
```

#### P0.2: 信号质量分析
```bash
python scripts/analyze_signal_quality.py \
  --logs bpc:results/train_final_20260228_154956_rr_extreme/bpc/logs_gated.parquet \
        fer:results/train_final_20260228_155621_rr_extreme/fer/logs_gated.parquet \
        me:results/train_final_20260228_160522_rr_extreme/me/logs_gated.parquet
```

**关键问题**:
- [ ] Top 30% 信号贡献多少 Total R？
- [ ] Evidence score 高分位的信号是否显著更好？
- [ ] 如果只跑 top 30% 信号 + slot=3，Sharpe 是多少？

### P1: PCM 智能选信号（代码: `src/time_series_model/portfolio/live_pcm.py`）

**原理**: slot 满时，新信号不被简单拒绝，而是和当前最弱持仓比较。

```
当前逻辑:
  slot 满 → 拒绝新信号 → return REJECT

改进逻辑:
  slot 满 → 新信号 evidence_score vs 最弱持仓的 remaining_expected_r
  如果新信号预期 R > 最弱持仓剩余 R × 1.5 (安全系数) → 替换
  否则 → 拒绝
```

**需要的数据**:
- 新信号: evidence_score, archetype, symbol
- 已有仓位: entry_time, current_pnl_r, holding_duration, archetype

**安全约束**:
- 只替换浮亏仓位或持仓超过 N bars 的仓位
- 刚开仓的 (<4 bars) 不替换
- 替换 = 市价平旧 + 市价开新，考虑双向手续费

### P2: 仓位替换模型

比 P1 更精细：
- 预测每个持仓的 **剩余 R 预期**（基于已持有时间、当前浮盈、ATR 状态）
- 新信号的 **预期 R** = f(evidence_score, archetype, market_regime)
- 如果 E[新] - E[旧] > 替换成本(约 0.1R) → 触发替换

### P3: Gate 阈值校准

> 注意: evidence_score 不影响是否开仓，只影响仓位大小(size_multiplier)和 SL/TP 参数。
> 控制是否开仓的是 **Gate**（硬 allow/deny）。

如果 P0.2 发现只有高质量信号赚钱：
- 收紧 **Gate 阈值**（如 shd_pct、vpin、cvd 等 deny_if 条件）→ 减少信号量 → 减少 slot 竞争
- 或添加 **evidence_score 最低门槛** 作为新 Gate 规则（evidence < 0.4 → deny）
- 等效于用更少但更好的信号填满有限 slot
- 可能是最简单的高 ROI 改进

代码关联:
- Gate 规则: `config/strategies/{bpc,fer,me}/archetypes/gate.yaml`
- Evidence 评分: `config/strategies/{bpc,fer,me}/archetypes/evidence.yaml`
- Tier 参数: `config/strategies/{bpc,fer,me}/archetypes/execution.yaml` (size_multiplier/SL/TP)

## 4. 预期收益估算

| 阶段 | 改进项 | 预计 Sharpe 提升 | 工作量 |
|------|--------|-----------------|--------|
| P0 | 基线建立 + 分析 | 无（诊断） | 1 天 |
| P1 | 智能选信号 | +50~100% | 3 天 |
| P2 | 仓位替换 | +30~50% | 5 天 |
| P3 | 阈值校准 | +20~40% | 1 天 |

**保守估计**: P0 + P3 就可能把 Sharpe 从 0.01 提升到 0.05+
**乐观估计**: P0 + P1 + P3 可能达到 0.1~0.2

## 5. 代码关联

| 模块 | 文件 | 作用 |
|------|------|------|
| PCM 仲裁 | `src/time_series_model/portfolio/live_pcm.py` | slot 分配、信号优先级 |
| 向量回测 | `scripts/backtest_execution_layer.py` | slot 限制后处理 |
| 事件回测 | `scripts/event_backtest.py` | 实时 PCM 仲裁模拟 |
| 信号分析 | `scripts/analyze_signal_quality.py` | 信号质量分布诊断 |
| Constitution | `config/constitution/constitution.yaml` | slot_count、per_strategy_limits |
| PCM 配置 | `config/pcm_regime.yaml` | 策略优先级、regime 缩放 |
