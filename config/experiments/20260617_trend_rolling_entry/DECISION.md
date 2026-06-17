## ✅ 最终决策: R2 VWAP滚仓 (2026-06-17)

### 选定方案

```yaml
entry:   champion (深熊5% + EMA1200金叉VWAP + 动量向上)
leverage: 2x → 3x
roll:    价格跌20% + 盈利 + 价格在VWAP1200±5%  # ← R2改进
exit:    权益5x阶梯卖出, ≥15x可清仓
risk:    权益DD 50%减半仓, 价格DD 60%减半仓 (28d冷却)
```

### R2 vs 基线对比

| 指标 | R0基线 | 🏆 R2 VWAP |
|------|--------|-----------|
| 总收益 | $263,313 (6.58x) | **$259,246 (8.64x)** |
| 最差回撤 | -53.7% | **-45.0%** ⬇8.7pp |
| SOL DD | -53.7% (1次风控) | **-45.0% (0次风控)** |
| Calmar | 12.25 | **19.20** |
| 爆仓 | 0 | **0** |

### 为什么R2更好

VWAP1200 = 过去1200根bar的市场平均成交价 = 主力成本区。
在VWAP附近滚仓 = 在"聪明钱"的成本区加杠杆，不是盲目追。

SOL是最大受益者：R0在非支撑位滚仓导致-53.7%回撤+风控减仓，
R2只在VWAP支撑位滚仓 → 回撤-45%且0次风控。

### 爆仓风险: 几乎不可能

- ETH 最低权益 ~$42k (距爆仓 420倍)
- SOL 最低权益 ~$36k (距爆仓 357倍)
- BNB 最低权益 ~$119k (距爆仓 1192倍)
- 所有币种 Busts = 0, RiskCuts = 0
- 原因: 深熊入场+风控减半仓+冷却期 = 多层防护

## Phase 2: 定参

### 选择: champion 入场 (F1_winner2)

```yaml
entry:
  weekly_ema_200_position_lt: -0.05   # 价格低于周线EMA200 至少5%
  ema1200_cross_above_vwap1200: true  # EMA1200金叉VWAP1200
  require_momentum: true              # roc_5 > 0 AND roc_20 > 0
```

### 理由

1. **综合得分碾压**: explosive_score=115.7, 是第二名的 3.2x
2. **风险控制最优**: 最差回撤 -53.7%（其他都 ≥-60%）
3. **选择性是优势**: 3/4 币种入场（BTC 被过滤），但每笔质量极高
   - ETH: $10k→$51k (5.1x, DD -28.6%)
   - SOL: $10k→$50k (5.0x, DD -53.7%)
   - BNB: $10k→$152k (15.2x, DD -27.6%)
4. **BTC 处理**: champion 对 BTC 不触发（EMA200下方深度不足）。若必须覆盖 BTC，
   可 fallback 到 compression 入场（BTC单独 $10k→$52k）

### 拒绝的备选

| 备选 | 原因 |
|------|------|
| F0_winner1 (深熊10%+金叉+放量) | DD -67.8% 太高，放量条件过滤了部分好机会 |
| B3_ema1200_near_vwap | 纯交叉，缺少深度和动量过滤，DD -60.7% |
| C1_atr_low_ema (压缩) | 收益仅 2.56x，不到冠军一半 |

### 压缩和动量实验结果

**压缩类**: 单独使用效果中等（C1: 2.56x），因为压缩后不一定有大趋势，
       可能是盘整后继续下跌。需要配合深度过滤。

**动量类**: 单独使用效果差（E1: 1.14x），因为动量信号在深熊中频繁出现假突破。
       动量必须配合深度（确保在底部）和交叉（确保拐点确认）。

**核心洞察**: "深熊底部 + 结构拐点 + 动量确认" 三者缺一不可。
   - 缺深度 → 高位追涨杀跌
   - 缺交叉 → 抄底在半山腰
   - 缺动量 → 买了继续跌

## Promote 建议

| 条件 | 状态 |
|------|------|
| Phase 1 扫描完成 | ✅ |
| Phase 2 定参 | ✅ |
| Phase 3 分段验证 | ⬜ pending — 需 canonical 三阶段回测 |
| Phase 4 交易地图 | ✅ SOL/ETH/BNB 已生成 |
| Phase 5 Promote | ⬜ 待 Phase 3 通过后执行 |

### 待 promote 配置

```bash
# Phase 3 通过后执行
cp config/strategies/rolling_trend/archetypes/execution.yaml \
   config/strategies/rolling_trend/archetypes/execution.yaml.bak

# 将 entry 段更新为 champion
# weekly_ema_200_position_lt: -0.05
# ema1200_cross_above_vwap1200: true
# require_momentum: true
```
