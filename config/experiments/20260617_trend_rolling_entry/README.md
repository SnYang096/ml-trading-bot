# Trend Rolling 入场扫描实验（2026-06-17）

## Phase checklist

| Phase        | 状态 | 说明                                                      |
| ------------ | ---- | --------------------------------------------------------- |
| 0 (特征)     | ✅    | 120T OHLC + 自计算指标 (wEMA200, EMA1200, VWAP1200, ATR%) |
| 1 (扫描)     | ✅    | `entry_scan.csv` — 24变体×4币种完整排名                   |
| 2 (定参)     | ✅    | 见 `DECISION.md`                                          |
| 3 (分段回测) | ⬜    | pending — 需要 canonical 三阶段 segment_matrix            |
| 4 (交易地图) | ✅    | `results/trend_rolling/trading_map_*.png`                 |
| 5 (promote)  | ⬜    | pending                                                   |

## 实验目标

为 **趋势滚仓策略 (rolling_trend)** 找到最优入场条件组合，
实现「深熊底部精准抄底 + 杠杆滚仓 + 阶梯止盈」的爆炸利润。

## 方法论

不同于 B/C 系统的 per-trade 入场优化（IC scan → label plateau → gate calibrate），
趋势滚仓是 **portfolio-level 策略**，入场信号是罕见的（每币种1-3次/4年），
无法用 IC 扫描（需要大量样本）。替代方法：

1. **入场变体矩阵扫描**: 定义 24 种入场条件变体，每种在 4 币种 × 4 年数据上
   运行完整滚仓模拟，按 Calmar × Return × Entries 综合排名
2. **入场预设对比**: champion / base / deep / compression 四预设横向对比
3. **分段验证** (Phase 3): 按 market_segment 三阶段验证稳健性

## Phase 1: 入场变体扫描

```bash
python scripts/trend_rolling_entry_scan.py \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT \
  --start 2022-01-01 --end 2026-06-01 \
  --top 24
```

产物: `results/trend_rolling/entry_scan.csv`

### 扫描结果摘要

| 排名 | 变体                 | 类别 | 总收益   | 倍数  | 最差DD | 入场数 |
| ---- | -------------------- | ---- | -------- | ----- | ------ | ------ |
| 🥇    | F1_winner2           | 组合 | $253,313 | 6.58x | -53.7% | 3      |
| 🥈    | F0_winner1           | 组合 | $211,355 | 5.28x | -67.8% | 4      |
| 🥉    | B3_ema1200_near_vwap | 交叉 | $169,257 | 4.23x | -60.7% | 4      |
| 4    | C1_atr_low_ema       | 压缩 | $102,347 | 2.56x | -64.2% | 4      |

### 关键发现

- **单用深度/量能/动量都弱** (0.6-1.1x)，必须组合使用
- **单用压缩中等** (C1: 2.56x)，仅次于组合类
- **冠军 = 深度 + 交叉 + 动量**：三重过滤 → 爆炸利润
- **BTC 冠军入场为0次**：BTC深熊程度不够(-5%阈值太严)，但对ETH/SOL/BNB精准

## Phase 3: 分段验证 (pending)

```bash
# 待 segment_matrix 配置后运行
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/20260617_trend_rolling_entry/phase3_grid.yaml
```

## 文件清单

| 文件                                         | 说明                            |
| -------------------------------------------- | ------------------------------- |
| `README.md`                                  | 本文件                          |
| `DECISION.md`                                | 定参决策 + promote 建议         |
| `entry_variants.yaml`                        | 24 种入场变体定义               |
| `phase3_grid.yaml`                           | Phase 3 分段回测 grid (pending) |
| `../../results/trend_rolling/entry_scan.csv` | Phase 1 扫描结果                |
