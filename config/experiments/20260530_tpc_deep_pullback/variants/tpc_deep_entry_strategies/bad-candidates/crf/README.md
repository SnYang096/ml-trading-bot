# CRF — Consolidation Range Fade (Bad Candidate)

CRF 原始假设是在已确认的盘整盒子内做双向均值回归：

- 价格接近盒子下沿：`box_pos <= edge`，做多，等待回归。
- 价格接近盒子上沿：`box_pos >= 1 - edge`，做空，等待回归。
- 目标可以是 ATR 短打、盒子中线或盒子对边。

该假设经过多轮实盘式事件回测和离线诊断后，暂时不再作为生产候选推进，因此移动到 `bad-candidates`。

## 为什么降级

### 1. Box 只能做 regime filter，不能稳定给方向

诊断显示，`box_pos` 在盒子上下沿给出的方向信号不稳定：

- 当前 120 box 中，`edge_chop + box_opposite` 几乎没有优势。
- 严格 120 box 中，等对边仍为负。
- 严格 240 大 box 中，等对边在总体上转正，但收益高度集中在特定年份和方向。

这说明“在 box 下沿做多、上沿做空”不是稳定 alpha。Box 更像一个市场状态过滤器，而不是可靠方向来源。

### 2. 双向 range fade 不稳，尤其是 SHORT 侧

固定 box 诊断结果：

```text
current_120 + edge_chop + box_opposite:
  n=422, sum_r=+3.02R, mean_r=+0.007

strict_120 + edge_chop + box_opposite:
  n=117, sum_r=-1.26R, mean_r=-0.011

strict_240 + edge_chop + box_opposite:
  n=207, sum_r=+47.15R, mean_r=+0.228
```

`strict_240` 看似有优势，但拆开后并不稳定：

```text
strict_240 + edge_chop + box_opposite, by year:
  2022: +2.69R
  2023: +48.15R
  2024: -4.56R
  2025: +10.04R
  2026Q1: -9.17R

strict_240 + edge_chop + box_opposite, by side:
  LONG:  +62.07R
  SHORT: -14.92R
```

收益主要来自 `2023 + LONG`，不是双向稳定收益。

### 3. 2026Q1 熊市里，做空也没有稳定优势

如果 CRF 真是“熊市做空、牛市做多”的 box 方向策略，那么 2026Q1 应该显示 SHORT 优势。但诊断相反：

```text
strict_240 + edge_chop + box_opposite, 2026Q1:
  ALL:   n=20, sum_r=-9.17R
  SHORT: n=5,  sum_r=-5.00R, win=0%

strict_240 + edge_only + box_opposite, 2026Q1:
  SHORT: n=28, sum_r=-7.75R
```

部分 `current_120 + edge_only + atr_fixed` 的 SHORT 在 2026Q1 为正，但那更像短周期波动/顺势短打，不再是“固定 box 对边回归”。

### 4. 如果加入大势方向，CRF 会退化成趋势策略

若把 CRF 改成：

- 大势向上，只做下沿多；
- 大势向下，只做上沿空；
- 或用 BTC/ETH/SOL 合成趋势过滤方向；

那么策略本质就不再是 consolidation range fade，而是“box/chop regime 下的趋势过滤入场”。这可能是一个新策略，但不应该继续叫 CRF。

更合适的名字可能是：

- `box_pullback`
- `chop_trend`
- `macro_box_trend`
- `chop_edge_scalp`

这些候选需要重新和 BPC/TPC/ME/SRB 做 overlap 和边际贡献分析。

## 与 chop_grid 的关系

无方向的 box/chop 区域更适合 `chop_grid`：

- CRF 必须赌方向，统计上不稳。
- `chop_grid` 不预测方向，而是在 chop regime 中用中性网格收割来回波动。
- 趋势出现时，`chop_grid` 通过 regime exit 退出库存。

因此，当前资源优先投入 `chop_grid` 和已有趋势策略，而不是继续调 CRF。

## 当前状态

- `status`: bad-candidate
- 保留配置与诊断脚本，便于复现历史结论。
- 不再运行 CRF fast/rolling pipeline。
- 若未来重新研究，应作为新策略命名和验证，而不是沿用 CRF 语义。

## 参考输出

主要诊断输出：

```text
results/crf_fixed_box_diag/current_120
results/crf_fixed_box_diag/strict_120
results/crf_fixed_box_diag/strict_240
```

相关脚本：

```text
scripts/diagnose_crf_edge.py
```
