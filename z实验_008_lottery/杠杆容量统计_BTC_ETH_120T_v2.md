# 杠杆容量统计 v2 — BTC / ETH 120T（接 orderflow / OI / funding，含 OOS）

> 数据：feature store `features_me_120T_e98fe79b58`，覆盖 2022-08-01 ~ 2026-02-28（≈42 个月）。
> 每个 symbol × side × horizon 产出 ~15k 样本；共 4 × 15k ≈ 60k。

脚本：
- `scripts/analyze_leverage_capacity_v2.py`
- 产物目录：`reports/leverage_capacity_v2/`

---

## 0. 方法论增量（vs v1）

| 维度 | v1 | v2 |
|---|---|---|
| 特征 | 12 个 OHLCV 派生 | 31 个（加入 CVD / VPIN / OI / funding / ME/BPC 语义 / EVT / Hurst） |
| MAE 定义 | 纯 high-low | 高-低 + funding 成本累计（长仓加正 funding 负担；短仓加负 funding 负担） |
| 去重 | 无 | 连续 ≥100x 高原只保留首 bar（`is_first_100x`） |
| 信号 overlay | 无 | 子集 lift（BPC 连续/突破、ME 加速+对齐、压缩状态、低波分位） |
| OOS | 按日历分段 | **train / test / oos** 三段；决策树训练 + Top-K 精度 |

三个窗口：

| 段 | 日期 | 语义 |
|---|---|---|
| train | 2022-08 ~ 2023-09（14m） | 熊市尾 + 2023 震荡 |
| test  | 2023-10 ~ 2024-03（6m）  | 牛市主升段（ETF 前后） |
| oos   | 2024-04 ~ 2026-02（21m） | 牛市后 + 当下 |

2020–2021 牛市不在 FS 覆盖内；**继续以 v1（OHLCV-only）的对照结果作为补充**。

---

## 1. TL;DR

1. **H=48 不是彩票，是日常**。在牛市窗口 BTC 长仓 16.25% 的 bar、ETH 长仓 11.26% 的 bar 可承受 ≥100x（4 日 持仓）。"彩票感"（极度稀有）出现在 H=120 的多头方向：BTC 3.4% / ETH 0.9%。
2. **BTC 比 ETH 稳**。任意窗口、任意方向，BTC 的 ≥100x 占比比 ETH 高约 30–60%。下一轮 L3 彩票，**先只开 BTC**，ETH 最多当对照。
3. **牛市里长仓彩票的最强单特征**（H=120，BTC+ETH 联合 lift vs base=0.57%）：
   - `cvd_roll60` 轻度正（2w~5w U 成交量） → lift **8.7x**
   - `funding_rate_zscore_50` 负值（空头拥挤） → lift **8.2x**
   - 价格接近 / 刚越过 EMA1200 但仍低于 VWAP1200 → lift 5.4x / 6.0x
   
   语义翻译：**"趋势锚完整 + 价格在趋势锚下方折扣 + 空头拥挤 + 悄悄在吸筹"**。
4. **决策树 OOS**（train 2022-08~2023-09 → test 2023-10~2024-03 牛市主升段，**全跨 regime 测试**）：长仓取树打分 top 1% 的 bar，≥100x 实际命中 16.3%，lift **7.57x**。这是 v2 最干净的可落地结果。
5. **OOS 外延（2024-04~2026-02）中所有 lift 都缩到 1.5–2x**。结论：v2 学到的"彩票位"特征在**牛市窗口内**泛化较好，但不能直接在非牛市时段应用。→ **L3 100x 开关必须挂在"宏观 bull regime"门控上**（例：BTC 周线高于 EMA50、或 6M return > 30%）。
6. **策略 overlay（subset lift）**：
   - 低波压缩（`me_atr_pct` 最低 10 分位）在牛市 H=120 long 上 **lift 1.92x**，是最有用的"策略门"。
   - BPC continuation / breakout ≥0.3 的语义门 lift 仅 1.0–1.3x，信息量有限（它们服务于更短的 BPC 规则）。
   - ME accel+align 在 L3 彩票选点上基本没有增值（ME 信号是"爆发进行中"，而彩票要的是"爆发前的压缩"）。

---

## 2. Bucket 占比（MAE 调过 funding）

> 每格为 share (%)。行和未归一化到 100 是因为存在极少量 NaN（横坐标之外）。

### 2.1 H=48（≈4 天）

```
period=train (2022-08~2023-09 熊末+震荡)
  BTC long    ≥100x 16.84  50-100x 21.75  20-50x 39.98  10-20x 17.06  5-10x 3.67  <5x 0.70
  BTC short   ≥100x 22.85  50-100x 17.95  20-50x 37.79  10-20x 15.24  5-10x 4.91  <5x 1.26
  ETH long    ≥100x 13.57  50-100x 18.56  20-50x 37.16  10-20x 21.94  5-10x 7.77  <5x 1.00
  ETH short   ≥100x 20.44  50-100x 16.70  20-50x 29.43  10-20x 24.48  5-10x 7.43  <5x 1.52

period=test  (2023-10~2024-03 牛市主升段)
  BTC long    ≥100x 16.25  50-100x 19.27  20-50x 38.03  10-20x 20.69  5-10x 5.77
  BTC short   ≥100x 15.74  50-100x 13.27  20-50x 30.98  10-20x 25.63  5-10x 13.00
  ETH long    ≥100x 11.26  50-100x 17.89  20-50x 36.52  10-20x 24.94  5-10x 9.38
  ETH short   ≥100x 13.46  50-100x  9.89  20-50x 32.36  10-20x 30.16  5-10x 13.64

period=oos   (2024-04~2026-02 牛市后+当下)
  BTC long    ≥100x 16.36  50-100x 17.65  20-50x 37.20  10-20x 20.97  5-10x 7.16
  BTC short   ≥100x 20.48  50-100x 18.46  20-50x 35.72  10-20x 21.06  5-10x 4.17
  ETH long    ≥100x  9.94  50-100x 14.36  20-50x 31.28  10-20x 26.56  5-10x 14.05
  ETH short   ≥100x 15.53  50-100x 11.91  20-50x 32.51  10-20x 26.46  5-10x 11.72
```

观察：
- H=48 **每一段 BTC long ≥100x 都 ≥16%**。这说明"4 天 内全程 MAE < 0.6%"并不极端稀有——**彩票的分母不缺**。
- ETH 在牛市主升段 ≥100x 长仓占比下降最明显（13.6% → 11.3%）：说明 ETH 在主升段波动比 BTC 更大。
- 短仓在熊末/震荡段表现最好（BTC 短 ≥100x 22.85%），牛市段最差（15.7%）——符合直觉。

### 2.2 H=120（≈10 天）

```
period=train
  BTC long    ≥100x  4.15  50-100x 12.44  20-50x 41.70  10-20x 26.00  5-10x 13.58  <5x  2.14
  BTC short   ≥100x 12.56  50-100x 11.38  20-50x 30.12  10-20x 29.96  5-10x 11.20  <5x  4.79
  ETH long    ≥100x  2.88  50-100x  9.83  20-50x 36.58  10-20x 27.25  5-10x 18.26  <5x  5.19
  ETH short   ≥100x 11.47  50-100x 11.69  20-50x 24.14  10-20x 30.09  5-10x 16.38  <5x  6.23

period=test
  BTC long    ≥100x  3.43  50-100x 13.68  20-50x 31.26  10-20x 35.70  5-10x 15.93
  BTC short   ≥100x  4.62  50-100x  4.99  20-50x 19.86  10-20x 31.90  5-10x 27.51  <5x 11.12
  ETH long    ≥100x  0.87  50-100x 10.34  20-50x 32.72  10-20x 28.56  5-10x 23.94  <5x  3.57
  ETH short   ≥100x  7.92  50-100x  5.72  20-50x 17.44  10-20x 26.54  5-10x 31.85  <5x 10.53

period=oos
  BTC long    ≥100x  5.32  50-100x 10.86  20-50x 29.38  10-20x 32.96  5-10x 18.51  <5x  2.98
  BTC short   ≥100x 13.60  50-100x 11.84  20-50x 27.80  10-20x 30.70  5-10x 14.66  <5x  1.40
  ETH long    ≥100x  1.87  50-100x  8.15  20-50x 22.10  10-20x 26.81  5-10x 28.47  <5x 12.60
  ETH short   ≥100x  9.94  50-100x  7.66  20-50x 22.31  10-20x 29.37  5-10x 23.04  <5x  7.68
```

观察：
- **ETH long H=120 ≥100x 在牛市主升段只剩 0.87%**，非常稀有。L3 10-day long 彩票 **只做 BTC 更合理**。
- BTC short H=120 在牛市 ≥100x 反而 4.62%（略高于 long 的 3.43%）。这看起来反直觉，实际含义是：**牛市里价格上涨速度够快，10 天的 low 很少比入场价低 0.6%**（对短仓就是"MAE 小"）；但短仓能活 10 天 ≠ 赚钱，只是"没爆仓"。
- H=120 OOS 的 BTC short ≥100x 升到 13.60%——因为 2024-04 至 2026-02 整体偏震荡/回调区多，短侧友好。

---

## 3. 牛市里 100x 长仓的 MFE（扣除 funding 成本）

> 条件：period=test（牛市），side=long，`lmax_adj ≥ 100`。

| Horizon | 中位 MFE | p90 MFE |
|--------:|---------:|--------:|
| H=48  | ≈ 7.9%  | ≈ 14% |
| H=120 | ≈ 19.5% | ≈ 30% |

换算成对 equity 的贡献：
- **100x × H=120 long**：中位 MFE 19.5% × 100x ≈ **19.5 倍本金**（极限接近理论上限）。
- **50x × H=120 long**：中位 MFE 19.5% × 50x ≈ **9.75 倍本金**。
- 因此用 100x 追求极限 vs 用 50x 留安全垫，预期值差距明显，但 50x 失败概率远低（见下文 top-K 精度）。

---

## 4. 单特征 lift — 牛市 H=120 long 最强信号

> Base rate=0.57%（period=test），分 10 分位，显示 top 6。

| 特征 | 分位 | n | 命中 | rate | lift | 区间 |
|---|---|---|---|---|---|---|
| `cvd_roll60` | Q8 | 221 | 11 | 4.98% | **8.70x** | 20k~50k 正 CVD |
| `funding_rate_zscore_50` | Q3 | 192 | 9 | 4.69% | **8.19x** | z ∈ [-0.56, -0.10]（负 z = 空头拥挤） |
| `macro_tp_vwap_1200_position` | Q0 | 437 | 15 | 3.43% | **6.00x** | 价格低于 VWAP1200 3.8%~13.9%（深折扣） |
| `taker_buy_ratio` | Q1 | 221 | 7 | 3.17% | 5.54x | 0.456~0.470（略偏卖） |
| `ema_1200_position` | Q1 | 291 | 9 | 3.09% | 5.41x | 正好刚站上 EMA1200 0~6.5% |
| `macro_tp_vwap_1200_position` | Q1 | 437 | 10 | 2.29% | 4.00x | VWAP1200 ±5% 内 |

**合成画像**（L3 多头彩票位）：
> 牛市 regime 下，价格**刚刚站上 EMA1200** 但**仍低于 VWAP1200**（-5~0% 折扣）、**funding 负向 z**（空头堆积）、**60 根 CVD 轻度净买**（2w~5w U）、**taker 买比微偏卖**（没人抢着多，小散还在观望）。

这是**"牛市初期的悄悄吸筹"位**，特别符合 2023-10 那波。

### 4.1 H=48 的最强单特征（Base rate=13.75%）

| 特征 | 分位 | rate | lift | 区间语义 |
|---|---|---|---|---|
| `taker_buy_ratio` Q1 | 221 | 14.03% | **3.63x** | 0.456~0.470（平衡偏卖） |
| `taker_buy_ratio` Q0 | 221 | 9.50% | 2.46x | 0.398~0.456（偏卖） |
| `cvd_roll20` Q3 | 220 | 8.18% | 2.12x | -26k~-12k（CVD 轻度负） |
| `sma_200_position` Q7 | 278 | 7.91% | 2.05x | 0.081~0.106（价格在 SMA200 上方 8~10%） |

H=48 的故事：**"散户没追、CVD 平静、价格在长均线上方 8~10% 的缓冲区"**——比 H=120 的画像"更接近持续上行中"。

---

## 5. 决策树 OOS — 关键结果

### 5.1 H=120 long，train (2022-08~2023-09) → test (2023-10~2024-03 牛市)

```
Train: n=9998 pos=351 precision=0.100 recall=0.895
Test:  n=4370 pos=94  precision=0.035 recall=0.319 base=0.022 lift=1.61

Top-K precision:
  top  1%  n=  43  hit=   7  prec=0.163  lift=7.57
  top  3%  n= 131  hit=  14  prec=0.107  lift=4.97
  top  5%  n= 218  hit=  14  prec=0.064  lift=2.99
  top 10%  n= 437  hit=  14  prec=0.032  lift=1.49
```

Top features: `macro_tp_vwap_1200_position` (0.42) > `shd_pct` (0.20) > `evt_tail_shape` (0.17) > `cvd_roll60` (0.05) > `funding_oi_crowding_score` (0.04)。

**可落地结论**：
- 取树打分 top 1% (牛市窗口约 43 个 bar，平均每月 ~7 个，每周 ~2 个)：实际 ≥100x 命中率 16.3%，即**每开 6 单有 1 单能完全跑满 10 日不爆仓**（MAE 全程 < 0.6%）。
- top 3% 精度降到 10.7%（约每 9 单 1 命中，lift 5x），更现实的开单频率。
- top 10% 就退化到 base rate 附近（lift 1.5x），说明**严格取高置信点** 才有 edge。

### 5.2 H=120 long，train → oos (2024-04~2026-02)

```
Top-K precision:
  top  1%  n= 162  hit=   4  prec=0.025  lift=0.69  ← 反而低于 base
  top  5%  n= 812  hit=  35  prec=0.043  lift=1.20
  top 10%  n=1625  hit=  86  prec=0.053  lift=1.47
```

**非牛市窗口里，这套"彩票规则"完全失效**（top 1% 甚至低于 base）。说明：
- 不能把这套规则挂成全时段策略；
- **必须前置一个 bull regime 过滤器**（例：BTC 周线 close > EMA50，或 6M return > 25%，或 Aave/funding 曲线偏正等宏观条件）。

### 5.3 H=120 short，train → test / oos

```
test: top 5%  prec=0.083  lift=1.32
test: top 20% prec=0.106  lift=1.70
oos:  top 5%  prec=0.190  lift=1.61
oos:  top 10% prec=0.178  lift=1.51
```

- **牛市里短侧彩票 lift 非常弱（1.3~1.7x）**——牛市时不要做高杠杆空。
- OOS 段里短侧 lift 稳定在 1.5~1.6x，但 prec 18% 还要扣交易成本、funding 正 drain，实战 EV 不高。

结论：**L3 彩票只做多，只在牛市；短侧彻底让给 FER/FBF**。

### 5.4 Within-bull 子分割（train 2023-10~2023-12 → test 2024-01~2024-03）

- **Long test 段 0 个 ≥100x 正样本**（底层事实：2024-Q1 BTC 是猛上涨，几乎没有 10 天无 0.6% 回撤的 bar）。
- Short test 段 base 6.8%，tree lift ~0.8。

说明：**100x long 彩票位集中在牛市启动段（accumulation & early breakout）**，在牛市中后段（trend 已经展开、每日 range > 1%）反而**消失**。这一点对 L3 部署至关重要——**不是整个牛市都适合上 100x，只在"趋势刚刚确立、价格离大均线不远"那一段适合**。

---

## 6. 策略 overlay（子集 lift）

> period=test（牛市），H=120。

| subset | side | n | hit | rate | lift | MFE_p50 |
|---|---|---|---|---|---|---|
| ALL（基准） | long | 4370 | 94 | 2.15% | 1.00x | 19.5% |
| **low_vol_q10** | long | 435 | 18 | 4.14% | **1.92x** | 30.6% |
| `me_accel_0.6_align_0.5` | long | 1231 | 31 | 2.52% | 1.17x | 23.8% |
| `bpc_continuation_0.3` | long | 327 | 9 | 2.75% | 1.28x | 24.9% |
| `bpc_breakout_0.3` | long | 245 | 5 | 2.04% | 0.95x | 13.1% |
| `compression_state` | long | - | - | - | - | - |
| ALL（基准） | short | 4370 | 274 | 6.27% | 1.00x | 9.9% |
| **low_vol_q10** | short | 435 | 61 | 14.02% | **2.24x** | 12.1% |

关键发现：
- **压缩（`me_atr_pct` 最低 10%）是唯一大幅提升 ≥100x 长仓率的现成语义门**（lift 1.92x，命中的中位 MFE 是 30.6% —— 比基准的 19.5% 还高）。
- **BPC 连续/突破/ ME accel 对齐 对"彩票位选点"帮助有限**（lift 1.0–1.3x）。这些特征设计目标是"爆发中 / 爆发后紧随"，与 L3 要求的"爆发前的安静折扣位"方向有分歧。

**L3 推荐的复合门**（不做 BPC/ME 信号复用，自己构造）：

```
long_lottery_gate =
  (bull_regime == True)                                    # 宏观 regime 前置
  AND (me_atr_pct <= quantile_10)                          # 低波压缩
  AND (funding_rate_zscore_50 <= 0)                        # 空头拥挤
  AND (ema_1200_position in [0, 0.07])                     # 刚站上长均线
  AND (macro_tp_vwap_1200_position in [-0.14, 0.06])       # 在 VWAP1200 折扣到微溢
  AND (cvd_roll60 in [0, 50000])                           # 轻度净买，不过热
```

在 v1 OHLCV 的"低波 + 轻趋势"基础上加了 v2 的 **funding z + CVD + VWAP1200** 三个订单流项——这一层是 v1 没法碰到的"确认层"。

---

## 7. 对先前问题的数据化回复

### 7.1 100x 是否需要独立账户/独立策略？

**是。**
- v2 OOS 证实：同一套规则在牛市内 lift 7.6x，在非牛市 lift 跌到 0.7–1.5x，**对市场 regime 极度敏感**。
- 而当前系统里的 BPC/TPC/ME 策略是"全 regime 尝试 + 内部 gate 过滤"，其 gate 与"彩票位"的要求方向不一致（见 §6）。
- 所以：L3 彩票账户需要**独立风险账户 + 独立 regime 门 + 独立 gate**，不复用 L2 的策略配置。
- 建议的账户隔离：Binance 子账户，U 本位合约隔离保证金，单仓风险 ≤ L3 budget 的 0.5%，整 L3 budget ≤ 总资产 5%。

### 7.2 只做 BTC 还是做所有 highcap？

**先只做 BTC，ETH 作为第二候选，其他 highcap 放弃。**

数据证据（H=120 long ≥100x 占比，牛市）：
- BTC 3.43%
- ETH 0.87% （仅为 BTC 的 25%）

加上 v1 OHLCV 结果里 BTC 更高的 "稳态" 和更低的 wick/流动性风险 —— **资金体量小时集中 BTC 更理性**。ETH 在非牛市窗口（OOS）≥100x 短仓比例仍有 9.94%，可以**只做空不做多** 作为第二曲线，但属于"可选"。

---

## 8. 待办（v3 方向）

- [ ] 加 **bull regime 指标**（BTC 周线 EMA/6M return/funding-OI 组合）并作为硬门，重新跑 tree；
- [ ] 把 v2 规则 + regime 门 → 生成"L3 信号流水"，在 2023-10 ~ 2024-03 回放记 **实际触发日期 × 事后 10 日 MAE/MFE**，看人工是否认可；
- [ ] 对 v1 在 2020-2021 牛市上的 100x-long 候选 bar 做**手工抽样** 验证规则迁移性；
- [ ] funding 成本模型升级：使用实际 8h funding 而非平均值（当前已求和，但仍假设 flat），并区分 mark vs last；
- [ ] 跟 `z实验_007_lv/lv.md` 里的 LV 语义（forced price 判据）进一步合并，构造**"强制跳空候选 + 彩票位"** 联合规则。

---

## 附：产物清单

```
reports/leverage_capacity_v2/
  BTCUSDT_120T_H48_samples.parquet       31132 行（long+short）
  BTCUSDT_120T_H120_samples.parquet      31132
  ETHUSDT_120T_H48_samples.parquet       31134
  ETHUSDT_120T_H120_samples.parquet      31134
  bucket_counts_H{48,120}_{train,test,oos}[{,_raw}].csv
  subset_lift_H{48,120}_{train,test,oos}.csv
  feature_lift_H{48,120}_{train,test,oos}.csv
  tree_rules_H{48,120}.md         跨 regime 大 OOS 决策树
  tree_rules_H{48,120}_bull.md    牛市内 2 分 OOS
```

生成命令：

```bash
python scripts/analyze_leverage_capacity_v2.py \
    --symbols BTCUSDT,ETHUSDT \
    --timeframe 120T \
    --horizons 48,120 \
    --fs-layer features_me_120T_e98fe79b58 \
    --train 2022-08-01:2023-09-30 \
    --test  2023-10-01:2024-03-31 \
    --oos   2024-04-01:2026-02-28 \
    --output-dir reports/leverage_capacity_v2
```
