# CRF / CBC 结构可行性诊断（BTC 2024-05~10）

## 目标

为两个新策略做数据支撑：
- **CRF** (Consolidation Range Fade)：盘整箱内部高抛低吸（双向短线）。
- **CBC** (Consolidation Breakout Continuation)：盘整结束后顺原趋势吃第二段。

结构模板：`强趋势 peak → decay 衰竭腿 → box 盘整 → break / timeout`。

## 归档动作（已完成）

- `config/strategies/{fbf, fbf_strict, fbf_exp_trail, msr, msr_exp_trail}` → `config/strategies/bad-candidates/`
- 14 个 `prod_train_pipeline_2h_*fbf*|*msr*|*fer*|*lottery*.yaml` → `config/strategies/bad-candidates/pipelines/`
- `live/highcap/config/strategies/{fer, lv}` → `config/strategies/bad-candidates/live_legacy/`
- `config/strategies/` 现在只剩 `bpc / me / srb / tpc`（活跃策略）。
- `live/highcap/config/strategies/` 只剩 `bpc / me`（实盘）。

## 诊断结果

脚本：`scripts/diag_consolidation_structure.py`（2H, BTC 2024-05-01~10-31）。

### Step 1: trend peak 命中率

同时满足「close 接近 60 根 2H 滚动高」「close > EMA1200（或 close > close[-200]）」「trend_r2_20 ≥ 0.35」的 bar：**92 个候选 peak**（在 2209 根 2H 中）。peak 侧语义是 OK 的。

### Step 2: decay 腿（peak 后的最大回撤）

| 窗口 | n | p50 | p75 | p90 | max |
|---|---|---|---|---|---|
| 30 bar  (2.5天) | 92 | 1.6% | 2.9% | 3.6% | **7.1%** |
| 60 bar  (5天)  | 92 | 3.1% | 6.3% | 7.8% | 11.8% |
| 120 bar (10天) | 92 | 4.3% | 7.9% | 10.6% | 26.2% |
| 240 bar (20天) | 92 | 8.3% | 16.7% | 21.8% | 26.2% |

**结论**：
- 用户初定「2.5 天内跌 8%」在 2024 BTC 牛市**零命中**——市场没有那么快的衰竭。
- 想维持 8% 深度，窗口得放到 10 天 (120 bar)；p75 才 7.9%。
- 想维持 2.5 天窗口，深度得降到 ~3%（p90）。

### Step 3: box 阶段（放宽 decay 后的盘整）

| 方案 | decay_max | decay_pct | events | outcome |
|---|---|---|---|---|
| A | 60 bar | 4% | **3** | 全 timeout |
| B | 120 bar | 8% | **2** | 全 timeout |

且全部触达 720 根 max_len 上限，box_width 高达 **29–34%**，touches_hi/lo 数十次。

**问题本质**：`tol = max(1 ATR, 2%)` ≈ 双边 ±2.5~4%，加 max_len=720，把 2024 BTC 5–10 月宏观宽震当成了**一个超级大盒子**。这不是"盘整"，是整段趋势暂停。

### 结论

1. **用户心目中的"强趋势 → 快速衰竭 → 紧凑盘整 → 第二段趋势"结构，在 2024 BTC 2H 上的原始形态稀少。** 要么放宽 decay（变成慢速衰竭），要么收紧 box（强制拆出紧凑段）。
2. **CRF 的"在 box 内高抛低吸"需要 box_width 合理**（经验上 < 10% 才像盘整）。当前 2% tol + 720 max_len 会吞掉整个宏观震荡。
3. **CBC 目前全部 timeout**——但这主要是因为 box_len 撞到 max_len 上限，look_fwd=120 还在 box 内。需要先解决 box 识别。

## 下一步建议（待确认）

**两条修正路径：**

- **路径 1（紧 box）：** 把 `max_len` 从 720 降到 120（2H，10 天），`tol_pct` 降到 0.015（1.5%）。语义是"10 天内收在 3% 宽窄通道"，这样才像交易员说的"盘整"。
- **路径 2（换数据窗）：** 在 2024 单边（1–4 月 或 11–12 月快速上涨段）而不是 5–10 月宽震段上扫，趋势 + 衰竭 + 小盒子的结构会更明显。

我建议**两条都跑一次**再下判断，避免被特定市况带跑。在此之前不急着写 `crf / cbc` 策略目录。

## 产物

- `scripts/diag_consolidation_structure.py`（CLI：`--symbol --start --end --decay-max --decay-pct --min-len --max-len --tol-atr --tol-pct`）
- `reports/consolidation_btc_2024_A.csv`（decay 60/4%）
- `reports/consolidation_btc_2024_B.csv`（decay 120/8%）
