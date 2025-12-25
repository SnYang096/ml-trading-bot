# SR Reversal 实验流程（可复现/可对比/可落地）

## 1. 为什么要做这套流程（本次工作的原因）
在 SR Reversal（`sr_reversal_*`）系列策略上，我们遇到两个核心问题：

- **不可复现**：同样的配置、同样的数据窗口，重复运行 `mlbot diagnose model-comparison`，得到的 `Sharpe/return/trades` 等指标会显著变化，甚至出现训练样本数变化（这会直接让“哪个方案更好”的结论失效）。
- **高方差**：即使修复了单次运行的随机性，策略的收益/夏普仍会随着 seed 波动；如果只看某一次 run 的结果，会把噪声当成优势（或把优势误判为无效）。

因此，本次工作的重点不是“再调一个更好看的 Sharpe”，而是先建立**一套统计上可信、能复现、能逐步迭代**的实验协议（Experiment Protocol）。

---

## 2. 最终目标（对齐）
**用可复现 + 多 seed 统计的方法，确定在统计意义上更优的 SR Reversal 方案**，并把它固化为可长期迭代的主线配置。

我们希望得到的结论是：

- **可复现**：同 seed、同配置，多次运行输出一致。
- **可比较**：多 seed 下输出 mean/std/min/max（至少 5 seeds；推荐 10+）。
- **可落地**：明确 1–2 套“均值更好且方差可接受”的赢家配置，作为后续加入 SR/权重/ticks 的主线。

---

## 3. 可复现改造（Deterministic Mode）
为了让“同 seed 重复跑”结果一致，我们做了以下改造：

- `scripts/train_strategy_pipeline.py` 新增 `--deterministic`
  - 强制设置：
    - `MLBOT_DETERMINISTIC=1`
    - `OMP_NUM_THREADS=1`
    - `OPENBLAS_NUM_THREADS=1`
    - `MKL_NUM_THREADS=1`
    - `NUMEXPR_NUM_THREADS=1`
    - `VECLIB_MAXIMUM_THREADS=1`
- `src/time_series_model/strategies/models/strategy_trainer.py`
  - LightGBM 在 `MLBOT_DETERMINISTIC=1` 时强制 `num_threads=1`
  - 修复：避免在 `_train_lightgbm` 中 `pop("n_estimators")` 修改共享的 `model_params`（会造成 fold 行为不稳定）
- `mlbot diagnose model-comparison`
  - 多策略对比模式会调用 `train_strategy_pipeline.py`，并默认加上 `--deterministic`

---

## 4. 评估协议（Evaluation Protocol）
为了保证各策略对比公平，先固定以下参数（除非明确实验目的为“调参”）：

- **数据窗口**：例如 `2023-01-01 ~ 2025-10-31`
- **timeframe**：例如 `240T`
- **test-size**：例如 `0.3`
- **seed 列表**：推荐 `1..5`（最低），更稳健用 `1..10`
- **回测退出**：只用 RR + trailing（如果策略配置如此），避免引入额外的“时间退出噪声”

我们只在一个维度上做 A/B（例如只改“特征组”，其他全固定），避免多因素耦合导致结论不清晰。

---

## 5. 实验流程（推荐顺序）

### 5.1 Step A：MVP（3 个 K 线特征）先打通并验证稳定
目的：建立“最小可用、可复现”的基线，然后再逐步加回特征组。

已新增 MVP 策略：
- `config/strategies/sr_reversal_long_mvp`（binary）
- `config/strategies/sr_reversal_rr_reg_long_mvp`（regression）

两者 `requested_features` 仅保留：
- `macd_f`, `rsi_f`, `sma_200_f`

单次对比（同 seed 可复现）：
```bash
mlbot diagnose model-comparison \
  --strategy-config sr_reversal_long_mvp,sr_reversal_rr_reg_long_mvp \
  --symbol BTCUSDT --timeframe 240T \
  --start-date 2023-01-01 --end-date 2025-10-31 \
  --test-size 0.3 --seed 42 \
  --output-dir results/model_comparison/mvp_baseline \
  --no-docker
```

### 5.2 Step B：分组回填（Staged Add-back）
目的：避免“一次加太多特征导致噪声上升”，并定位“哪一组开始有效/开始拉跨”。

已新增 kline-only(+volume) 策略：
- `sr_reversal_long_kline_plus`（binary，kline+volume；入场建议用 `entry_quantile=q90 + cross`）
- `sr_reversal_rr_reg_long_kline_plus`（reg，kline+volume）

回归分阶段（示例）：
- `sr_reversal_rr_reg_long_kline_core`
- `sr_reversal_rr_reg_long_kline_core_plus`
- `sr_reversal_rr_reg_long_kline_plus`

### 5.3 Step C：多 seed 统计（均值/方差）——用统计结论替代单次 run
使用脚本：
- `scripts/run_model_comparison_seed_sweep.py`

例：跑 5 个 seeds，并输出 summary（mean/std/min/max）：
```bash
python3 scripts/run_model_comparison_seed_sweep.py \
  --strategies sr_reversal_long_mvp,sr_reversal_rr_reg_long_mvp \
  --symbol BTCUSDT --timeframe 240T \
  --start-date 2023-01-01 --end-date 2025-10-31 \
  --test-size 0.3 \
  --seeds 1,2,3,4,5 \
  --output-dir results/model_comparison/seed_sweep_mvp
```

输出文件：
- `seed_sweep_all_rows.csv`：每个 seed 的原始结果
- `seed_sweep_summary.csv`：按策略聚合的 mean/std/min/max

### 5.4 Step D：只动“入场规则”，把 trades 稳定下来（降低回测噪声）
目的：入场阈值的小变化会放大 trades 波动，导致 Sharpe 不稳定；先用更稳的入场机制。

建议默认：
- **binary**：`entry_quantile: 0.9` + `entry_mode: cross`
- **regression**：`top_quantile: 0.1` + `entry_mode: cross`

随后只扫一个维度（例如回归扫 `top_quantile=0.05/0.08/0.1`）并跑 seed sweep。

---

## 6. 何时回到 SR/weights/ticks 主线？
当 kline-only 的赢家配置满足：
- mean 明显为正（return%、Sharpe）
- std 在可接受范围（不会“均值小、方差巨大”）
- trades 达到足够数量（否则 Sharpe 不稳定）

再把 SR/weights/ticks **逐个加回**（一次只加一个因素），每次都做 seed sweep，才能判断“是否真实提升”。

---

## 7. 经验法则（避免常见误判）
- **不要用单次 Sharpe 做结论**：必须看多 seed 的 mean/std。
- **CV/IC 好 ≠ 一定赚钱**：尤其在交易频率低、费用/滑点显著时。
- **先稳定回测 trades**：不稳定的 trades 会让收益分布漂移。
- **逐组加回特征**：一次性加很多，很难定位问题源头。

---

## 8. 当前已落地的实验配置（Checklist）
为了让实验“可复现 + 可追踪”，我们把关键阶段都落成独立的 `config/strategies/<name>/` 目录（避免一份配置频繁改来改去导致结果不可追溯）。

### 8.1 MVP（3 特征）
- `sr_reversal_long_mvp`：binary，`macd_f/rsi_f/sma_200_f`
- `sr_reversal_rr_reg_long_mvp`：regression，`macd_f/rsi_f/sma_200_f`

### 8.2 kline-only(+volume)（不含 SR/ticks/orderflow）
- `sr_reversal_long_kline_plus`：binary，kline+volume；入场建议 `entry_quantile=q90 + entry_mode=cross`
- `sr_reversal_rr_reg_long_kline_plus`：regression，kline+volume；入场 `top_quantile=0.1 + cross`

### 8.3 回归分阶段（用于定位“哪一组开始有效/开始拉跨”）
- `sr_reversal_rr_reg_long_kline_core`：core 指标小集合（6 个）
- `sr_reversal_rr_reg_long_kline_core_plus`：core + bbands/trend/price action（仍不含 SR/ticks）

---

## 9. 下一步推荐实验顺序（严格一次只动一个因素）
### 9.1 固定协议（不要动）
- window/timeframe/test-size/seed 列表固定
- `--deterministic` 固定开启

### 9.2 回归：先把入场频率调到“更稳”
在 `kline_plus` 上只扫一个参数：
- `top_quantile ∈ {0.05, 0.08, 0.10}`

目标：
- trades 不要太少（否则 Sharpe 不稳）
- mean Sharpe/return% 提升，同时 std 降低或不恶化

### 9.3 二分类：验证 `entry_quantile + cross` 是否显著降低方差
对比 `sr_reversal_long_mvp` vs `sr_reversal_long_kline_plus`：
- seeds=1..5（或 10）跑完后看：
  - trades std 是否显著下降
  - Sharpe/return% 的 mean/std 是否改善

### 9.4 选出赢家后再回到 SR/weights/ticks 主线
按顺序逐个加回（每次都 seed sweep）：
1) `sr_fuse`
2) SR filter（label 或 backtest 侧）
3) sample weights
4) tick/orderflow features

---

## 10. 实验结论记录（随实验追加，避免“口口相传”）

### 10.1 sr_fuse（回测侧）在公平对照下：当前为负面
我们做过一次容易踩坑的情况：**sr_fuse 依赖 `dist_to_nearest_sr`**，如果策略特征里没有生成该列，回测会日志提示并跳过（不会生效）。

为了“只测 fuse 开关”，我们做了**公平对照**：
- 控制组：`sr_reversal_rr_reg_long_kline_core_plus_q05_srfeats`（同 SR 距离特征，但 `sr_fuse.enabled=false`）
- 实验组：`sr_reversal_rr_reg_long_kline_core_plus_q05_srfuse`（同 SR 距离特征，`sr_fuse.enabled=true, max_dist_atr=3.0`）
- seeds=1..5

结果（mean）：
- **fuse OFF**：return%≈ **2.06**，Sharpe≈ **0.35**，DD%≈ **6.44**，trades≈ **25.8**
- **fuse ON**：return%≈ **-3.06**，Sharpe≈ **-0.44**，DD%≈ **9.44**，trades≈ **28.6**

结论：在该基线与窗口下，`sr_fuse.max_dist_atr=3.0` **伤害收益与 Sharpe**，暂不作为主线默认。

### 10.2 SR filter（标签侧）在回归主线：显著正面
在相同特征集（包含 SR 距离特征）的前提下，我们把 **SR 过滤放到标签生成**：
- baseline：`sr_reversal_rr_reg_long_kline_core_plus_q05_srfeats`（全量标签）
- SR-filter label：`sr_reversal_rr_reg_long_kline_core_plus_q05_srfeats_srfilter`（`dist_to_nearest_sr` 归一化后 ≤ `1.5 ATR` 的样本才保留）
- seeds=1..5

关键变化：
- 训练样本数从 **4274 → 2636**（SR 过滤生效）
- mean（SR-filter label）：
  - return%≈ **7.12**（std≈1.16）
  - Sharpe≈ **1.66**（std≈0.41）
  - DD%≈ **3.20**（std≈1.08）
  - trades≈ **16.8**

结论：在当前回归主线上，“标签侧 SR 过滤”带来**更稳定的正期望**（均值更高、方差更小、回撤更低），是目前最值得继续推进的方向。

### 10.3 SR filter 强度 sweep（dist_atr_mult）：推荐 1.5
我们对同一套回归主线（`kline_core_plus_q05` + SR 距离特征）做了 label-side SR filter 的强度 sweep：
- `dist_atr_mult ∈ {1.0, 1.2, 1.5, 2.0}`
- seeds=1..5

汇总（mean）：
- **1.0**：return%≈ **4.48**，Sharpe≈ **1.43**，DD%≈ **2.75**，trades≈ **11.0**（交易偏少，方差偏大）
- **1.2**：return%≈ **4.52**，Sharpe≈ **0.92**，DD%≈ **3.31**，trades≈ **11.6**（不稳定，收益方差大）
- **1.5**：return%≈ **6.52**，Sharpe≈ **1.47**，DD%≈ **3.12**，trades≈ **17.0**（综合最好：均值高、方差更可控、交易数更健康）
- **2.0**：return%≈ **-2.70**，Sharpe≈ **-0.31**，DD%≈ **10.41**，trades≈ **28.2**（过滤太宽，效果变差）

结论：当前窗口/特征下，**`dist_atr_mult=1.5` 是最优折中点**，建议作为回归 SR-filter 标签的默认主线参数。

### 10.4 Sample Weights（训练侧）在赢家基线上：显著正面
在“回归赢家基线”（同特征、同 SR-filter 标签：`dist_atr_mult=1.5`）上，我们做了 sample weights 的公平对照：

- unweighted：`sr_reversal_rr_reg_long_kline_core_plus_q05_srfeats_srfilter`
- weighted：`sr_reversal_rr_reg_long_kline_core_plus_q05_srfeats_srfilter_weighted`
  - 标签函数：`compute_sr_reversal_rr_continuous_label_with_weights`
  - `weight_strategy=result_based_rr`
  - 训练侧：LightGBM 使用 `weight_col=sample_weight`
- seeds=1..5

结果（mean）：
- **unweighted**：return%≈ **6.87**（std≈1.85），Sharpe≈ **1.60**（std≈0.30），DD%≈ **2.89**，trades≈ **16.4**
- **weighted**：return%≈ **12.30**（std≈4.02），Sharpe≈ **2.03**（std≈0.59），DD%≈ **4.00**，trades≈ **14.8**

结论：在当前窗口/特征/标签过滤下，**sample weights 能显著提升收益与 Sharpe**，代价是 DD% 与波动略上升；目前可以将 weighted 作为回归主线候选，并继续做权重参数微调（例如 `loss_weight/high_rr_boost`）。

### 10.5 Weights 参数 sweep（loss_weight）：推荐 0.05
在 weighted 主线（同 SR-filter=1.5，同特征、同回测）上，我们只扫一个参数：
- `loss_weight ∈ {0.05, 0.10, 0.15}`（其余保持不变）
- seeds=1..5

结果（mean）：
- **loss_weight=0.05**：return%≈ **13.46**（std≈1.88），Sharpe≈ **2.16**（std≈0.22），DD%≈ **4.13**，trades≈ **15.8**
- **loss_weight=0.10**：return%≈ **8.76**（std≈3.44），Sharpe≈ **1.47**（std≈0.52），DD%≈ **4.99**，trades≈ **16.6**
- **loss_weight=0.15**：return%≈ **11.09**（std≈5.27），Sharpe≈ **1.80**（std≈0.74），DD%≈ **4.74**，trades≈ **14.8**

结论：当前窗口/配置下，**`loss_weight=0.05` 同时给出最高 Sharpe 均值 + 最低 Sharpe 方差**，是最稳的选择；建议将 weighted 主线的 `loss_weight` 设为 `0.05`。

### 10.6 Weights 参数 sweep（high_rr_boost）：推荐 1.5
`high_rr_boost` 的含义：当样本的未来 RR 达到 `high_rr_threshold`（默认 2.0）时，它会把该样本的权重按比例放大（例如 `log(1+RR) * high_rr_boost`），从而让模型训练更偏向“高回报形态”的拟合（训练侧加权，不影响推理阶段）。

我们在 fixed `loss_weight=0.05` 下做了 sweep：
- `high_rr_boost ∈ {1.5, 2.0, 3.0}`
- seeds=1..5

结果（mean）：
- **1.5**：return%≈ **14.57**（std≈3.27），Sharpe≈ **2.14**（std≈0.28），DD%≈ **5.45**，trades≈ **18.4**
- **2.0**：return%≈ **10.61**（std≈3.14），Sharpe≈ **1.70**（std≈0.45），DD%≈ **4.72**，trades≈ **16.8**
- **3.0**：return%≈ **6.63**（std≈2.92），Sharpe≈ **1.02**（std≈0.40），DD%≈ **6.08**，trades≈ **19.2**

结论：当前窗口/配置下，**`high_rr_boost=1.5` 给出最高 Sharpe 均值且方差最小**，建议作为默认主线参数。

### 10.7 回归主线（Mainline）最终配置与确认结果
我们将“当前最佳组合”固化为一个明确的主线策略目录，便于后续复用与滚动训练：

- **主线策略**：`config/strategies/sr_reversal_rr_reg_long_mainline`
- **关键设定**：
  - 标签侧 SR filter：`dist_to_sr_col=dist_to_nearest_sr`，`dist_atr_mult=1.5`
  - 样本权重：`weight_strategy=result_based_rr`，`loss_weight=0.05`，`high_rr_boost=1.5`
  - 回测入场：`top_quantile=0.05`，`entry_mode=cross`
  - `sr_fuse`：关闭（根据 10.1 的公平对照结论）

确认对比（seeds=1..5）：
- baseline（unweighted + SR-filter）：return% mean≈ **7.10**（std≈2.82），Sharpe mean≈ **1.51**（std≈0.40），DD% mean≈ **3.04**
- **mainline**：return% mean≈ **15.01**（std≈1.85），Sharpe mean≈ **2.29**（std≈0.28），DD% mean≈ **4.12**

结论：在该窗口与协议下，**mainline 在收益与 Sharpe 的均值上显著更优，且方差更小**；代价是 DD% 略有上升。


