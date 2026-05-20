# FBF & MSR 止盈/追踪实验 · 2026-04-22

**背景**：诊断指出 FBF / MSR 赢单全部精确卡在 `target_r` 封顶，
未能吃到趋势尾部（FBF 最大赢单 1.98R / MSR 最大赢单 2.48R）。
本实验在 baseline 入场不变的前提下，用 2H bar 重放比较不同 exit 策略，
定量衡量"去掉固定 TP、改用 trail"能多吃多少 R。

## 方法

- 入场数据：baseline rolling_sim 的原始 trades（FBF 04-16 fatter-TP：272 笔；MSR 04-22 clean L3：23 笔）
- 价格路径：每笔 trade 的 feature_store/120T 2H bar（含 OHLC + ATR）
- 重放引擎：`scripts/simulate_exec_trail.py`，bar-by-bar 套用 position_logic 顺序
  （time_stop → breakeven → HWM/LWM → activation trail → SL → TP）
- 精度限制：2H 颗粒比实际 event_backtest 的 1min 粗；
  baseline 自校准显示 FBF +33R（1min）→ +55R（2H-resim，系统性偏乐观 ~69%），
  MSR +7.4R（1min）→ +4.5R（2H-resim，微悲观）；**方向性结论可信，绝对值不可直接套用**。
- 重放时用同一 2H bar 引擎跑 baseline 与 alt，差值稳定。

## FBF 实验（272 笔，2H resim）

| 变体 | totalR | meanR | win% | Sharpe | maxDD | MaxWinR | ≥3R 赢单 |
|---|---|---|---|---|---|---|---|
| baseline (TP=2R) | +55.55 | +0.204 | 36.8% | 2.26 | -10.0 | 2.00 | 0 |
| trail act=1.5 / trail=0.8 | +78.41 | +0.288 | 34.9% | 2.44 | -15.8 | 12.89 | 12 |
| **trail act=1.5 / trail=0.5 ← 最优** | **+102.27** | **+0.376** | 34.9% | **2.86** | -15.3 | 13.19 | **15** |
| trail act=1.5 / trail=1.2 | +57.94 | +0.213 | 34.9% | 1.96 | -16.9 | 12.49 | 14 |
| 混合 TP=3.5R + trail act=2.0/trail=1.0 | +68.76 | +0.253 | 27.6% | 1.93 | -16.9 | 19.69 | 15 |

**结论：tight trail（act=1.5R, trail_r=0.5ATR）胜。**

- totalR +55 → +102（2H resim 下 +85%，1min 下等价预估 +45-60R，比原始 +33R 高 35-80%）
- Sharpe 2.26 → 2.86（提升 27%）
- 最大赢单 2R → 13.19R（解锁尾部）
- 赢单 ≥3R 从 0 笔升到 15 笔（5.5% 的样本贡献了 45R+ 的 extra PnL）
- 代价：maxDD 从 -10 → -15.3（+53% 更深，因为 trail 不如 TP 早落袋）；win% 略降 2pt
- Win-rate trade-off 可接受：赢单分布尾部完全打开

**FBF winners MFE 统计**（baseline resim 的 100 个 TP 赢单）：
```
mean MFE = 2.83R    max MFE = 13.5R
MFE >= 2R: 99% (基本所有赢单至少够到 2R 封顶)
MFE >= 3R: 22%
MFE >= 4R: 7%
MFE >= 5R: 2%
MFE >= 10R: 2%  (XRP 2024-03-11 LONG: 13.5R; ETH 2024-05-20 LONG: 13.4R)
```
尾部很肥：22% 的赢单 MFE 至少到 3R，2% 的赢单 MFE >10R。固定 2R TP 直接把这 22% 的
额外 alpha 砍掉；tight trail 基本可以吃到这些尾部（对应 MFE 3R+ 映射成 15 笔 realized 3R+）。

## MSR 实验（23 笔，2H resim）

| 变体 | totalR | meanR | win% | Sharpe | maxDD | MaxWinR |
|---|---|---|---|---|---|---|
| **baseline (TP=3R) ← 最优** | **+4.50** | **+0.196** | 30.4% | **0.79** | -5.0 | 2.50 |
| trail act=2.0 / trail=1.0 | +3.37 | +0.147 | 39.1% | 0.65 | -5.0 | 2.89 |
| trail act=1.5 / trail=1.5 | +3.53 | +0.154 | 43.5% | 0.66 | -5.0 | 2.63 |
| 固定 TP=4.5R | +5.75 | +0.250 | 21.7% | 0.83 | -6.0 | 3.75 |
| 混合 TP=3.5R + trail act=1.8/trail=1.2 | +4.88 | +0.212 | 39.1% | 0.82 | -5.0 | 2.92 |

**结论：MSR 当前 TP=3R 接近最优，trail 反而减益。**

**MSR winners MFE 统计**（7 个 baseline TP 赢单）：
```
mean MFE = 3.89R    max MFE = 4.59R
所有 7 笔 TP 赢单 MFE 分布：3.08 / 3.28 / 3.70 / 3.75 / 4.39 / 4.46 / 4.59
```

MSR 赢单的 MFE **集中在 3-4.6R**，没有 FBF 那种 >5R 的长尾。且 L3 反转完成后
迅速回撤（所以所有 trail 变体捕捉的都是 peak → retrace 的一段，平均 1.8R，反而不如
直接 TP=3R 在 peak 截胡）。

**为什么 MSR 不适合 trail**：
- L3 是**反转**型（从边界弹回），不是趋势持续；价格到目标即完成，后续是中线震荡
- Activation@2ATR 太迟（MFE 到 2ATR ≈ 1.67R 时趋势已经在末端）
- Trail@1ATR 相对 MSR 平均 MFE 3.89R，给回 1 ATR = 0.83R，在 23 笔小样本下显著

**可考虑的微调（但样本不足不建议马上动）**：
- 固定 TP 提到 3.5R 可多拿 0.38R/赢单，但 5 个赢单 drop 到 2 个（小样本不稳）
- 保留 TP=3R，专注"抬 SHORT 通道阈值"扩样本（当前 SHORT n=3）更重要

## 配置落地

### FBF（准备上线）
- `config/strategies/fbf_exp_trail/archetypes/execution.yaml` 已设为 act=1.5R / trail_r=0.5R / TP=off / BE=1R / time_stop=48
- 对应 pipeline: `config/prod_train_pipeline_2h_slow_fbf_exp_trail.yaml`
- **建议下一步**：跑一轮 full 1-min rolling_sim 验证（当前 2H resim 说明方向清楚）
  ```
  mlbot pipeline run --all --config config/prod_train_pipeline_2h_slow_fbf_exp_trail.yaml --stage rolling_sim --skip-shap
  ```

### MSR（不动）
- `config/strategies/msr_exp_trail/` 留档但不激活
- production 仍用 `config/strategies/msr/`（TP=3R，Sharpe 0.96 已够上线）

## 2H 重放脚本

新建 `scripts/simulate_exec_trail.py`，通用化的 exit-only 重放工具：
- 输入：baseline trades csv glob + feature_store 目录 + execution.yaml
- 输出：alt pnl_r/exit_reason + per-trade MFE/MAE
- 运行时 ~10-30s，用于未来任何 exit 策略对比的快速迭代

## 数据文件

| 文件 | 说明 |
|---|---|
| `reports/trail_exp/fbf_sanity_baseline_exec.csv` | FBF baseline 2H 自校准 |
| `reports/trail_exp/fbf_trail_alt.csv` | FBF trail act=1.5/0.8 |
| `reports/trail_exp/fbf_alt_trail_tight.csv` | FBF 最优 trail act=1.5/0.5 |
| `reports/trail_exp/fbf_alt_trail_wide.csv` | FBF trail act=1.5/1.2 |
| `reports/trail_exp/fbf_alt_tp2_plus_trail.csv` | FBF 混合 TP+trail |
| `reports/trail_exp/msr_sanity_baseline_exec.csv` | MSR baseline 2H 自校准 |
| `reports/trail_exp/msr_trail_alt.csv` | MSR trail act=2.0/1.0 |
| `reports/trail_exp/msr_alt_trail_wider.csv` | MSR trail act=1.5/1.5 |
| `reports/trail_exp/msr_alt_tp45.csv` | MSR 固定 TP=4.5R |
| `reports/trail_exp/msr_alt_tp35_trail.csv` | MSR 混合 TP=3.5R+trail |

## 下一步建议

1. **FBF trail 生产验证（优先 P0）**：在 1-min event_backtest 下跑一轮 full rolling_sim，
   确认 Sharpe / maxDD 是否维持 2H resim 的改善方向。预计 10-12h。

2. **Scale-out 原生支持（后续 P2）**：当前 `position_logic.py` 只支持单档 TP。
   若未来想搞 FBF "50% at 2R + 50% trail"（可能是本地最优解），需要扩展 position_logic
   支持 `partial_tp_targets: [{fraction, target_r}, ...]` 和子仓尺寸缩减。

3. **MSR 样本扩大（优先 P1）**：MSR 的边际价值来自更多高质量 L3 事件，不是更好的 exit。
   短通道（当前 n=3）严重不足；应研究 `wide_sr_side >= 0.3` 是否可降到 0.2。

4. **FBF 分析问题（待查）**：2H 重放里有 28 笔 (10.3%) `no_bars` —— 月末/月初的 trades
   feature_store 未覆盖。若要精确，要跨月 bar 拼接；当前不影响方向性结论。

---

# P0 v2 生产验证结果（2026-04-22 深夜追加）

## 起因：P0 v1 trailing stop 完全没生效（pipeline bug）

首次生产验证（`_rolling_sim/20260422_165551`）出现异常：maxR=1.98，exit_reason 只有 `sl/tp`，
totalR=18.09。与 baseline 几乎相同——说明 trailing 根本没运行。

**根因**：`scripts/auto_research_pipeline.py:1849` 的 `_copy_from` 优先级逻辑 bug：

```python
_src_strategy_dir = _src_root / strategy      # config/strategies/fbf
_fallback_prod_dir = PROJECT_ROOT / prod_config_dir  # config/strategies/fbf_exp_trail
_copy_from = _src_strategy_dir if _src_strategy_dir.exists() else _fallback_prod_dir
```

因为 strategy key 是 `fbf`（pipeline YAML 里 `strategies.fbf.config: config/strategies/fbf_exp_trail`），
copytree 优先用 `_src_root / "fbf"`（即原版 FBF），**完全忽略 `scfg["config"]` 的指向**。
所有以 `_exp_*` 命名的实验（fbf_exp_trail / msr_exp_trail / fbf_exp_fatter_tp / rmr）
都中了同一个坑，旧的"好 run"可能压根没真正用新配置。

**修复**：改 `_copy_from` 优先级——默认 bootstrap 场景下（`_src_root == config/strategies`）
且用户显式指定非默认 prod 目录时，用 `prod_config_dir`；snapshot 传入场景保持原语义。
提交在 `auto_research_pipeline.py` L1849-1872。

**产物隔离**：将 bug 结果目录改名 `_BUGGY_20260422_165551_trailing_not_applied` 留证。

## P0 v2 最终数据（`_rolling_sim/20260422_202736`，修复后真跑 16 月 1-min）

| 指标 | **FBF-trail v2** | fatter-TP 4R | baseline 2R | v1 BUG |
|---|---|---|---|---|
| trades | 173 | 272 | 213 | 267 |
| **totalR** | **37.83** | 32.73 | 16.96 | 18.09 |
| **Sharpe** | **2.22** | 1.30 | 0.86 | 0.74 |
| PF | **1.39** | 1.19 | 1.12 | 1.10 |
| win% | **48.0%** | 40.1% | 38.5% | 38.2% |
| maxR | **9.84** | 1.98 | 1.98 | 1.98 |
| maxDD | -16.11R | -12.16R | -18.55R | -16.84R |
| exits | sl + **trailing_sl** | sl + tp@2 | sl + tp@2 | sl + tp@2 |

R 分桶（>2R 赢单全部 `trailing_sl` 退出）：

| 桶 | v2 | fatter-TP | baseline |
|---|---|---|---|
| >5R | **1** | 0 | 0 |
| 3-5R | **2** | 0 | 0 |
| 2-3R | **13** | 0 | 0 |
| 1-2R | 50 | 109 | 82 |

**vs 历史最佳（fatter-TP）**：totalR +15.6%，Sharpe **+71%**，trades −36%（单笔效率 0.120 → 0.219 R）。
**vs baseline**：totalR **+123%**，Sharpe **+158%**。

### 关键观察

- **尾部革命**：XRPUSDT 2024-09 LONG +9.84R（持仓 ~17h 被 trail 释放）单笔贡献 26% totalR。
  即便剔除仍 totalR 28R > baseline 17R。
- **窗口选择性**：2024-12 杀人月（baseline −12.5R）v2 直接 0 trades 规避；2024-04/05/10/11 全部 0 trades，
  怀疑 threshold_calibration 过度保守，可能漏掉好机会，需要后续追查。
- **坏月在**：2024-06 v2 拿 18 笔亏 -12.07R（baseline 0 笔）——trailing 放宽了 signal 门槛，
  但某些 symbol/side 质量差；可能需要 per-symbol gate 处理。

### 2H resim vs 1-min 一致性

| 预测 | 实际 |
|---|---|
| totalR +85%（2H resim） | **+123%**（1-min，放大） |
| Sharpe 2.26 → 2.86 | 0.86 → **2.22** |

方向性结论完全成立，1-min 颗粒度让 trail 捕捉更细腻，效果放大。

## 生产落地

### FBF：已落地（2026-04-22）

- `config/strategies/fbf/archetypes/execution.yaml` → 替换为 trail 版（`version: 54`）
- Git rollback tag: `pre-fbf-trail-promote-20260422`
- 回滚命令：`git checkout pre-fbf-trail-promote-20260422 -- config/strategies/fbf/archetypes/execution.yaml`
- `config/strategies/fbf_exp_trail/` 保留作实验对照组

### MSR：1-min 验证进行中

- 启动 `_rolling_sim/20260423_081933`，预计 ~3h 完成。
- 验证 2H resim 的"MSR 上 trail 反而更差"是否在 1-min 成立：
  - 若 confirmed → MSR 保持 TP=3R 不动，结束 exit strategy 研究
  - 若 1-min 下 trail 有意外收益 → 重新评估

## Pipeline Bug 的连带影响

以下历史实验目录都曾在 bug 下运行，结果仅供参考，必要时需重跑：
- `config/strategies/fbf_exp_fatter_tp/` → 20260416_153251 run
- `config/strategies/msr_exp_trail/` → 尚未真跑过
- `config/strategies/bad-candidates/rmr/` → RMR 被废弃，无需重跑
- 任何 `config/strategies/X_exp_*/` 形式的未来实验，自动修复后生效
