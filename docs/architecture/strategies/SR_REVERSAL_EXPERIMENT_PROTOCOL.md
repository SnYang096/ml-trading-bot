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

> 相关设计文档（建议配合阅读）：`docs/architecture/strategies/SEMANTIC_FEATURES_4_SCENARIOS.md`
> 语义化模板库（可复用）：`docs/architecture/strategies/ORDERFLOW_SEMANTIC_MAPPING_TEMPLATES.md`
> 研究系统架构（Experiment Loop / Layer A-B-C / TaskSpec）：`docs/architecture/EXPERIMENT_LOOP_ARCHITECTURE.md`

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

## 5.5 Layer A（必须先固定）：SR Reversal 的主线/基线 Task（label + sample）

> 目的：后续所有“特征迭代/消融/搜索”必须复用同一套 Layer A（否则结论不可比较）。

当前 SR Reversal（回归 rr）推荐固定两套候选：

| Task | 策略目录 | 说明 | 用途 |
|---|---|---|---|
| baseline | `config/strategies/sr_reversal_rr_reg_long` | 稳定的 kline+SR-distance 特征基线（不含 ticks） | 对照 |
| mainline | `config/strategies/sr_reversal_rr_reg_long_mainline` | **SR-filter labels + sample weights**（当前主线） | 主线 |

### 固定命令模板：seed sweep（强制可复现）

```bash
python3 scripts/run_model_comparison_seed_sweep.py \
  --strategies sr_reversal_rr_reg_long,sr_reversal_rr_reg_long_mainline \
  --symbols BTCUSDT --timeframe 240T \
  --start-date 2023-01-01 --end-date 2025-10-31 \
  --test-size 0.3 --seeds 1,2,3,4,5 \
  --output-dir results/model_comparison/seed_sweep_sr_reversal_layer_a \
  --no-docker
```

> 约定：Layer B（features）只允许在 `mainline` 上做增量；Layer C（model/params）需要另开章节，不要混在 feature search 中。

---

## 5.6 YAML 工件约定（Pool B / Base Pool / Pool A Suggested）

> 目的：让“候选池（filter）→ 组合验证（wrapper）→ 写回配置（suggested）”完全可追溯、可复用。

### Pool B（Filter 输出：候选池 YAML）

- **默认生成位置（factor-eval）**：`results/factor_ts_eval/{strategy_name}_{symbol}_features_suggested.yaml`
- **推荐归档位置**：`results/pools/<strategy_name>/pool_b/features_pool_b.yaml`
- **语义**：
  - `feature_pipeline.requested_features`：候选特征（很多）
  - `feature_pipeline.invert_features`：候选反向清单（很多；此阶段不是最终定型）

### Base Pool（Wrapper 起点：默认就是策略主线 features.yaml）

- **默认 base pool 文件**：`config/strategies/<strategy_name>/features.yaml`
- **默认 base pool 字段**：`feature_pipeline.requested_features`
- **可选覆盖**：`mlbot diagnose feature-group-search --base-features-yaml <yaml_list>`

### Pool A Suggested（Wrapper 输出：建议 YAML）

- **实验记录输出目录**：`results/feature_group_search/<run_name>/`
  - `feature_group_search_result.json`
  - `feature_group_search_candidates.csv`
  - `feature_group_search_report.html`
- **写回建议 YAML**：`config/strategies/<strategy_name>/features_suggested.yaml`
  - `invert_features` 会自动裁剪为：`invert_candidates ∩ final_requested_features`

示例命令（从主线 features.yaml 出发，叠加候选组并写回 suggested）：

```bash
mlbot diagnose feature-group-search \
  -c config/strategies/sr_reversal_rr_reg_long_mainline \
  -s BTCUSDT -t 240T \
  --start-date 2023-01-01 --end-date 2025-10-31 \
  --test-size 0.3 --seeds 1,2,3,4,5 \
  --groups-yaml config/feature_groups_sr_reversal_semantic.yaml \
  --invert-candidates-yaml results/pools/sr_reversal_rr_reg_long_mainline/pool_b/features_pool_b.yaml \
  --writeback-yaml config/strategies/sr_reversal_rr_reg_long_mainline/features_suggested.yaml \
  --output-dir results/feature_group_search/sr_reversal_rr_reg_long_mainline__search \
  --deterministic --no-docker
```

> `features_suggested.yaml` 里会额外包含一个顶层字段 `feature_group_search`，用于记录：
> baseline / stop_reason / selected_groups / final_features / groups_source 等审计信息。  
> 训练不会读取这部分（只读取 `feature_pipeline.*`），因此你可以安全地整文件 copy 到 `features.yaml` 使用。  
> 详细字段解释见：`docs/architecture/EXPERIMENT_LOOP_ARCHITECTURE.md` 的 “如何解读 feature_group_search 元数据” 小节。

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

### 10.X Orderflow 分组消融（BTCUSDT）：TradeCluster 明显拖累反转，CVD 接近中性
目的：解释“为什么加 orderflow 后 Sharpe 下降”，并定位到底是哪一组 orderflow 在拖累反转策略。

协议：
- symbol：`BTCUSDT`
- timeframe：`240T`
- window：`2023-01-01 ~ 2025-10-31`
- test-size：`0.3`
- seeds：`1,2,3,4,5`
- 其他配置：完全固定（同一 label/backtest/model），**只改 `features.yaml` 的 orderflow 子组**

策略（A/B 对照）：
- baseline：`sr_reversal_rr_reg_long`
- VPIN-only：`sr_reversal_rr_reg_long_of_vpin`（仅加 `vpin_derived_features_f`）
- TradeCluster-only：`sr_reversal_rr_reg_long_of_trade_cluster`（仅加 `trade_cluster_block_features_f`）
- CVD-only：`sr_reversal_rr_reg_long_of_cvd`（仅加 bar-level CVD 相关特征：`cvd_slope_5_f/hurst_cvd_f/wpt_cvd_fluctuation_f`）
- full orderflow：`sr_reversal_rr_reg_long_orderflow`（orderflow 全量：VPIN + TradeCluster）

命令（可复现）：
```bash
python3 scripts/run_model_comparison_seed_sweep.py \
  --strategies sr_reversal_rr_reg_long,sr_reversal_rr_reg_long_of_vpin,sr_reversal_rr_reg_long_of_trade_cluster,sr_reversal_rr_reg_long_of_cvd,sr_reversal_rr_reg_long_orderflow \
  --symbols BTCUSDT --timeframe 240T \
  --start-date 2023-01-01 --end-date 2025-10-31 \
  --test-size 0.3 --seeds 1,2,3,4,5 \
  --output-dir results/model_comparison/seed_sweep_orderflow_ablation_btc_v2_seedwired \
  --no-docker
```

结果摘要（mean over seeds）：
- baseline：Sharpe≈ **1.84**，return%≈ **11.64**
- CVD-only：Sharpe≈ **1.72**，return%≈ **10.86**（接近中性）
- VPIN-only：Sharpe≈ **1.33**，return%≈ **8.28**（偏负）
- TradeCluster-only：Sharpe≈ **0.82**，return%≈ **4.75**（显著负）
- full orderflow：Sharpe≈ **0.90**，return%≈ **5.12**（整体被 TradeCluster 拉低）

结论：
- 对“SR 附近反转”这一类任务：**TradeCluster 特征高度可疑**（更像突破/延续类信息），会显著伤害 Sharpe。
- VPIN 在该窗口下也偏负，需要更细分（例如只留 spike/zscore，或改成“触发型 gate”）。
- **CVD-only 更接近中性**（至少没有明显拖累），更可能作为反转的辅助确认项，而不是主驱动。

### 10.Z TradeCluster 语义化（BTCUSDT）：raw 为负，semantic 可转正
动机：不要直接喂 `trade_cluster_*` 原始统计量（容易学到“趋势仍强”），而是转成“路径语义”：
- **Exhaustion（衰竭）**：放量/聚集，但价格位移不大（Effort without progress）→ 反转友好
- **Absorption（吸收）**：放量/聚集，且价格位移很大 → 突破/延续友好

实现（新增 feature node）：
- `trade_cluster_semantic_scores_f`（输出三列）：
  - `trade_cluster_flow_intensity`
  - `trade_cluster_exhaustion_score`
  - `trade_cluster_absorption_score`
（代码：`src/features/time_series/utils_order_flow_features.py`）

协议：
- symbol：`BTCUSDT`
- timeframe：`240T`
- window：`2023-01-01 ~ 2025-10-31`
- test-size：`0.3`
- seeds：`1,2,3,4,5`

策略对照：
- baseline：`sr_reversal_rr_reg_long`
- raw trade cluster：`sr_reversal_rr_reg_long_of_trade_cluster`
- semantic trade cluster：`sr_reversal_rr_reg_long_trade_cluster_semantic`
- liquidity void：`sr_reversal_rr_reg_long_liquidity_void`

结果摘要（mean over seeds）：
- baseline：Sharpe≈ **1.61**，return%≈ **9.90**
- raw trade cluster：Sharpe≈ **0.56**，return%≈ **3.57**（明显拖累）
- semantic trade cluster：Sharpe≈ **2.00**，return%≈ **13.15**（显著转正，且 corr_mean 更高）
- liquidity void：Sharpe≈ **2.05**，return%≈ **13.05**

结论：
- 这次假设成立：**TradeCluster raw 的负面主要来自“语义混杂”**，语义化后可以把信息拆出来，对反转不再致命。
- 下一步建议：把 `disp_atr_threshold / ma_window / window_size` 做小范围 sweep，并在 ETH 上复核一致性。

### 10.AA VPIN / Imbalance 语义化（BTCUSDT）：VPIN_semantic 显著正面，Imbalance_semantic 较弱
目的：验证“VPIN/Imbalance raw 异义 → 语义化后对齐反转逻辑”的可迁移性（继 TradeCluster 之后的第二组/第三组 orderflow 语义化）。

实现（新增 feature nodes）：
- `vpin_semantic_scores_f`：`vpin_stress_score`, `vpin_directional_pressure`, `vpin_exhaustion_score`
- `tbr_imbalance_semantic_scores_f`：`imbalance_ratio`, `imbalance_exhaustion_score`（bar-level，用 `taker_buy_ratio` 近似不平衡）

协议：
- symbol：`BTCUSDT`
- timeframe：`240T`
- window：`2023-01-01 ~ 2025-10-31`
- test-size：`0.3`
- seeds：`1,2,3,4,5`

策略对照：
- liquidity void：`sr_reversal_rr_reg_long_liquidity_void`
- trade cluster semantic：`sr_reversal_rr_reg_long_trade_cluster_semantic`
- vpin semantic：`sr_reversal_rr_reg_long_vpin_semantic`
- imbalance semantic：`sr_reversal_rr_reg_long_imbalance_semantic`

结果摘要（mean over seeds）：
- **vpin semantic**：Sharpe≈ **2.15**，return%≈ **13.82**（trades≈15.8，DD%≈4.20）
- **trade cluster semantic**：Sharpe≈ **2.02**，return%≈ **13.40**（DD%≈5.12）
- **imbalance semantic (TBR)**：Sharpe≈ **1.49**，return%≈ **9.35**
- **liquidity void**：Sharpe≈ **1.43**，return%≈ **9.26**（该组在本次 sweep 下方差较大：Sharpe_std≈0.93）

结论：
- 在该窗口/参数下，**VPIN 的语义化版本是当前最强的 orderflow 语义信号之一**（均值 Sharpe/return% 领先）。
- `taker_buy_ratio` 作为 imbalance proxy 的语义化效果一般：可能需要换成更“微观”的不平衡原料（例如 ticks footprint 的 imbalance / delta divergence）或加上 SR/压缩 gating。
- liquidity_void 在单次 seed 上可以很强，但方差偏大；建议后续在 ETH 上复核，或与 “void + exhaustion” 组合做稳定性对比。

### 10.Y “限价墙/难突破”非 L2 代理特征（BTCUSDT）：组合版不稳定且均值接近 0
动机：没有 L2 订单簿（深度/挂单）数据时，用可获得的代理特征模拟“限价单墙/拒绝/难突破”。

我们构造了一组 proxy 特征（在 `sr_reversal_rr_reg_long` 基础上增量添加）：
- SR 结构/质量：`sr_strength_max_close_f`, `sqs_f`, `sqs_hal_*`
- 蜡烛“拒绝”代理：`wick_ratios_f`
- Volume Profile：`volume_profile_vpvr_f`, `volume_profile_volatility_features_f`
- WPT 假突破风险：`wpt_volume_energy_f`（`wpt_false_breakout_risk` 等）
- Liquidity void：`liquidity_void_f`
- 形态匹配（DTW，近 SR 才算）：`dtw_features_reversal_f`

协议：
- symbol：`BTCUSDT`
- timeframe：`240T`
- window：`2023-01-01 ~ 2025-10-31`
- test-size：`0.3`
- seeds：`1,2,3,4,5`

策略对照：
- baseline：`sr_reversal_rr_reg_long`
- liquidity void：`sr_reversal_rr_reg_long_liquidity_void`
- limit-wall proxy bundle：`sr_reversal_rr_reg_long_limit_wall_proxy`

结果摘要（mean over seeds）：
- baseline：Sharpe≈ **1.99**，return%≈ **12.63**
- liquidity void：Sharpe≈ **1.87**，return%≈ **12.62**（依旧正、较稳）
- limit-wall proxy bundle：Sharpe≈ **-0.01**，return%≈ **-0.36**（std 很大、DD% 高）

结论：
- 这一大包 proxy 特征**组合后对回测不友好**（均值接近 0 且方差/回撤偏大），不建议直接作为主线。
- 下一步如果要继续探索“限价墙/拒绝”，建议拆成更小的 A/B：
  - 仅加 `volume_profile_*`（不含 DTW/WPT）
  - 仅加 `wpt_volume_energy_f`
  - 仅加 `dtw_features_reversal_f`（并确认依赖包与缺失率）
  逐个 seed sweep，定位到底是哪一块在拉跨。

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

---

### 10.AB 语义化特征（SR Reversal）“有没有意义？”快速结论表

> 口径：BTCUSDT / 240T / 2023-01-01~2025-10-31 / test_size=0.3；除非特别说明，均为 seeds=1..5 的 mean。

| 特征组/策略 | 结论（对 SR Reversal） | 证据（mean 级别） |
|---|---|---|
| raw TradeCluster（`trade_cluster_block_features_f`） | **显著负面**（更像突破/延续信号） | Sharpe≈0.56（见 10.Z） |
| TradeCluster semantic（`trade_cluster_semantic_scores_f`） | **显著正面**（Exhaustion/Absorption 解耦后转正） | Sharpe≈2.00（见 10.Z） |
| VPIN semantic（`vpin_semantic_scores_f`） | **显著正面**（当前最强 orderflow 语义之一） | Sharpe≈2.15（见 10.AA） |
| Imbalance semantic（`tbr_imbalance_semantic_scores_f`） | **较弱/接近中性**（proxy 可能不够微观） | Sharpe≈1.49（见 10.AA） |
| Liquidity void（`liquidity_void_f`） | **可能正面但方差偏大**（单次 seed 很强但稳定性需复核） | Sharpe≈1.43 且 std≈0.93（见 10.AA 注释） |
| “限价墙代理 bundle”（VPVR+WPT+DTW+…） | **组合版不稳定且均值接近 0**（不建议直接上主线） | Sharpe≈-0.01（见 10.Y） |

> 说明：上表回答“有没有意义”；但“最优组合是什么”需要做组合搜索（而不是单组 ablation），这也是 `mlbot diagnose feature-group-search` 的目标。


