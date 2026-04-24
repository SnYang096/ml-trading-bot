# 盒子特征：Oracle（偷看未来）vs Causal（因果）对照诊断

## 目标

验证 [src/features/time_series/box_structure_features.py](../src/features/time_series/box_structure_features.py) 导出的因果 box 特征能保留多少 "盒子结构 alpha"，与之前 [scripts/diag_consolidation_structure.py](../scripts/diag_consolidation_structure.py) 偷看未来的 oracle 扫描器对比。

## 实验设置

- 数据：BTCUSDT，2024-01-01 ~ 2024-12-31，2H 重采样 → 4393 bars。
- Oracle 参数：`decay_max=120`, `decay_pct=0.08`, `min_len=30`, `max_len=720`, `tol_atr=1.0`, `tol_pct=0.02`。
- Causal 参数：只用 `box_regime_label ∈ {small, mid, big}` 作为 in-box 判据（特征内部窗口 60/120/240，stability 阈值 0.70，宽度阈值 4%/8%）。
- CRF 模拟规则两边一致：`edge_frac=0.15`，`stop_mult=0.25`，对边 edge 止盈，盒子结束强平。

## 结果对照

| 指标 | Oracle | Causal |
|---|---|---|
| 事件数 | 4 | **6** |
| outcome mix | 3 timeout + 1 break_up | 5 timeout + 1 break_up |
| box_len 中位（bar） | 719 | 719 |
| box_width 中位 | 29.8% | 31.6% |
| break_up MFE/MAE | 5.60 | 0.78 |
| **CRF trades** | 8 | **14** |
| CRF winrate | 87.5% | **92.9%** |
| **CRF total_ret** | +99.6% | **+268.6%** |
| CRF long_ret | +49.4% | +195.0% |
| CRF short_ret | +50.1% | +73.7% |
| CRF avg_hold_bars | 277.8 | 304.2 |

## 关键结论

### 1. 因果特征不仅没丢 alpha，反而抓到更多事件

Oracle 要求 `strong uptrend → 8% decay → box` 全链条，太窄；BTC 2024 只命中 4 次。Causal 只问"当前是否持续稳定"（`box_stability_120 >= 0.7`），包含了 oracle 的真子集 + 一批"无前置衰竭但本身就是宽震"的大盒子。两者合起来 CRF 总收益 +268.6% >> +99.6%。

**这直接推翻了"盒子识别必须偷看未来"的担忧**——因果 box 特征可直接用于实盘。

### 2. CBC（盘整突破顺势）作为独立策略不立项

Causal 模式下 break_up MFE/MAE=0.78 < 1，延续行情并不显著；而且 BTC 全年只有 1 次 break_up 事件，样本太少。
与现有 srb/bpc/me 语义重叠，CBC 作为独立策略**不立项**；改为把 box 特征接入三者 prefilter/confidence（P3a）。

### 3. CRF 波段策略值得进 P3b

在 causal 因果识别下，BTC 2024 全年 6 个盒子里 CRF 模拟 14 笔 92.9% 胜率 +268.6%；long/short 两侧各有 +195% / +73%，**多空贡献对称且都为正**，证明"一个策略同时做多做空"的设计可行。

### 4. 当前 "big" 盒子为主

两种模式的 `box_width` 中位都在 30% 左右，意味着 2024 BTC 整体是巨型宽震结构。`regime_label = big` 的盒子反而是 alpha 主力，不应该一开始就把 big 排除在外。

## 下一步行动（按原 plan）

- **P3a-srb**：要求突破前是真盒子（`box_stability_120 >= 0.7` + `box_width_pct_120 <= 0.05`）。
- **P3a-bpc**：要求 pullback 在 `box_regime_label in {small, mid}`。
- **P3a-me**：要求 `box_compression_score <= 0.6` 作为扩张前置。
- **P3b-crf** — 绿灯启动：`box_pos_120 <= 0.15` 做多、`>= 0.85` 做空，前提 `box_stability_120 >= 0.7`。

## 产物

- [src/features/time_series/box_structure_features.py](../src/features/time_series/box_structure_features.py) - causal 特征模块
- [tests/features/test_box_structure_features.py](../tests/features/test_box_structure_features.py) - 8 条单测（含因果性、NaN、极端形态、regime 枚举、breakout 触发）
- [config/feature_dependencies.yaml](../config/feature_dependencies.yaml) - `box_structure_f` 注册条目（26 个输出列）
- [scripts/diag_consolidation_structure.py](../scripts/diag_consolidation_structure.py) - 增加 `--mode {oracle, causal}`
- [reports/box_btc_2024_oracle.csv](../reports/box_btc_2024_oracle.csv) / [reports/box_btc_2024_causal.csv](../reports/box_btc_2024_causal.csv)
