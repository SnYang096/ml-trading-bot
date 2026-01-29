1. 'BreakoutPullbackContinuation','HTFBiasLTFEntry',  'MomentumExpansion','FailedBreakoutFade','LiquiditySweepRejection'，'AuctionExhaustionReversal'的方向方案重构 [策略中dir的使用方式](策略中dir的使用方式.md), 
2. nn模型保持仓位下注大小，止损止盈辅助功能。这个对三种mean模式是不是不友好
3. safety功能，用来降速甚至停止，但nn模型也能控制size，是不是不要多个输入：最大回撤，连亏次数，kl散度，mi互信息
4. 跑通order management和前面特征计算gate，archtype链接
5. 拿到上面6个archtype的稳定参数和稳健性报告。对比树模型测语义，确保语义实现的没问题
6. 上测试网测试
7. 规则类的消融方法，在archetypes那边做的
8. 一个新特征 vp_boundary_stability_score  dist_to_nearest_sr sr_strength_max
9. 建立6个树模型 


1. 你的目标（对齐一下）
不做：把 execution_archetypes 里现有规则搬到树那边（那些规则你也不确定对不对）。
要做：按 6 个 archetype 的语义 在树这边设计 6 个新策略；树还是「自己学规则、自己训」，只是策略的 交易逻辑/语义 和 6 个 archetype 一一对应。
现状：原来 4 个策略（sr_reversal, compression_breakout, sr_breakout, trend_following）既对不上 6 个 archetype，部分 Sharpe 还负，所以你想用这 6 个「语义对齐 archetype」的新策略来替代/补充。
2. 能复用什么、不能复用什么
目录结构 —— 可以完全复用
和现在树策略一样即可：
config/strategies/<strategy_name>/
labels.yaml
model.yaml
features.yaml
meta.yaml
backtest.yaml
evaluation.yaml
6 个新策略就是 6 个新目录，例如：bpc、htf、me、fbf、lsr、aer（或你喜欢的命名）。
模型训练方式 —— 可以完全复用
继续用同一套：train_strategy_model、LightGBM、task_type（regression / multiclass）、target_col、TS-CV 等。
每个新策略的 model.yaml 和现有策略同构即可：指定 target_col、task_type、model_params 等，不需要新训练框架。
Labels —— 部分可复用，部分要新写
6 个 archetype 的 语义 和现有 4 个策略的 label 对比如下：
Archetype	语义（交易想法）	和现有 label 的关系
MomentumExpansion	压缩后波动/区间扩张，放量突破	和 compression_breakout 非常接近（compression → breakout 方向）。可 直接复用 compute_compression_breakout_label，必要时调参数（confirmation_bars、compression_col 等）。
AuctionExhaustionReversal	趋势末端衰竭（量/波动极值）后反转	和 sr_reversal 都是「反转」，但多了「衰竭」前提。可 在 sr_reversal 思路上加条件：例如只在 atr_percentile / path_efficiency 等满足「衰竭」的 bar 上打 SR 附近的反转标签；或写一个 exhaustion_reversal 的小变体。
BreakoutPullbackContinuation	趋势中先回踩再延续原方向	现有没有「回踩 + 延续」的标签。需要 新 label：例如用 trend_sign 定方向 + 回踩定义（如 LL in uptrend）+ 后续 N bar 是否延续且达到一定 RR。
HTFBiasLTFEntry	大周期定方向，小周期定入场	现有没有显式 HTF/LTF 的标签。需要 新 label：例如 HTF 趋势方向 + LTF 入场信号（如 wick rejection、订单流）出现后的 forward return。
FailedBreakoutFade	假突破（突破后失败）→ 反手 fade	现有没有「假突破 + fade」的标签。需要 新 label：例如识别 failed_breakout（破高/破低后收回去）+ 按 fade 方向算 forward RR。
LiquiditySweepRejection	流动性扫损（sweep）后价格拒绝并反向	现有没有「sweep + rejection」的标签。需要 新 label：例如 sweep 检测（如 wick 扫前高/前低后收回）+ 反向的 forward return。
结论：
可以复用的：
MomentumExpansion → 复用 compression_breakout 的 label 逻辑（同一套 label 生成函数，可能不同 params）。
AuctionExhaustionReversal → 复用或小幅扩展 sr_reversal（例如加「衰竭」过滤或新函数 exhaustion_reversal_label）。
需要新 label 的：
BreakoutPullbackContinuation、HTFBiasLTFEntry、FailedBreakoutFade、LiquiditySweepRejection 这 4 个，语义在现有 4 个策略里都没有现成对应，要 新写 4 个 label 生成函数（或 2–3 个通用函数 + 不同参数）。
3. 落地时怎么选：复用 vs 新写
目录：6 个新策略 = 6 个 config/strategies/<name>/，结构照搬现有。
训练：6 个都继续用现在的树模型训练方式（同一套 model.yaml / trainer）。
Labels：
ME：labels.yaml 里 module/function 指向现有的 compression_breakout label，必要时改 params。
AER：要么在现有 sr_reversal label 上包一层「仅衰竭区」的过滤，要么新写一个 compute_exhaustion_reversal_label，再在 labels.yaml 里引用。
BPC / HTF / FBF / LSR：各写一个 compute_xxx_label(df, **params)，在对应策略的 labels.yaml 里引用；这些函数的「语义」按上表对应 archetype 来设计，和 execution_archetypes 里的具体规则无关，只是交易想法一致。
这样：树这边 还是和现在一样（同样的目录结构、同样的训练方式），只是从「4 个旧策略」变成「6 个按 archetype 语义设计的新策略」；其中 2 个可尽量复用现有 label，4 个需要新 label，但训练流水线和目录都可以不变。


python src/time_series_model/visualization/feature_indicator_visualizer.py \
  --data-path data/parquet_data \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-12-31 \
  --strategy-config config/strategies/compression_breakout