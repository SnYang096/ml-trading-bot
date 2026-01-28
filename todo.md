1. 'BreakoutPullbackContinuation','HTFBiasLTFEntry',  'MomentumExpansion','FailedBreakoutFade','LiquiditySweepRejection'，'AuctionExhaustionReversal'的方向方案重构 [策略中dir的使用方式](策略中dir的使用方式.md), 
2. nn模型保持仓位下注大小，止损止盈辅助功能。这个对三种mean模式是不是不友好
3. safety功能，用来降速甚至停止，但nn模型也能控制size，是不是不要多个输入：最大回撤，连亏次数，kl散度，mi互信息
4. 跑通order management和前面特征计算gate，archtype链接
5. 拿到上面6个archtype的稳定参数和稳健性报告。对比树模型测语义，确保语义实现的没问题
6. 上测试网测试
7. 规则类的消融方法
8. 一个新特征 vp_boundary_stability_score 
sr_distance_normalized 
sr_strength_max	建议用	判断 SR 质量（所有 SR 的最大 SQS）	未使用
dist_to_nearest_sr	已用	判断距离（ATR 归一化）	通过 sr_distance_normalized 使用
direction_to_nearest_sr	可选	判断方向	未使用
sqs_hal_high / sqs_hal_low	可选	HAL 特定质量	未使用
sqs	不推荐	太通用，不如 sr_strength_max	未使用

sr_strength_max_f
sr_strength_max 是特征，建议使用：它综合了所有 SR 的 SQS，比单独的 sqs 更全面

vp_boundary_stability_score sr_strength_max dist_to_nearest_sr