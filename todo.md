1. 'BreakoutPullbackContinuation','HTFBiasLTFEntry',  'MomentumExpansion','FailedBreakoutFade','LiquiditySweepRejection'，'AuctionExhaustionReversal'的方向方案重构 [策略中dir的使用方式](策略中dir的使用方式.md), 
2. nn模型保持仓位下注大小，止损止盈辅助功能。这个对三种mean模式是不是不友好
3. safety功能，用来降速甚至停止，但nn模型也能控制size，是不是不要多个输入：最大回撤，连亏次数，kl散度，mi互信息
4. 跑通order management和前面特征计算gate，archtype链接
5. 拿到上面6个archtype的稳定参数和稳健性报告。对比树模型测语义，确保语义实现的没问题
6. 上测试网测试
