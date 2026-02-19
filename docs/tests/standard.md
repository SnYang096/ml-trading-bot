# 特征测试标准

非常清晰的「特征测试标准」清单与当前覆盖情况梳理，整体结构合理、覆盖重点明确。以下是对现状的总结，并结合你的待补/改进建议，给出优先级排序 + 补位建议 + 可执行方案，便于后续落地。

一、现状评估（按标准逐项）

标准 当前覆盖情况 覆盖强度 备注
------ --------------- ---------- ------
无未来数据 部分覆盖（VPIN/RSI/Hurst 等有容忍度检查） 中 缺少对窗口右闭左开的精确验证，尤其在 Footprint / 频谱类中未显式验证
NaN/Inf 防护 覆盖良好 高 多个模块已加 eps + replace(inf, nan)，且允许 NaN 输出
归一化/可比性 几乎无覆盖 低 仅 VPIN 多维特征隐含部分可比性，但无跨标的缩放验证
边界/空数据 部分覆盖（Footprint 有空 bar NaN） 中 缺列/空输入场景未系统测试
稳定性/数值容忍 部分覆盖（VPIN/TC 有状态连续性） 中 未统一 assert 容差策略，随机种子未强制
数据类型/索引对齐 未显式测试 低 聚合测试隐含部分对齐，但无 merge/index 重复等边界测试
单位/字段要求 未覆盖 低 缺列错误处理缺失
缓存/状态 覆盖良好 高 VPIN/TC 的流式、跨月、回退均有测试
✅ 强项：Inf/Nan 防护、缓存/状态管理
⚠️ 短板：归一化可比性、无未来数据精确验证、缺列报错、索引对齐

二、补位建议（按优先级排序）
🔴 P0（高优）：归一化/可比性 + 缺列报错
理由：
归一化是跨资产建模的核心前提，若不可比将导致模型失效；
缺列报错属于基础健壮性，用户体验关键。
建议动作：
1. 新增通用辅助函数 assert_normalized_invariance(feature_fn, scale_factors=[(10, 0.1), (100, 0.01)])：
输入两组同分布但价格/volume 缩放后的数据；
断言输出（如 zscore、量价比）相对误差 < tol（如 1e-5）。
2. 为以下模块添加缺列测试：
compute_footprint_features
extract_order_flow_features
所有依赖 price, volume, side, tick_size 的特征函数
验证抛出 ValueError 且含“missing column: xxx”字样。
🟠 P1（中优）：无未来数据精确验证 + 频谱类（Hilbert/WPT/EVT）补测
理由：
频谱类特征天然易引入未来信息（如 FFT 全局窗）；
当前“容忍度检查”不够严谨，需构造单调标签+时间戳对齐验证。
建议动作：
1. 构造合成数据：t=0..N，标签 y_t = t（严格递增）；
2. 对每个 t，验证 feature(t) 与 y_{t+1} 相关系数 ≈ 0（或 p > 0.05）；
3. 为 Hilbert/WPT/EVT 添加最小窗口测试（如 window=10），验证：
前 window-1 个输出为 NaN；
第 window 个输出仅依赖 [0, window-1]；
无 inf/nan（恢复之前移除的 inf guard）。
🟡 P2（低优但必要）：索引对齐 + 单元测试容差标准化
建议动作：
在 conftest.py 或 test_utils.py 中定义：
python
def assert_series_equal(a, b, rtol=1e-5, atol=1e-8):
pd.testing.assert_series_equal(a, b, rtol=rtol, atol=atol, check_names=False)
统一替换所有 == 或 np.allclose 为该函数；
新增测试：merge 后索引去重、tick 与 K 线时间包含关系（如 tick in [open_time, close_time)）。

三、可执行下一步（我可直接落地）

请确认以下任一选项，我将生成完整测试代码：
✅ 选项 A（推荐）：先补 P0 —— 归一化可比性 + 缺列报错
覆盖模块：Footprint / VPIN / Trade Clustering / 所有 zscore/百分比类特征
输出：
test_utils.py：新增 assert_normalized_invariance
test_feature_robustness.py：缺列报错测试
各模块补充 1~2 个可比性测试用例
✅ 选项 B：先补 P1 —— Hilbert/WPT/EVT 无未来 + 稳定性
输出：
test_spectral_features.py：含 lookahead 验证 + inf guard + 最小窗口
恢复并加固之前移除的频谱类 inf 防护
✅ 选项 C：全量推进（分 PR）
我可按 P0 → P1 → P2 顺序，分 3 个 PR 提交，每份含：
测试代码
对应文档更新（如特征是否“跨标的可比”）
必要的特征函数微调（如加缺列检查）

请指示优先方向（A/B/C），我即可开始编码。

非常清晰的「特征测试标准」清单与当前覆盖情况梳理，整体结构合理、覆盖重点明确。以下是对现状的总结，并结合你的待补/改进建议，给出优先级排序 + 补位建议 + 可执行方案，便于后续落地。

一、现状评估（按标准逐项）

标准 当前覆盖情况 覆盖强度 备注
------ --------------- ---------- ------
无未来数据 部分覆盖（VPIN/RSI/Hurst 等有容忍度检查） 中 缺少对窗口右闭左开的精确验证，尤其在 Footprint / 频谱类中未显式验证
NaN/Inf 防护 覆盖良好 高 多个模块已加 eps + replace(inf, nan)，且允许 NaN 输出
归一化/可比性 几乎无覆盖 低 仅 VPIN 多维特征隐含部分可比性，但无跨标的缩放验证
边界/空数据 部分覆盖（Footprint 有空 bar NaN） 中 缺列/空输入场景未系统测试
稳定性/数值容忍 部分覆盖（VPIN/TC 有状态连续性） 中 未统一 assert 容差策略，随机种子未强制
数据类型/索引对齐 未显式测试 低 聚合测试隐含部分对齐，但无 merge/index 重复等边界测试
单位/字段要求 未覆盖 低 缺列错误处理缺失
缓存/状态 覆盖良好 高 VPIN/TC 的流式、跨月、回退均有测试
✅ 强项：Inf/Nan 防护、缓存/状态管理
⚠️ 短板：归一化可比性、无未来数据精确验证、缺列报错、索引对齐

二、补位建议（按优先级排序）
🔴 P0（高优）：归一化/可比性 + 缺列报错
理由：
归一化是跨资产建模的核心前提，若不可比将导致模型失效；
缺列报错属于基础健壮性，用户体验关键。
建议动作：
1. 新增通用辅助函数 assert_normalized_invariance(feature_fn, scale_factors=[(10, 0.1), (100, 0.01)])：
输入两组同分布但价格/volume 缩放后的数据；
断言输出（如 zscore、量价比）相对误差 < tol（如 1e-5）。
2. 为以下模块添加缺列测试：
compute_footprint_features
extract_order_flow_features
所有依赖 price, volume, side, tick_size 的特征函数
验证抛出 ValueError 且含“missing column: xxx”字样。
🟠 P1（中优）：无未来数据精确验证 + 频谱类（Hilbert/WPT/EVT）补测
理由：
频谱类特征天然易引入未来信息（如 FFT 全局窗）；
当前“容忍度检查”不够严谨，需构造单调标签+时间戳对齐验证。
建议动作：
1. 构造合成数据：t=0..N，标签 y_t = t（严格递增）；
2. 对每个 t，验证 feature(t) 与 y_{t+1} 相关系数 ≈ 0（或 p > 0.05）；
3. 为 Hilbert/WPT/EVT 添加最小窗口测试（如 window=10），验证：
前 window-1 个输出为 NaN；
第 window 个输出仅依赖 [0, window-1]；
无 inf/nan（恢复之前移除的 inf guard）。
🟡 P2（低优但必要）：索引对齐 + 单元测试容差标准化
建议动作：
在 conftest.py 或 test_utils.py 中定义：
python
def assert_series_equal(a, b, rtol=1e-5, atol=1e-8):
pd.testing.assert_series_equal(a, b, rtol=rtol, atol=atol, check_names=False)
统一替换所有 == 或 np.allclose 为该函数；
新增测试：merge 后索引去重、tick 与 K 线时间包含关系（如 tick in [open_time, close_time)）。

三、可执行下一步（我可直接落地）

请确认以下任一选项，我将生成完整测试代码：
✅ 选项 A（推荐）：先补 P0 —— 归一化可比性 + 缺列报错
覆盖模块：Footprint / VPIN / Trade Clustering / 所有 zscore/百分比类特征
输出：
test_utils.py：新增 assert_normalized_invariance
test_feature_robustness.py：缺列报错测试
各模块补充 1~2 个可比性测试用例
✅ 选项 B：先补 P1 —— Hilbert/WPT/EVT 无未来 + 稳定性
输出：
test_spectral_features.py：含 lookahead 验证 + inf guard + 最小窗口
恢复并加固之前移除的频谱类 inf 防护
✅ 选项 C：全量推进（分 PR）
我可按 P0 → P1 → P2 顺序，分 3 个 PR 提交，每份含：
测试代码
对应文档更新（如特征是否“跨标的可比”）
必要的特征函数微调（如加缺列检查）

请指示优先方向（A/B/C），我即可开始编码。