"""比较基线模型(Wavelet)和增强模型(WPT+Hurst+Entropy)的性能."""

import pickle
import json

print("=" * 80)
print("📊 模型对比分析")
print("=" * 80)

# 1. 加载训练指标
print("\n### 1. 训练阶段对比（5T周期）\n")

print("**基线模型 (Wavelet):**")
with open("model_info_wavelet_may_2025.json", "r") as f:
    baseline_info = json.load(f)
    baseline_metrics = baseline_info["metrics"]["stage1"]["5T"]
    print(
        f"  CV准确率: {baseline_metrics['cv_accuracy']*100:.2f}% ± {baseline_metrics['cv_accuracy_std']*100:.2f}%"
    )

print("\n**增强模型 (WPT+Hurst+Entropy):**")
with open("model_info_enhanced_may_2025.json", "r") as f:
    enhanced_info = json.load(f)
    enhanced_metrics = enhanced_info["metrics"]["stage1"]["5T"]
    print(
        f"  CV准确率: {enhanced_metrics['cv_accuracy']*100:.2f}% ± {enhanced_metrics['cv_accuracy_std']*100:.2f}%"
    )

# 2. OOS测试对比
print("\n### 2. OOS测试对比（6-9月）\n")

with open("oos_test_results_with_timeseries_cv.json", "r") as f:
    baseline_oos = json.load(f)

print("**基线模型 OOS表现：**")
print("\n| 月份 | 准确率 | 胜率 | 收益率 | 交易次数 |")
print("|------|--------|------|--------|----------|")
for month, data in baseline_oos.items():
    if "5T" in data:
        d = data["5T"]
        print(
            f"| {month} | {d['accuracy']*100:.2f}% | {d['win_rate']*100:.2f}% | {d['total_return']*100:.2f}% | {d['num_signals']} |"
        )

# 计算平均
baseline_avg_acc = sum([baseline_oos[m]["5T"]["accuracy"] for m in baseline_oos]) / len(
    baseline_oos
)
baseline_avg_wr = sum([baseline_oos[m]["5T"]["win_rate"] for m in baseline_oos]) / len(
    baseline_oos
)
baseline_avg_ret = sum(
    [baseline_oos[m]["5T"]["total_return"] for m in baseline_oos]
) / len(baseline_oos)

print(
    f"\n**平均值**: {baseline_avg_acc*100:.2f}% 准确率, {baseline_avg_wr*100:.2f}% 胜率, {baseline_avg_ret*100:.2f}% 收益"
)

print("\n" + "=" * 80)
print("✅ 对比完成")
print("=" * 80)

print("\n### 📝 结论\n")
print(f"**训练阶段:**")
print(f"  基线模型 CV准确率: {baseline_metrics['cv_accuracy']*100:.2f}%")
print(f"  增强模型 CV准确率: {enhanced_metrics['cv_accuracy']*100:.2f}%")
print(
    f"  差异: {(enhanced_metrics['cv_accuracy'] - baseline_metrics['cv_accuracy'])*100:.2f}%"
)

if enhanced_metrics["cv_accuracy"] < baseline_metrics["cv_accuracy"]:
    print(f"\n⚠️  增强模型训练准确率较低，可能原因：")
    print(f"  1. WPT+Hurst特征较多（66个），可能需要更多数据")
    print(f"  2. 某些高级特征可能包含噪声")
    print(f"  3. 需要在OOS数据上验证是否真的更差")
else:
    print(f"\n✅ 增强模型训练准确率更高")

print(f"\n**建议:**")
print(f"  1. 需要在OOS数据（6-9月）上测试增强模型")
print(f"  2. 比较两个模型的实际表现")
print(f"  3. 分析特征重要性，可能需要特征选择")
