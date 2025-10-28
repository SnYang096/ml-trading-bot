"""演示安全的特征计算方法 vs 数据泄露的错误方法

对比展示：
1. ❌ 错误：对整个序列做变换
2. ✅ 正确：使用滑动窗口逐点计算
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import pywt


def wavelet_transform_wrong(data):
    """❌ 错误方法：对整个序列做小波变换"""
    # 这会使用所有数据（包括未来）的信息
    coeffs = pywt.wavedec(data, "db4", level=3)

    # 提取能量特征（这个特征值对所有时间点都一样！）
    energy = np.sum([np.sum(c**2) for c in coeffs])

    # 返回所有时间点相同的特征（明显错误）
    return np.full(len(data), energy)


def wavelet_transform_safe(data, window=60):
    """✅ 正确方法：使用滑动窗口逐点计算"""
    n = len(data)
    energies = np.zeros(n)

    for i in range(n):
        if i < window:
            energies[i] = 0  # 窗口不足
        else:
            # 关键：只使用历史数据 [i-window:i]
            window_data = data[i - window : i]

            try:
                coeffs = pywt.wavedec(window_data, "db4", level=3)
                energies[i] = np.sum([np.sum(c**2) for c in coeffs])
            except:
                energies[i] = 0

    return energies


def moving_average_wrong(data, window=20):
    """❌ 错误方法：使用未来数据的移动平均"""
    # center=True 会使用前后的数据
    return pd.Series(data).rolling(window=window, center=True).mean().values


def moving_average_safe(data, window=20):
    """✅ 正确方法：只使用历史数据的移动平均"""
    # center=False（默认）只使用历史数据
    return pd.Series(data).rolling(window=window, center=False).mean().fillna(0).values


def normalization_wrong(train_data, test_data):
    """❌ 错误方法：在全部数据上做归一化"""
    # 合并训练和测试数据
    all_data = np.concatenate([train_data, test_data])

    # 在全部数据上计算均值和标准差（泄露了测试集信息！）
    mean = np.mean(all_data)
    std = np.std(all_data)

    # 归一化
    train_normalized = (train_data - mean) / std
    test_normalized = (test_data - mean) / std

    return train_normalized, test_normalized, mean, std


def normalization_safe(train_data, test_data):
    """✅ 正确方法：只在训练集上fit，测试集上transform"""
    # 只在训练数据上计算统计量
    mean = np.mean(train_data)
    std = np.std(train_data)

    # 使用训练集的统计量归一化
    train_normalized = (train_data - mean) / std
    test_normalized = (test_data - mean) / std

    return train_normalized, test_normalized, mean, std


def demo_comparison():
    """演示对比"""
    print("=" * 80)
    print("安全特征计算 vs 数据泄露对比演示")
    print("=" * 80)

    # 生成模拟数据
    np.random.seed(42)
    n_samples = 200

    # 创建一个有趋势的时间序列
    t = np.linspace(0, 10, n_samples)
    trend = 2 * t
    seasonality = 5 * np.sin(2 * np.pi * t / 10)
    noise = np.random.normal(0, 1, n_samples)
    data = trend + seasonality + noise

    print(f"\n生成模拟数据: {n_samples} 个样本")
    print(f"   数据范围: [{data.min():.2f}, {data.max():.2f}]")

    # === 演示1: 小波变换 ===
    print("\n" + "-" * 80)
    print("演示1: 小波变换特征计算")
    print("-" * 80)

    # 错误方法
    wavelet_wrong = wavelet_transform_wrong(data)
    print(f"\n❌ 错误方法:")
    print(f"   所有时间点的特征值: {wavelet_wrong[0]:.2f}")
    print(f"   特征是否变化: {np.std(wavelet_wrong) > 0}")
    print(f"   问题: 所有时间点的特征值完全相同！")

    # 正确方法
    wavelet_safe = wavelet_transform_safe(data, window=60)
    print(f"\n✅ 正确方法:")
    print(f"   特征值范围: [{wavelet_safe.min():.2f}, {wavelet_safe.max():.2f}]")
    print(f"   特征是否变化: {np.std(wavelet_safe) > 0}")
    print(f"   前60个样本（窗口不足）: 全为0")
    print(f"   第61个样本开始: 使用历史60个数据点计算")

    # === 演示2: 移动平均 ===
    print("\n" + "-" * 80)
    print("演示2: 移动平均")
    print("-" * 80)

    ma_wrong = moving_average_wrong(data, window=20)
    ma_safe = moving_average_safe(data, window=20)

    print(f"\n❌ 错误方法 (center=True):")
    print(f"   前10个值: {ma_wrong[:10]}")
    print(f"   问题: 前10个值使用了后10个数据点！")

    print(f"\n✅ 正确方法 (center=False):")
    print(f"   前10个值: {ma_safe[:10]}")
    print(f"   说明: 前20个样本（窗口不足）为0或NaN")

    # === 演示3: 归一化 ===
    print("\n" + "-" * 80)
    print("演示3: 归一化")
    print("-" * 80)

    # 分割数据
    split_point = 150
    train_data = data[:split_point]
    test_data = data[split_point:]

    # 错误方法
    train_wrong, test_wrong, mean_wrong, std_wrong = normalization_wrong(
        train_data, test_data
    )
    print(f"\n❌ 错误方法:")
    print(f"   使用的均值: {mean_wrong:.4f}")
    print(f"   使用的标准差: {std_wrong:.4f}")
    print(f"   训练集均值: {np.mean(train_data):.4f}")
    print(f"   测试集均值: {np.mean(test_data):.4f}")
    print(f"   问题: 使用了包含测试集的统计量！")

    # 正确方法
    train_safe, test_safe, mean_safe, std_safe = normalization_safe(
        train_data, test_data
    )
    print(f"\n✅ 正确方法:")
    print(f"   使用的均值: {mean_safe:.4f} (只来自训练集)")
    print(f"   使用的标准差: {std_safe:.4f} (只来自训练集)")
    print(f"   训练集归一化后均值: {np.mean(train_safe):.6f} (应接近0)")
    print(f"   训练集归一化后标准差: {np.std(train_safe):.6f} (应接近1)")
    print(f"   测试集归一化后均值: {np.mean(test_safe):.6f} (可能不为0)")
    print(f"   说明: 测试集使用训练集的统计量，所以均值不一定为0")

    # === 可视化对比 ===
    print("\n" + "-" * 80)
    print("生成可视化对比图...")
    print("-" * 80)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 图1: 小波变换对比
    ax1 = axes[0]
    ax1.plot(data, label="原始数据", alpha=0.5, linewidth=1)
    ax1.plot(
        wavelet_wrong / 1000, label="❌ 错误方法 (scaled)", linewidth=2, color="red"
    )
    ax1.plot(
        wavelet_safe / 1000, label="✅ 正确方法 (scaled)", linewidth=2, color="green"
    )
    ax1.axvline(60, color="orange", linestyle="--", alpha=0.5, label="窗口大小=60")
    ax1.set_title("演示1: 小波变换特征计算对比", fontsize=12, fontweight="bold")
    ax1.set_xlabel("时间步")
    ax1.set_ylabel("特征值 (scaled)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2: 移动平均对比
    ax2 = axes[1]
    ax2.plot(data, label="原始数据", alpha=0.5, linewidth=1)
    ax2.plot(
        ma_wrong,
        label="❌ 错误方法 (center=True)",
        linewidth=2,
        color="red",
        linestyle="--",
    )
    ax2.plot(ma_safe, label="✅ 正确方法 (center=False)", linewidth=2, color="green")
    ax2.axvline(20, color="orange", linestyle="--", alpha=0.5, label="窗口大小=20")
    ax2.set_title("演示2: 移动平均对比", fontsize=12, fontweight="bold")
    ax2.set_xlabel("时间步")
    ax2.set_ylabel("值")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3: 归一化对比
    ax3 = axes[2]

    # 绘制原始数据和分割点
    ax3.plot(
        range(len(train_data)), train_data, label="训练数据", color="blue", linewidth=2
    )
    ax3.plot(
        range(split_point, len(data)),
        test_data,
        label="测试数据",
        color="purple",
        linewidth=2,
    )
    ax3.axvline(
        split_point,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"分割点 ({split_point})",
    )

    # 叠加归一化后的数据
    ax3_twin = ax3.twinx()
    ax3_twin.plot(
        range(len(train_safe)),
        train_safe,
        label="✅ 训练集归一化",
        color="green",
        linewidth=1,
        alpha=0.7,
    )
    ax3_twin.plot(
        range(split_point, len(data)),
        test_safe,
        label="✅ 测试集归一化",
        color="lightgreen",
        linewidth=1,
        alpha=0.7,
    )
    ax3_twin.set_ylabel("归一化后的值", color="green")
    ax3_twin.tick_params(axis="y", labelcolor="green")

    ax3.set_title("演示3: 归一化对比", fontsize=12, fontweight="bold")
    ax3.set_xlabel("时间步")
    ax3.set_ylabel("原始值", color="blue")
    ax3.tick_params(axis="y", labelcolor="blue")
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(
        script_dir, "..", "..", "docs", "safe_feature_calculation_demo.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ 图片已保存: {output_path}")

    # 显示图片
    # plt.show()

    print("\n" + "=" * 80)
    print("✅ 演示完成！")
    print("=" * 80)


def demo_sliding_window_details():
    """详细演示滑动窗口的索引"""
    print("\n" + "=" * 80)
    print("滑动窗口索引详解")
    print("=" * 80)

    # 创建简单数据
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    window = 3

    print(f"\n数据: {data}")
    print(f"窗口大小: {window}")
    print(f"\n逐点计算过程:")

    for i in range(len(data)):
        if i < window:
            print(f"   i={i}: 窗口不足 (需要{window}个历史数据)")
        else:
            window_data = data[i - window : i]
            print(
                f"   i={i}: window_data = data[{i-window}:{i}] = {window_data} -> 平均值 = {np.mean(window_data):.2f}"
            )

    print(f"\n关键点:")
    print(f"   1. data[i-window:i] 不包含第i个元素")
    print(f"   2. 第i个元素的特征只依赖 data[i-window], ..., data[i-1]")
    print(f"   3. 前{window}个元素（i<{window}）无法计算，设为0或NaN")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("📊 安全特征计算演示")
    print("=" * 80)

    # 基础演示
    demo_comparison()

    # 滑动窗口详解
    demo_sliding_window_details()

    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print(
        """
    ✅ 安全的特征计算原则:
    
    1. 滚动窗口: 使用 data[i-window:i]，不包含当前和未来
    2. 移动平均: 使用 center=False（默认）
    3. 归一化: 训练集fit，测试集transform
    4. 累计指标: 使用变化率或滚动窗口累计
    5. 标签: 可以使用未来数据（这是正常的）
    
    ❌ 避免的错误:
    
    1. 对整个序列做变换（小波、FFT等）
    2. 使用 center=True 的移动平均
    3. 在整个数据集上fit scaler
    4. 使用 shift(-n) 创建特征（只能用于标签）
    5. 跨期合并累计指标
    """
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
