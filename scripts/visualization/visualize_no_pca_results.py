#!/usr/bin/env python3
"""可视化No PCA版本的滚动训练结果"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 读取数据
df = pd.read_csv(
    "results/monthly_rolling_2025_advanced/monthly_results_advanced_2025.csv"
)

# 创建图表
fig = plt.figure(figsize=(16, 12))

# 1. 月度收益柱状图
ax1 = plt.subplot(3, 2, 1)
colors = [
    "green" if x > 5 else "lightgreen" if x > 0 else "red" for x in df["total_return"]
]
bars = ax1.bar(
    df["test_month"], df["total_return"], color=colors, alpha=0.7, edgecolor="black"
)
ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
ax1.axhline(
    y=df["total_return"].mean(),
    color="blue",
    linestyle="--",
    linewidth=2,
    label=f'平均: {df["total_return"].mean():.2f}%',
)
ax1.set_xlabel("月份")
ax1.set_ylabel("收益率 (%)")
ax1.set_title("📈 月度收益率", fontsize=14, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, df["total_return"])):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        val,
        f"{val:.1f}%",
        ha="center",
        va="bottom" if val > 0 else "top",
        fontsize=9,
    )

# 2. 累计收益曲线
ax2 = plt.subplot(3, 2, 2)
cumulative_return = (1 + df["total_return"] / 100).cumprod() - 1
ax2.plot(
    range(len(df)),
    cumulative_return * 100,
    marker="o",
    linewidth=2.5,
    color="darkgreen",
    markersize=8,
    markerfacecolor="lightgreen",
    markeredgecolor="darkgreen",
)
ax2.fill_between(range(len(df)), 0, cumulative_return * 100, alpha=0.3, color="green")
ax2.set_xlabel("月份")
ax2.set_ylabel("累计收益率 (%)")
ax2.set_title(
    f"📊 累计收益曲线 (总计: {cumulative_return.iloc[-1]*100:.2f}%)",
    fontsize=14,
    fontweight="bold",
)
ax2.set_xticks(range(len(df)))
ax2.set_xticklabels(df["test_month"])
ax2.grid(True, alpha=0.3)
for i, val in enumerate(cumulative_return * 100):
    ax2.annotate(
        f"{val:.1f}%",
        (i, val),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=8,
    )

# 3. 胜率趋势
ax3 = plt.subplot(3, 2, 3)
ax3.plot(
    df["test_month"],
    df["win_rate"],
    marker="s",
    linewidth=2,
    color="purple",
    markersize=8,
    markerfacecolor="lavender",
    markeredgecolor="purple",
)
ax3.axhline(y=50, color="red", linestyle="--", linewidth=2, label="50%基准线")
ax3.axhline(
    y=df["win_rate"].mean(),
    color="blue",
    linestyle="--",
    linewidth=2,
    label=f'平均: {df["win_rate"].mean():.1f}%',
)
ax3.set_xlabel("月份")
ax3.set_ylabel("胜率 (%)")
ax3.set_title("🎯 胜率趋势", fontsize=14, fontweight="bold")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 100)

# 4. 利润因子
ax4 = plt.subplot(3, 2, 4)
colors_pf = [
    "darkgreen" if x > 2 else "green" if x > 1.5 else "orange" if x > 1 else "red"
    for x in df["profit_factor"]
]
bars = ax4.bar(
    df["test_month"], df["profit_factor"], color=colors_pf, alpha=0.7, edgecolor="black"
)
ax4.axhline(y=1.0, color="red", linestyle="-", linewidth=1, label="盈亏平衡线")
ax4.axhline(
    y=df["profit_factor"].mean(),
    color="blue",
    linestyle="--",
    linewidth=2,
    label=f'平均: {df["profit_factor"].mean():.2f}',
)
ax4.set_xlabel("月份")
ax4.set_ylabel("利润因子")
ax4.set_title("💰 利润因子", fontsize=14, fontweight="bold")
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Sharpe Ratio
ax5 = plt.subplot(3, 2, 5)
ax5.plot(
    df["test_month"],
    df["sharpe_ratio"],
    marker="D",
    linewidth=2.5,
    color="darkblue",
    markersize=8,
    markerfacecolor="lightblue",
    markeredgecolor="darkblue",
)
ax5.axhline(y=0, color="red", linestyle="-", linewidth=1)
ax5.axhline(
    y=1, color="orange", linestyle="--", linewidth=1, alpha=0.5, label="优秀线(1.0)"
)
ax5.axhline(
    y=df["sharpe_ratio"].mean(),
    color="blue",
    linestyle="--",
    linewidth=2,
    label=f'平均: {df["sharpe_ratio"].mean():.2f}',
)
ax5.set_xlabel("月份")
ax5.set_ylabel("Sharpe Ratio")
ax5.set_title("📐 Sharpe比率趋势", fontsize=14, fontweight="bold")
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. 质量得分
ax6 = plt.subplot(3, 2, 6)
ax6.plot(
    df["test_month"],
    df["quality_score"],
    marker="*",
    linewidth=2.5,
    color="darkred",
    markersize=12,
    markerfacecolor="pink",
    markeredgecolor="darkred",
)
ax6.axhline(
    y=df["quality_score"].mean(),
    color="blue",
    linestyle="--",
    linewidth=2,
    label=f'平均: {df["quality_score"].mean():.2f}',
)
ax6.fill_between(range(len(df)), 0, df["quality_score"], alpha=0.3, color="red")
ax6.set_xlabel("月份")
ax6.set_ylabel("质量得分")
ax6.set_title("⭐ 综合质量得分", fontsize=14, fontweight="bold")
ax6.set_xticks(range(len(df)))
ax6.set_xticklabels(df["test_month"])
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "results/monthly_rolling_2025_advanced/performance_dashboard.png",
    dpi=300,
    bbox_inches="tight",
)
print("✅ 图表已保存: results/monthly_rolling_2025_advanced/performance_dashboard.png")

# 打印详细表格
print("\n" + "=" * 100)
print("📊 详细月度业绩报告".center(100))
print("=" * 100)
print(
    f"\n{'月份':<12} {'收益率':>10} {'交易数':>8} {'胜率':>8} {'利润因子':>10} {'最大回撤':>10} {'Sharpe':>8} {'质量分':>8}"
)
print("-" * 100)

for _, row in df.iterrows():
    ret_str = f"{row['total_return']:>9.2f}%"
    if row["total_return"] > 5:
        ret_color = "\033[92m"  # Green
    elif row["total_return"] > 0:
        ret_color = "\033[93m"  # Yellow
    else:
        ret_color = "\033[91m"  # Red

    print(
        f"{row['test_month']:<12} {ret_color}{ret_str}\033[0m {row['total_trades']:>7} "
        f"{row['win_rate']:>7.1f}% {row['profit_factor']:>9.2f} "
        f"{row['max_drawdown']:>9.2f}% {row['sharpe_ratio']:>7.2f} "
        f"{row['quality_score']:>7.2f}"
    )

print("-" * 100)
print(
    f"{'平均/总计':<12} \033[92m{df['total_return'].mean():>9.2f}%\033[0m {df['total_trades'].sum():>7} "
    f"{df['win_rate'].mean():>7.1f}% {df['profit_factor'].mean():>9.2f} "
    f"{df['max_drawdown'].mean():>9.2f}% {df['sharpe_ratio'].mean():>7.2f} "
    f"{df['quality_score'].mean():>7.2f}"
)
print("=" * 100)

# 计算累计收益
cumulative = (1 + df["total_return"] / 100).cumprod()
final_return = (cumulative.iloc[-1] - 1) * 100

print(f"\n💰 累计6个月收益: \033[92m+{final_return:.2f}%\033[0m")
print(
    f"📊 如果初始资金10万: \033[92m最终: {100000 * cumulative.iloc[-1]:.0f}元 (盈利: {100000 * (cumulative.iloc[-1]-1):.0f}元)\033[0m"
)

# 年化收益
monthly_avg = df["total_return"].mean()
annual_return = (1 + monthly_avg / 100) ** 12 - 1
print(f"📈 年化收益率(简单估算): \033[92m{annual_return*100:.2f}%\033[0m")

print("\n")
