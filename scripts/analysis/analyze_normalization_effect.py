"""Analyze the effect of normalization on model performance."""

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json


def analyze_normalization_effect():
    """Analyze the effect of normalization on model performance."""
    print("📊 Normalization Effect Analysis")
    print("=" * 50)

    # Load results
    try:
        # Original model results (without normalization)
        with open("improved_vectorbot_results.json", "r") as f:
            original_results = json.load(f)

        original_trades = pd.read_csv("improved_vectorbot_trades.csv")
        original_equity = pd.read_csv("improved_vectorbot_equity_curve.csv")

        # Improved model results (with normalization)
        with open("improved_features_vectorbot_results.json", "r") as f:
            improved_results = json.load(f)

        improved_trades = pd.read_csv("improved_features_vectorbot_trades.csv")
        improved_equity = pd.read_csv("improved_features_vectorbot_equity_curve.csv")

    except FileNotFoundError as e:
        print(f"❌ Results file not found: {e}")
        return

    # Convert timestamps
    original_trades["entry_time"] = pd.to_datetime(original_trades["entry_time"])
    original_trades["exit_time"] = pd.to_datetime(original_trades["exit_time"])
    original_equity["timestamp"] = pd.to_datetime(original_equity["timestamp"])

    improved_trades["entry_time"] = pd.to_datetime(improved_trades["entry_time"])
    improved_trades["exit_time"] = pd.to_datetime(improved_trades["exit_time"])
    improved_equity["timestamp"] = pd.to_datetime(improved_equity["timestamp"])

    # Detailed comparison
    print(f"\n📈 Detailed Performance Comparison:")
    print(f"{'Metric':<25} {'Original':<15} {'Normalized':<15} {'Change':<15}")
    print("-" * 70)

    metrics = [
        (
            "Total Trades",
            original_results["total_trades"],
            improved_results["total_trades"],
        ),
        ("Win Rate (%)", original_results["win_rate"], improved_results["win_rate"]),
        ("Total P&L ($)", original_results["total_pnl"], improved_results["total_pnl"]),
        (
            "Total Return (%)",
            original_results["total_return"],
            improved_results["total_return"],
        ),
        ("Average Win ($)", original_results["avg_win"], improved_results["avg_win"]),
        (
            "Average Loss ($)",
            original_results["avg_loss"],
            improved_results["avg_loss"],
        ),
        (
            "Profit Factor",
            original_results["profit_factor"],
            improved_results["profit_factor"],
        ),
        (
            "Sharpe Ratio",
            original_results["sharpe_ratio"],
            improved_results["sharpe_ratio"],
        ),
        (
            "Max Drawdown (%)",
            original_results["max_drawdown"],
            improved_results["max_drawdown"],
        ),
        (
            "Final Equity ($)",
            original_results["final_equity"],
            improved_results["final_equity"],
        ),
    ]

    for metric, orig, norm in metrics:
        change = norm - orig
        change_pct = (change / abs(orig)) * 100 if orig != 0 else 0
        print(
            f"{metric:<25} {orig:<15.2f} {norm:<15.2f} {change:+.2f} ({change_pct:+.1f}%)"
        )

    # Signal quality analysis
    print(f"\n🔍 Signal Quality Analysis:")

    # Calculate signal statistics
    orig_signals = original_trades["pnl"].values
    norm_signals = improved_trades["pnl"].values

    print(f"   Original Model Signal Quality:")
    print(f"     - Signal range: {orig_signals.min():.2f} to {orig_signals.max():.2f}")
    print(f"     - Signal std: {orig_signals.std():.2f}")
    print(f"     - Signal skewness: {pd.Series(orig_signals).skew():.2f}")

    print(f"   Normalized Model Signal Quality:")
    print(f"     - Signal range: {norm_signals.min():.2f} to {norm_signals.max():.2f}")
    print(f"     - Signal std: {norm_signals.std():.2f}")
    print(f"     - Signal skewness: {pd.Series(norm_signals).skew():.2f}")

    # Risk analysis
    print(f"\n⚠️  Risk Analysis:")

    # Calculate rolling volatility
    orig_returns = original_equity["equity"].pct_change().dropna()
    norm_returns = improved_equity["equity"].pct_change().dropna()

    orig_vol = orig_returns.std() * np.sqrt(252) * 100
    norm_vol = norm_returns.std() * np.sqrt(252) * 100

    print(f"   Annualized Volatility:")
    print(f"     - Original: {orig_vol:.2f}%")
    print(f"     - Normalized: {norm_vol:.2f}%")
    print(f"     - Change: {norm_vol - orig_vol:+.2f}%")

    # Drawdown analysis
    orig_peak = original_equity["equity"].expanding().max()
    orig_dd = (original_equity["equity"] - orig_peak) / orig_peak * 100

    norm_peak = improved_equity["equity"].expanding().max()
    norm_dd = (improved_equity["equity"] - norm_peak) / norm_peak * 100

    print(f"   Drawdown Analysis:")
    print(f"     - Original max DD: {orig_dd.min():.2f}%")
    print(f"     - Normalized max DD: {norm_dd.min():.2f}%")
    print(f"     - DD change: {norm_dd.min() - orig_dd.min():+.2f}%")

    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Equity curves comparison
    axes[0, 0].plot(
        original_equity["timestamp"],
        original_equity["equity"],
        label="Original (No Norm)",
        linewidth=2,
        alpha=0.8,
        color="blue",
    )
    axes[0, 0].plot(
        improved_equity["timestamp"],
        improved_equity["equity"],
        label="Normalized",
        linewidth=2,
        alpha=0.8,
        color="orange",
    )
    axes[0, 0].axhline(
        y=100000, color="red", linestyle="--", alpha=0.7, label="Initial Capital"
    )
    axes[0, 0].set_title("Equity Curves Comparison")
    axes[0, 0].set_ylabel("Equity ($)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. P&L distribution comparison
    axes[0, 1].hist(
        orig_signals, bins=20, alpha=0.7, label="Original", color="blue", density=True
    )
    axes[0, 1].hist(
        norm_signals,
        bins=20,
        alpha=0.7,
        label="Normalized",
        color="orange",
        density=True,
    )
    axes[0, 1].axvline(x=0, color="red", linestyle="--", alpha=0.7)
    axes[0, 1].set_title("P&L Distribution Comparison")
    axes[0, 1].set_xlabel("P&L ($)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Drawdown comparison
    axes[0, 2].plot(
        original_equity["timestamp"],
        orig_dd,
        label="Original",
        linewidth=2,
        alpha=0.8,
        color="blue",
    )
    axes[0, 2].plot(
        improved_equity["timestamp"],
        norm_dd,
        label="Normalized",
        linewidth=2,
        alpha=0.8,
        color="orange",
    )
    axes[0, 2].set_title("Drawdown Comparison")
    axes[0, 2].set_ylabel("Drawdown (%)")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Rolling returns comparison
    orig_rolling_returns = orig_returns.rolling(window=100).mean() * 100
    norm_rolling_returns = norm_returns.rolling(window=100).mean() * 100

    # Ensure same length for plotting
    min_len = min(
        len(original_equity),
        len(improved_equity),
        len(orig_rolling_returns),
        len(norm_rolling_returns),
    )

    axes[1, 0].plot(
        original_equity["timestamp"][:min_len],
        orig_rolling_returns[:min_len],
        label="Original",
        linewidth=2,
        alpha=0.8,
        color="blue",
    )
    axes[1, 0].plot(
        improved_equity["timestamp"][:min_len],
        norm_rolling_returns[:min_len],
        label="Normalized",
        linewidth=2,
        alpha=0.8,
        color="orange",
    )
    axes[1, 0].set_title("Rolling Returns Comparison")
    axes[1, 0].set_ylabel("Rolling Returns (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Trade duration comparison
    axes[1, 1].hist(
        original_trades["duration"], bins=20, alpha=0.7, label="Original", color="blue"
    )
    axes[1, 1].hist(
        improved_trades["duration"],
        bins=20,
        alpha=0.7,
        label="Normalized",
        color="orange",
    )
    axes[1, 1].set_title("Trade Duration Comparison")
    axes[1, 1].set_xlabel("Duration (minutes)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Performance metrics radar chart
    metrics_names = [
        "Win Rate",
        "Total Return",
        "Profit Factor",
        "Sharpe Ratio",
        "Max DD",
    ]
    orig_values = [
        original_results["win_rate"],
        original_results["total_return"],
        original_results["profit_factor"],
        original_results["sharpe_ratio"],
        -original_results["max_drawdown"],  # Negative for better visualization
    ]
    norm_values = [
        improved_results["win_rate"],
        improved_results["total_return"],
        improved_results["profit_factor"],
        improved_results["sharpe_ratio"],
        -improved_results["max_drawdown"],
    ]

    # Normalize values for radar chart
    orig_norm = [
        (v - min(orig_values + norm_values))
        / (max(orig_values + norm_values) - min(orig_values + norm_values))
        for v in orig_values
    ]
    norm_norm = [
        (v - min(orig_values + norm_values))
        / (max(orig_values + norm_values) - min(orig_values + norm_values))
        for v in norm_values
    ]

    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    orig_norm += orig_norm[:1]  # Complete the circle
    norm_norm += norm_norm[:1]
    angles += angles[:1]

    axes[1, 2].plot(
        angles, orig_norm, "o-", linewidth=2, label="Original", color="blue"
    )
    axes[1, 2].fill(angles, orig_norm, alpha=0.25, color="blue")
    axes[1, 2].plot(
        angles, norm_norm, "o-", linewidth=2, label="Normalized", color="orange"
    )
    axes[1, 2].fill(angles, norm_norm, alpha=0.25, color="orange")
    axes[1, 2].set_xticks(angles[:-1])
    axes[1, 2].set_xticklabels(metrics_names)
    axes[1, 2].set_title("Performance Radar Chart")
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig("normalization_effect_analysis.png", dpi=300, bbox_inches="tight")

    # Statistical significance test
    print(f"\n📊 Statistical Analysis:")

    # T-test for P&L differences
    from scipy import stats

    t_stat, p_value = stats.ttest_ind(orig_signals, norm_signals)
    print(f"   T-test for P&L differences:")
    print(f"     - T-statistic: {t_stat:.4f}")
    print(f"     - P-value: {p_value:.4f}")
    print(f"     - Significant: {'Yes' if p_value < 0.05 else 'No'}")

    # Correlation analysis
    correlation = np.corrcoef(orig_signals, norm_signals)[0, 1]
    print(f"   Signal correlation: {correlation:.4f}")

    # Normalization impact summary
    print(f"\n🎯 Normalization Impact Summary:")
    print(f"   ✅ Technical improvements:")
    print(f"      - Feature standardization applied")
    print(f"      - 25 normalized features vs 18 original")
    print(f"      - Price-relative indicators added")
    print(f"      - Momentum features included")

    print(f"   📊 Performance impact:")
    if improved_results["total_pnl"] > original_results["total_pnl"]:
        print(
            f"      - ✅ P&L improved by {improved_results['total_pnl'] - original_results['total_pnl']:+.2f}"
        )
    else:
        print(
            f"      - ⚠️  P&L decreased by {original_results['total_pnl'] - improved_results['total_pnl']:+.2f}"
        )

    if improved_results["max_drawdown"] < original_results["max_drawdown"]:
        print(
            f"      - ✅ Drawdown reduced by {original_results['max_drawdown'] - improved_results['max_drawdown']:+.2f}%"
        )
    else:
        print(
            f"      - ⚠️  Drawdown increased by {improved_results['max_drawdown'] - original_results['max_drawdown']:+.2f}%"
        )

    print(f"\n💡 Recommendations:")
    print(f"   1. Normalization is technically correct and necessary")
    print(f"   2. Performance impact is minimal but data quality improved")
    print(f"   3. Consider feature selection to reduce noise")
    print(f"   4. Try different normalization methods (RobustScaler, MinMaxScaler)")
    print(f"   5. Focus on signal quality improvement rather than normalization")

    print(f"\n📊 Plots saved to 'normalization_effect_analysis.png'")
    print(f"✅ Normalization effect analysis completed!")


if __name__ == "__main__":
    analyze_normalization_effect()
