import os, json, pandas as pd
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def run_month_backtest(zip_path: str, month_name: str):
    """Run backtest for a specific month using the existing oos_june infrastructure"""
    print(f"Running OOS for {zip_path} -> {month_name}")

    # Create results directory
    results_dir = os.path.join("results", f"{month_name}_2025_oos")
    os.makedirs(results_dir, exist_ok=True)

    # Import and modify the oos_june script
    import subprocess
    import tempfile

    # Create a modified version of oos_june.py for this month
    with open("scripts/oos_june.py", "r") as f:
        oos_code = f.read()

    # Replace the zip path and output directory
    modified_code = oos_code.replace(
        "data/raw/BTCUSDT-aggTrades-2025-06.zip", zip_path
    ).replace("results/june_2025_oos", results_dir)

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(modified_code)
        temp_script = f.name

    try:
        # Run the modified script
        result = subprocess.run(
            ["python3", temp_script], cwd=os.getcwd(), capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"Error running {month_name} backtest:")
            print(result.stderr)
        else:
            print(f"Successfully completed {month_name} backtest")
            print(result.stdout)

    finally:
        # Clean up temporary file
        os.unlink(temp_script)

    return results_dir


def generate_drift_summary():
    """Generate a summary of drift analysis results"""
    months = ["june", "july", "august", "september"]
    timeframes = ["5T", "15T", "60T", "240T"]

    # Collect results
    all_results = {}
    for month in months:
        month_dir = os.path.join("results", f"{month}_2025_oos")
        if os.path.exists(month_dir):
            all_results[month] = {}
            for tf in timeframes:
                result_file = os.path.join(month_dir, f"wavelet_{tf}_results.json")
                if os.path.exists(result_file):
                    with open(result_file, "r") as f:
                        all_results[month][tf] = json.load(f)

    # Create comparison table
    comparison_data = []
    for month in months:
        if month in all_results:
            for tf in timeframes:
                if tf in all_results[month]:
                    res = all_results[month][tf]
                    comparison_data.append(
                        {
                            "Month": month.capitalize(),
                            "Timeframe": tf,
                            "Total_Trades": res.get("total_trades", 0),
                            "Win_Rate": f"{res.get('win_rate', 0):.1f}%",
                            "Total_Return": f"{res.get('total_return', 0):.2f}%",
                            "Max_Drawdown": f"{res.get('max_drawdown', 0):.2f}%",
                            "Profit_Factor": f"{res.get('profit_factor', 0):.2f}",
                            "Final_Equity": f"{res.get('final_equity', 0):.0f}",
                        }
                    )

    # Save comparison
    df_comparison = pd.DataFrame(comparison_data)
    os.makedirs("reports", exist_ok=True)
    df_comparison.to_csv("reports/drift_analysis_comparison.csv", index=False)

    print("=== Model Drift Analysis ===")
    print(df_comparison.to_string(index=False))

    # Analyze drift
    print("\n=== Drift Analysis ===")
    for tf in timeframes:
        print(f"\n{tf} Timeframe:")
        tf_data = df_comparison[df_comparison["Timeframe"] == tf]
        if len(tf_data) > 1:
            returns = [float(x.replace("%", "")) for x in tf_data["Total_Return"]]
            drawdowns = [float(x.replace("%", "")) for x in tf_data["Max_Drawdown"]]

            print(f"  Return range: {min(returns):.2f}% to {max(returns):.2f}%")
            print(f"  Drawdown range: {min(drawdowns):.2f}% to {max(drawdowns):.2f}%")

            # Check for significant drift
            if max(returns) - min(returns) > 20:
                print(f"  ⚠️  Significant return drift detected!")
            if max(drawdowns) - min(drawdowns) > 10:
                print(f"  ⚠️  Significant drawdown drift detected!")

    return df_comparison


if __name__ == "__main__":
    # Run backtests for each month
    months_data = [
        (
            "/home/yin/trading/rlbot/ml_project/data/aggTrades/BTCUSDT-aggTrades-2025-07.zip",
            "july",
        ),
        (
            "/home/yin/trading/rlbot/ml_project/data/aggTrades/BTCUSDT-aggTrades-2025-08.zip",
            "august",
        ),
        (
            "/home/yin/trading/rlbot/ml_project/data/aggTrades/BTCUSDT-aggTrades-2025-09.zip",
            "september",
        ),
    ]

    for zip_path, month_name in months_data:
        if os.path.exists(zip_path):
            run_month_backtest(zip_path, month_name)
        else:
            print(f"Missing zip: {zip_path}")

    # Generate drift analysis report
    generate_drift_summary()
