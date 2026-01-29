#!/usr/bin/env python3
"""
汇总长时间段和短时间段的测试结果
"""

import json
from pathlib import Path

STRATEGIES = [
    "sr_reversal_rr_reg_long",
    "compression_breakout",
    "sr_breakout",
    "trend_following",
]


def extract_results(strategy: str):
    """提取策略的测试结果"""
    results = {
        "long_term": None,
        "short_term": None,
    }

    # 长时间段结果
    long_file = Path(f"results/fixed_long/{strategy}/{strategy}/results.json")
    if long_file.exists():
        with open(long_file) as f:
            data = json.load(f)
            bt = data.get("backtest", {})
            results["long_term"] = {
                "sharpe": bt.get("sharpe"),
                "return_pct": bt.get("total_return_pct"),
                "trades": bt.get("total_trades"),
                "win_rate": bt.get("win_rate"),
                "max_drawdown": bt.get("max_drawdown_pct"),
                "train_period": "2023-01-01 → 2024-12-31",
                "test_period": "OOS: 2025-01-01 → 2025-05-31",
            }

    # 短时间段结果
    short_file = Path(f"results/fixed_short/{strategy}/{strategy}/results.json")
    if short_file.exists():
        with open(short_file) as f:
            data = json.load(f)
            bt = data.get("backtest", {})
            results["short_term"] = {
                "sharpe": bt.get("sharpe"),
                "return_pct": bt.get("total_return_pct"),
                "trades": bt.get("total_trades"),
                "win_rate": bt.get("win_rate"),
                "max_drawdown": bt.get("max_drawdown_pct"),
                "train_period": "2024-07-01 → 2024-12-31 (6个月)",
                "test_period": "OOS: 2025-01-01 → 2025-05-31 (5个月)",
            }

    return results


def main():
    print("=" * 80)
    print("汇总测试结果")
    print("=" * 80)

    all_results = {}
    for strategy in STRATEGIES:
        results = extract_results(strategy)
        all_results[strategy] = results

        print(f"\n{strategy}:")
        if results["long_term"]:
            lt = results["long_term"]
            print(f"  长时间段:")
            print(
                f"    Sharpe: {lt['sharpe']:.4f}" if lt["sharpe"] else "    Sharpe: N/A"
            )
            print(
                f"    Return%: {lt['return_pct']:.2f}%"
                if lt["return_pct"]
                else "    Return%: N/A"
            )
            print(f"    Trades: {lt['trades']}" if lt["trades"] else "    Trades: N/A")
            print(f"    训练期: {lt['train_period']}")
            print(f"    测试期: {lt['test_period']}")

        if results["short_term"]:
            st = results["short_term"]
            print(f"  短时间段:")
            print(
                f"    Sharpe: {st['sharpe']:.4f}" if st["sharpe"] else "    Sharpe: N/A"
            )
            print(
                f"    Return%: {st['return_pct']:.2f}%"
                if st["return_pct"]
                else "    Return%: N/A"
            )
            print(f"    Trades: {st['trades']}" if st["trades"] else "    Trades: N/A")
            print(f"    训练期: {st['train_period']}")
            print(f"    测试期: {st['test_period']}")

        # 对比
        if results["long_term"] and results["short_term"]:
            lt_sharpe = results["long_term"]["sharpe"]
            st_sharpe = results["short_term"]["sharpe"]
            if lt_sharpe and st_sharpe:
                diff = st_sharpe - lt_sharpe
                diff_pct = (diff / abs(lt_sharpe) * 100) if lt_sharpe != 0 else 0
                print(f"  对比:")
                print(f"    Sharpe 差异: {diff:+.4f} ({diff_pct:+.1f}%)")
                if diff > 0:
                    print(f"    ✅ 短时间段表现更好")
                else:
                    print(f"    ⚠️ 长时间段表现更好")

    # 保存汇总
    summary_file = Path("results/test_results_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n汇总已保存到: {summary_file}")
    return all_results


if __name__ == "__main__":
    main()
