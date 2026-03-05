#!/usr/bin/env python3
"""
Gate Subset Selection — 在通过质量检查的 gate 规则上做组合优选.

prefilter 规则始终保留 (frozen), 只对 optimized 规则做子集搜索:
  exhaustive:     枚举所有 2^n 子集 (n≤5), 选 Sharpe 最优
  leave_one_out:  每次去掉一条, 选最优 (n+1 次 backtest)
  all:            保持全部规则 (不做选择, noop)

用法:
  python scripts/select_gate_subset.py \
    --strategy me \
    --strategies-root results/.../strategies \
    --method exhaustive \
    --start-date 2025-08-01 --end-date 2026-02-01 \
    --promote
"""
import argparse
import copy
import itertools
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


# ── Helpers ──────────────────────────────────────────────────────────────


def _is_prefilter(rule: dict) -> bool:
    """判断规则是否为 prefilter (frozen, 不参与子集选择)."""
    rid = rule.get("id", "")
    return (
        rid.startswith("prefilter_")
        or rule.get("frozen", False)
        or rule.get("phase") == "system_safety"
    )


def _subset_label(optimized: list, indices: list) -> str:
    """生成子集的可读标签."""
    if not indices:
        return "prefilter_only"
    names = [optimized[i].get("id", f"rule_{i}") for i in indices]
    return "+".join(n.replace("gate_", "") for n in names)


def _write_gate_yaml(
    template: dict,
    prefilter: list,
    subset_rules: list,
    output_path: Path,
) -> None:
    """写入 gate.yaml, 保留 schema/guardrails, 只替换 hard_gates."""
    config = copy.deepcopy(template)
    config["hard_gates"] = prefilter + subset_rules
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Gate Subset Selection — auto-generated\n")
        yaml.dump(
            config, f, allow_unicode=True, default_flow_style=False, sort_keys=False
        )


def _run_backtest(
    strategy: str,
    strategies_root: str,
    start_date: str,
    end_date: str,
    data_path: str,
    output_json: Path,
) -> dict:
    """运行 event_backtest, 返回 metrics dict."""
    cmd = [
        sys.executable,
        "scripts/event_backtest.py",
        "--strategy",
        strategy,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--data-path",
        data_path,
        "--strategies-root",
        strategies_root,
        "--output",
        str(output_json),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return {"error": r.stderr[-300:] if r.stderr else "unknown"}
    try:
        return json.loads(output_json.read_text())
    except Exception as e:
        return {"error": str(e)}


# ── Main ─────────────────────────────────────────────────────────────────


def select_best_subset(
    strategy: str,
    strategies_root: str,
    method: str,
    start_date: str,
    end_date: str,
    data_path: str = "data/parquet_data",
    min_trades: int = 30,
    promote: bool = False,
) -> dict:
    """
    对 gate 规则做子集选择, 返回最优子集信息.

    Returns:
        {"best_indices": [...], "best_sharpe": float, "results": [...]}
    """
    gate_path = Path(strategies_root) / strategy / "archetypes" / "gate.yaml"
    if not gate_path.exists():
        print(f"❌ gate.yaml not found: {gate_path}")
        return {"error": "gate.yaml not found"}

    config = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    all_rules = config.get("hard_gates", [])

    prefilter = [r for r in all_rules if _is_prefilter(r)]
    optimized = [r for r in all_rules if not _is_prefilter(r)]

    n = len(optimized)
    print(f"\n{'='*80}")
    print(f"  Gate Subset Selection: {strategy.upper()}")
    print(f"  method={method}, prefilter={len(prefilter)}, optimized={n}")
    print(f"{'='*80}")

    if n == 0 or method == "all":
        print("  ℹ️  无 optimized 规则或 method=all, 跳过选择")
        return {"best_indices": [], "best_sharpe": None, "skipped": True}

    # 生成子集
    if method == "exhaustive" and n <= 5:
        subsets = []
        for k in range(n + 1):  # 0 到 n
            for combo in itertools.combinations(range(n), k):
                subsets.append(list(combo))
    elif method == "leave_one_out":
        subsets = [list(range(n))]  # 全部保留
        for i in range(n):
            subsets.append([j for j in range(n) if j != i])
        subsets.append([])  # 仅 prefilter
    else:
        # n > 5 且 exhaustive → 自动降级为 leave_one_out
        print(f"  ⚠️  {n} rules > 5, 降级为 leave_one_out")
        subsets = [list(range(n))]
        for i in range(n):
            subsets.append([j for j in range(n) if j != i])
        subsets.append([])

    print(f"  📋 {len(subsets)} 种组合待测试\n")

    results = []
    for idx, subset_indices in enumerate(subsets):
        label = _subset_label(optimized, subset_indices)
        subset_rules = [optimized[i] for i in subset_indices]
        n_gates = len(prefilter) + len(subset_rules)

        with tempfile.TemporaryDirectory(prefix=f"gate_sel_{strategy}_") as tmpdir:
            tmp = Path(tmpdir)
            # 复制整个策略目录
            tmp_strategies = tmp / "strategies"
            shutil.copytree(
                Path(strategies_root) / strategy,
                tmp_strategies / strategy,
            )
            # 覆写 gate.yaml
            tmp_gate = tmp_strategies / strategy / "archetypes" / "gate.yaml"
            _write_gate_yaml(config, prefilter, subset_rules, tmp_gate)

            # 运行 backtest
            bt_json = tmp / "bt.json"
            metrics = _run_backtest(
                strategy,
                str(tmp_strategies),
                start_date,
                end_date,
                data_path,
                bt_json,
            )

        if "error" in metrics:
            print(f"  [{idx+1}/{len(subsets)}] {label:40s} ❌ {metrics['error'][:60]}")
            results.append(
                {
                    "indices": subset_indices,
                    "label": label,
                    "gates": n_gates,
                    "error": metrics["error"],
                }
            )
            continue

        sharpe = metrics.get("sharpe_r", 0)
        trades = metrics.get("n_trades", 0)
        win = metrics.get("win_rate", 0)
        mean_r = metrics.get("mean_r", 0)
        total_r = metrics.get("total_r", 0)

        results.append(
            {
                "indices": subset_indices,
                "label": label,
                "gates": n_gates,
                "sharpe": sharpe,
                "trades": trades,
                "win_rate": win,
                "mean_r": mean_r,
                "total_r": total_r,
            }
        )
        print(
            f"  [{idx+1}/{len(subsets)}] {label:40s} "
            f"Sharpe={sharpe:+.4f}  Trades={trades:>4}  "
            f"Win={win:.1%}  MeanR={mean_r:+.4f}  Gates={n_gates}"
        )

    # 选最优
    valid = [r for r in results if "error" not in r and r["trades"] >= min_trades]
    if not valid:
        print(f"\n  ❌ 没有组合满足 min_trades={min_trades}")
        return {"best_indices": None, "results": results}

    best = max(valid, key=lambda r: r["sharpe"])

    # 打印汇总表
    print(f"\n{'='*80}")
    print(f"  📊 Subset Selection Results: {strategy.upper()}")
    print(f"{'='*80}")
    print(
        f"  {'Subset':40s} {'Sharpe':>8} {'Trades':>7} {'Win%':>6} {'MeanR':>8} {'Gates':>5}"
    )
    print(f"  {'-'*75}")
    for r in results:
        if "error" in r:
            print(f"  {r['label']:40s} {'ERROR':>8}")
            continue
        marker = " 🏆" if r is best else ""
        print(
            f"  {r['label']:40s} {r['sharpe']:>+8.4f} {r['trades']:>7} "
            f"{r['win_rate']:>5.1%} {r['mean_r']:>+8.4f} {r['gates']:>5}{marker}"
        )

    print(
        f"\n  🏆 Best: {best['label']} → Sharpe={best['sharpe']:+.4f}, Trades={best['trades']}"
    )

    # promote
    if promote:
        subset_rules = [optimized[i] for i in best["indices"]]
        _write_gate_yaml(config, prefilter, subset_rules, gate_path)
        print(
            f"  📦 Promoted → {gate_path} ({len(prefilter)} prefilter + {len(subset_rules)} optimized)"
        )

    return {
        "best_indices": best["indices"],
        "best_label": best["label"],
        "best_sharpe": best["sharpe"],
        "best_trades": best["trades"],
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Gate Subset Selection — 在 optimized 规则上做组合优选",
    )
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--strategies-root", required=True)
    parser.add_argument(
        "--method",
        choices=["all", "leave_one_out", "exhaustive"],
        default="exhaustive",
    )
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--data-path", default="data/parquet_data")
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="将最优子集写入 archetypes/gate.yaml",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="保存选择结果 JSON",
    )
    args = parser.parse_args()

    result = select_best_subset(
        strategy=args.strategy,
        strategies_root=args.strategies_root,
        method=args.method,
        start_date=args.start_date,
        end_date=args.end_date,
        data_path=args.data_path,
        min_trades=args.min_trades,
        promote=args.promote,
    )

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  📄 Results saved → {args.output}")

    # 退出码: 0=成功, 1=无有效组合
    sys.exit(0 if result.get("best_indices") is not None else 1)


if __name__ == "__main__":
    main()
