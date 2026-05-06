#!/usr/bin/env python3
"""
Grid search min_combined_pass_rate — 复用已有 gate train 产出,
只重跑 gate optimize + event backtest, 对比不同 pass rate 下的 Sharpe/Trades.

用法:
  python scripts/grid_search_pass_rate.py --strategy me
  python scripts/grid_search_pass_rate.py --strategy bpc --rates 0.15,0.20,0.30,0.40
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def find_latest_run(strategy: str) -> Path:
    """找到最新一次 research_history 运行目录."""
    hist = Path("results/research_history") / strategy
    runs = sorted(hist.iterdir()) if hist.exists() else []
    if not runs:
        raise FileNotFoundError(f"No runs found for {strategy}")
    return runs[-1]


def find_gate_artifacts(run_dir: Path, strategy: str, logs_override: str = None):
    """从 pipeline.log 中提取 gate_dir 和 gate_draft 路径."""
    # 找 gate_draft.yaml
    gate_draft = run_dir / "strategies" / strategy / "gate_draft.yaml"

    # 如果用户指定了 logs 路径, 直接使用
    if logs_override:
        logs_gated = Path(logs_override)
        if not logs_gated.exists():
            raise FileNotFoundError(f"Specified logs not found: {logs_gated}")
        if not gate_draft.exists():
            raise FileNotFoundError(f"gate_draft.yaml not found: {gate_draft}")
        return logs_gated, gate_draft

    # 从 pipeline.log 提取 — 只匹配 rr_extreme (gate train 产出)
    log_path = run_dir / "pipeline.log"
    logs_gated = None
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8")
        for line in log_text.splitlines():
            # 匹配 gate apply 的 --out 参数 (rr_extreme 目录)
            if (
                "logs_gated.parquet" in line
                and "--out" in line
                and "rr_extreme" in line
            ):
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "--out" and i + 1 < len(parts):
                        candidate = Path(parts[i + 1])
                        if candidate.exists():
                            logs_gated = candidate
                        break
                if logs_gated:
                    break

    if logs_gated is None or not logs_gated.exists():
        # fallback: 扫描 results/train_final_*_rr_extreme 目录 (取最新)
        _r = Path("results")
        candidates = sorted(
            list(_r.glob(f"train_final_*_rr_extreme/{strategy}/logs_gated.parquet"))
            + list(
                _r.glob(
                    f"{strategy}/train_final_*_rr_extreme/{strategy}/logs_gated.parquet"
                )
            )
        )
        if candidates:
            logs_gated = candidates[-1]

    if logs_gated is None or not logs_gated.exists():
        raise FileNotFoundError(
            f"logs_gated.parquet (rr_extreme) not found for {strategy}. "
            f"Run the full pipeline first, or use --logs to specify."
        )
    if not gate_draft.exists():
        raise FileNotFoundError(f"gate_draft.yaml not found: {gate_draft}")

    return logs_gated, gate_draft


def run_one(
    strategy: str,
    rate: float,
    logs_gated: Path,
    gate_draft: Path,
    base_strategies: Path,
    holdout_start: str,
    holdout_end: str,
) -> dict:
    """对单个 min_combined_pass_rate 值, 运行 gate optimize + event backtest."""
    with tempfile.TemporaryDirectory(prefix=f"grid_{strategy}_{rate}_") as tmpdir:
        tmp = Path(tmpdir)
        # 复制 strategies 目录
        tmp_strategies = tmp / "strategies"
        shutil.copytree(base_strategies / strategy, tmp_strategies / strategy)

        # 1. Gate Optimize
        opt_json = tmp / "gate_opt.json"
        gate_cmd = [
            sys.executable,
            "scripts/optimize_gate_unified.py",
            "--strategy",
            strategy,
            "--strategies-root",
            str(tmp_strategies),
            "--logs",
            str(logs_gated),
            "--output",
            str(opt_json),
            "--gate-path",
            str(gate_draft),
            "--promote",
            "--min-combined-pass-rate",
            str(rate),
        ]
        r1 = subprocess.run(gate_cmd, capture_output=True, text=True, timeout=300)
        if r1.returncode != 0:
            return {"rate": rate, "error": f"gate_optimize failed: {r1.stderr[-500:]}"}

        # 提取 gate 信息
        gate_info = ""
        for line in r1.stdout.splitlines():
            if "累积 AND pass rate" in line or "裁剪" in line or "无需裁剪" in line:
                gate_info += line.strip() + " | "

        # 2. Event Backtest
        bt_json = tmp / "backtest.json"
        bt_csv = tmp / "trades.csv"
        bt_cmd = [
            sys.executable,
            "scripts/event_backtest.py",
            "--strategy",
            strategy,
            "--start-date",
            holdout_start,
            "--end-date",
            holdout_end,
            "--data-path",
            "data/parquet_data",
            "--strategies-root",
            str(tmp_strategies),
            "--output",
            str(bt_json),
            "--export",
            str(bt_csv),
        ]
        r2 = subprocess.run(bt_cmd, capture_output=True, text=True, timeout=600)
        if r2.returncode != 0:
            return {"rate": rate, "error": f"backtest failed: {r2.stderr[-500:]}"}

        # 解析结果
        try:
            metrics = json.loads(bt_json.read_text())
        except Exception as e:
            return {"rate": rate, "error": f"parse backtest json: {e}"}

        # 读 gate.yaml 规则数
        gate_yaml = tmp_strategies / strategy / "archetypes" / "gate.yaml"
        n_rules = 0
        if gate_yaml.exists():
            import yaml

            gc = yaml.safe_load(gate_yaml.read_text())
            n_rules = len(gc.get("hard_gates", []))

        return {
            "rate": rate,
            "sharpe": metrics.get("sharpe_r", "?"),
            "trades": metrics.get("n_trades", "?"),
            "win_rate": metrics.get("win_rate", "?"),
            "mean_r": metrics.get("mean_r", "?"),
            "total_r": metrics.get("total_r", "?"),
            "max_dd": metrics.get("max_drawdown_r", "?"),
            "gate_rules": n_rules,
            "gate_info": gate_info.strip(" |"),
        }


def main():
    parser = argparse.ArgumentParser(description="Grid search min_combined_pass_rate")
    parser.add_argument("--strategy", required=True)
    parser.add_argument(
        "--rates",
        default="0.15,0.20,0.25,0.30,0.35,0.40,0.50",
        help="逗号分隔的 pass rate 值",
    )
    parser.add_argument("--holdout-start", default=None)
    parser.add_argument("--holdout-end", default=None)
    parser.add_argument(
        "--strategies-root",
        default=None,
        help="策略配置根目录, 默认使用最新 research_history 的 strategies",
    )
    parser.add_argument(
        "--logs",
        default=None,
        help="显式指定 logs_gated.parquet 路径 (必须是 rr_extreme gate train 产出)",
    )
    args = parser.parse_args()

    strategy = args.strategy
    rates = [float(r) for r in args.rates.split(",")]

    # 找最新 run
    run_dir = find_latest_run(strategy)
    print(f"📁 Latest run: {run_dir.name}")

    # 找 gate artifacts
    logs_gated, gate_draft = find_gate_artifacts(run_dir, strategy, args.logs)
    print(f"📊 logs_gated: {logs_gated}")
    print(f"📄 gate_draft: {gate_draft}")

    # 确定 strategies root
    if args.strategies_root:
        base_strategies = Path(args.strategies_root)
    else:
        base_strategies = run_dir / "strategies"
    print(f"📂 strategies: {base_strategies}")

    # 确定 holdout 日期
    holdout_start = args.holdout_start
    holdout_end = args.holdout_end
    if not holdout_start:
        report = run_dir / "report.json"
        if report.exists():
            rdata = json.loads(report.read_text())
            dr = rdata.get("data_range", {})
            holdout_start = dr.get("holdout_start", "2025-08-01")
            holdout_end = holdout_end or dr.get("end_date", "2026-02-01")
    holdout_start = holdout_start or "2025-08-01"
    holdout_end = holdout_end or "2026-02-01"
    print(f"📅 Holdout: {holdout_start} → {holdout_end}")

    print(f"\n{'='*90}")
    print(f"  Grid Search: min_combined_pass_rate = {rates}")
    print(f"{'='*90}\n")

    results = []
    for i, rate in enumerate(rates):
        print(f"\n[{i+1}/{len(rates)}] Testing rate={rate:.2f} ...")
        result = run_one(
            strategy,
            rate,
            logs_gated,
            gate_draft,
            base_strategies,
            holdout_start,
            holdout_end,
        )
        results.append(result)
        if "error" in result:
            print(f"  ❌ {result['error']}")
        else:
            print(
                f"  ✅ Sharpe={result['sharpe']:.4f}  Trades={result['trades']}  "
                f"Win={result['win_rate']:.1%}  MeanR={result['mean_r']:.4f}  "
                f"Gates={result['gate_rules']}"
            )

    # 汇总表
    print(f"\n{'='*90}")
    print(f"  📊 Grid Search Results: {strategy.upper()}")
    print(f"{'='*90}")
    print(
        f"  {'Rate':>6}  {'Sharpe':>8}  {'Trades':>7}  {'Win%':>6}  "
        f"{'MeanR':>8}  {'TotalR':>8}  {'MaxDD':>7}  {'Gates':>5}"
    )
    print(f"  {'-'*70}")

    best = None
    for r in results:
        if "error" in r:
            print(f"  {r['rate']:>6.2f}  {'ERROR':>8}  {r['error'][:50]}")
            continue
        marker = ""
        if best is None or (
            isinstance(r["sharpe"], (int, float))
            and isinstance(best["sharpe"], (int, float))
            and r["sharpe"] > best["sharpe"]
            and isinstance(r["trades"], int)
            and r["trades"] >= 30
        ):
            best = r
        print(
            f"  {r['rate']:>6.2f}  {r['sharpe']:>8.4f}  {r['trades']:>7}  "
            f"{r['win_rate']:>5.1%}  {r['mean_r']:>8.4f}  {r['total_r']:>8.2f}  "
            f"{r['max_dd']:>7.2f}  {r['gate_rules']:>5}"
        )

    if best:
        print(
            f"\n  🏆 Best: rate={best['rate']:.2f} → Sharpe={best['sharpe']:.4f}, Trades={best['trades']}"
        )
        print(
            f"     建议写入 research_pipeline.yaml: min_combined_pass_rate: {best['rate']:.2f}"
        )


if __name__ == "__main__":
    main()
