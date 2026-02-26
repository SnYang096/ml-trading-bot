#!/usr/bin/env python3
"""
滚动 Deploy 门禁模拟 — 最近 N 个月逐月训练, 检验哪些月需要 deploy

用法:
    # 单策略, 最近 10 个月
    python scripts/test_rolling_deploy_gate.py --strategy fer

    # 全部策略 (自动发现 config 中的所有策略)
    python scripts/test_rolling_deploy_gate.py --all

    # 多策略指定
    python scripts/test_rolling_deploy_gate.py --strategy fer bpc

    # 自定义月数
    python scripts/test_rolling_deploy_gate.py --strategy fer --months 6

    # dry-run (只打印命令)
    python scripts/test_rolling_deploy_gate.py --all --dry-run

    # 跳过已有月份 (断点续跑)
    python scripts/test_rolling_deploy_gate.py --strategy fer --resume

流程:
    1. 生成 N 个月末 end_date: [T-N, T-N+1, ..., T-1]
    2. 对每个 strategy × end_date 调用完整 pipeline (训练+优化+回测)
    3. 逐月对比: comparison + drift + deploy gate
    4. 输出汇总表: 哪些月触发 deploy
"""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Setup path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.auto_research_pipeline import (
    check_deploy_gate,
    compare_runs,
    compute_holdout_start,
    load_pipeline_config,
    run_strategy_pipeline,
    _find_arch_dir,
    _print_drift_report,
    _load_report_metrics,
    save_report,
)

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "research_pipeline.yaml"
ROLLING_ROOT = PROJECT_ROOT / "results" / "rolling_deploy_test"


def generate_end_dates(months: int, ref_date: Optional[str] = None) -> List[str]:
    """生成最近 N 个月的 end_date 列表 (每月 1 号).

    例: months=10, ref=2026-02-23 → [2025-05-01, 2025-06-01, ..., 2026-02-01]
    """
    if ref_date:
        ref = datetime.strptime(ref_date, "%Y-%m-%d")
    else:
        ref = datetime.now()

    dates = []
    cur_y, cur_m = ref.year, ref.month
    for i in range(months - 1, -1, -1):
        m = cur_m - i
        y = cur_y
        while m <= 0:
            m += 12
            y -= 1
        dates.append(f"{y:04d}-{m:02d}-01")
    return dates


def load_month_report(history_dir: Path, end_date: str) -> Optional[Dict[str, Any]]:
    """加载某个月的 report.json."""
    report_path = history_dir / f"month_{end_date}" / "report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    return None


# ====================================================================
# 单策略滚动执行
# ====================================================================


def _run_strategy_rolling(
    strategy: str,
    cfg: dict,
    months: int,
    ref_date: Optional[str],
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> List[Dict[str, Any]]:
    """单策略滚动测试, 返回每月结果列表."""
    end_dates = generate_end_dates(months, ref_date)
    holdout_months = cfg["dates"]["holdout_months"]
    start_date = cfg["dates"]["start_date"]
    symbols = cfg["symbols"]
    data_path = cfg["data_path"]
    deploy_cfg = cfg.get("deploy_gate", {})
    # per-strategy kpi_gates.deploy 覆盖全局默认
    scfg = cfg.get("strategies", {}).get(strategy, {})
    deploy_kpi = scfg.get("kpi_gates", {}).get("deploy", {})
    if deploy_kpi.get("min_trades") is not None:
        deploy_cfg = {**deploy_cfg, "min_trades": deploy_kpi["min_trades"]}
    comparison_cfg = cfg.get("comparison", {})

    test_dir = ROLLING_ROOT / strategy
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 74}")
    print(f"  滚动测试: {strategy.upper()}")
    print(f"  月份:    {end_dates[0]} → {end_dates[-1]} ({len(end_dates)} 个月)")
    print(f"  输出:    {test_dir}")
    print(f"{'=' * 74}\n")

    monthly_results: List[Dict[str, Any]] = []

    for idx, end_date in enumerate(end_dates):
        month_label = end_date[:7]
        holdout_start = compute_holdout_start(end_date, holdout_months)
        run_dir = test_dir / f"month_{end_date}"

        print(f"\n{'═' * 74}")
        print(f"📅 [{idx + 1}/{len(end_dates)}] {month_label}")
        print(f"   end_date={end_date}  holdout_start={holdout_start}")
        print(f"   output: {run_dir}")
        print(f"{'═' * 74}")

        # 断点续跑
        if resume and (run_dir / "report.json").exists():
            print(f"   ⏭️  已有结果, 跳过 (--resume)")
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            monthly_results.append(
                {
                    "end_date": end_date,
                    "month": month_label,
                    "report": report,
                    "skipped": True,
                }
            )
            continue

        t0 = time.time()

        pipeline_result = run_strategy_pipeline(
            strategy,
            cfg,
            end_date=end_date,
            holdout_start=holdout_start,
            start_date=start_date,
            symbols=symbols,
            data_path=data_path,
            run_dir=run_dir,
            dry_run=dry_run,
        )

        elapsed = time.time() - t0

        if "error" in pipeline_result:
            print(f"\n   ❌ Pipeline 失败: {pipeline_result['error']}")
            monthly_results.append(
                {
                    "end_date": end_date,
                    "month": month_label,
                    "error": pipeline_result["error"],
                    "elapsed_s": elapsed,
                }
            )
            continue

        # 找上一个月的 report
        prev_report = None
        if idx > 0:
            for prev in reversed(monthly_results):
                if "report" in prev:
                    prev_report = prev["report"]
                    break

        # 对比决策
        comparison = compare_runs(
            {"backtest_metrics": pipeline_result["backtest_metrics"]},
            prev_report,
            comparison_cfg,
        )

        # 保存 report
        report_path = save_report(
            strategy,
            cfg,
            run_dir,
            pipeline_result,
            comparison,
            start_date=start_date,
            end_date=end_date,
            holdout_start=holdout_start,
        )

        report = json.loads(report_path.read_text(encoding="utf-8"))

        # 漂移分析
        drift_levels = None
        if prev_report and not dry_run and idx > 0:
            prev_end = monthly_results[-1].get("end_date", "")
            prev_dir = test_dir / f"month_{prev_end}"
            prev_arch = _find_arch_dir(prev_dir, strategy)
            cur_arch = _find_arch_dir(run_dir, strategy)
            if prev_arch and cur_arch:
                prev_metrics = prev_report.get("backtest_metrics", {})
                cur_metrics = pipeline_result["backtest_metrics"]
                drift_levels = _print_drift_report(
                    strategy,
                    f"month_{prev_end}",
                    f"month_{end_date}",
                    prev_arch,
                    cur_arch,
                    prev_metrics,
                    cur_metrics,
                )

        # Deploy 门禁检查
        decision = comparison["decision"]
        deploy_result = check_deploy_gate(
            decision, comparison, drift_levels, deploy_cfg
        )

        bt = pipeline_result["backtest_metrics"]
        monthly_results.append(
            {
                "end_date": end_date,
                "month": month_label,
                "report": report,
                "sharpe": bt.get("sharpe_per_trade", 0),
                "trades": bt.get("total_trades", 0),
                "decision": decision,
                "deploy_result": deploy_result,
                "drift_levels": drift_levels,
                "elapsed_s": elapsed,
            }
        )

        # 打印本月结果
        deploy_ready = deploy_result["deploy_ready"]
        triggered = deploy_result.get("triggered", False)
        if deploy_ready:
            deploy_str = "🚀 DEPLOY"
        elif not triggered:
            deploy_str = "⏭️  SKIP"
        else:
            deploy_str = "🚫 BLOCKED"
        print(
            f"\n   📊 Sharpe={bt.get('sharpe_per_trade', 0):.4f}  "
            f"Trades={bt.get('total_trades', 0)}  "
            f"Decision={decision}  Deploy={deploy_str}  "
            f"({elapsed:.0f}s)"
        )

    # ── 单策略汇总 ──
    _print_strategy_summary(strategy, end_dates, monthly_results)
    _save_summary_json(strategy, test_dir, end_dates, deploy_cfg, monthly_results)

    return monthly_results


# ====================================================================
# 输出 & 保存
# ====================================================================


def _print_strategy_summary(
    strategy: str,
    end_dates: List[str],
    monthly_results: List[Dict[str, Any]],
):
    """打印单策略汇总表."""
    w = 96
    print(f"\n\n╔{'═' * w}╗")
    title = (
        f"{strategy.upper()} 滚动 Deploy 门禁汇总 ({end_dates[0]} → {end_dates[-1]})"
    )
    print(f"║  {title}{' ' * max(0, w - len(title) - 2)}║")
    print(f"╚{'═' * w}╝\n")

    print(
        f"  {'月份':<10s} {'Sharpe':>8s} {'Trades':>7s} {'Decision':>8s} "
        f"{'Sharpe变化':>10s} {'最大漂移':>8s} {'Deploy':>10s} {'原因'}"
    )
    print(f"  {'─' * 95}")

    deploy_months = []
    for i, r in enumerate(monthly_results):
        month = r.get("month", "?")

        if "error" in r:
            print(f"  {month:<10s} {'ERROR':>8s}")
            continue

        sharpe = r.get("sharpe", 0)
        trades = r.get("trades", 0)
        decision = r.get("decision", "?")

        # Sharpe 变化
        if i > 0 and "sharpe" in monthly_results[i - 1]:
            prev_sharpe = monthly_results[i - 1]["sharpe"]
            if prev_sharpe and prev_sharpe != 0:
                sharpe_chg = (sharpe - prev_sharpe) / abs(prev_sharpe)
                sharpe_chg_str = f"{sharpe_chg:+.1%}"
            else:
                sharpe_chg_str = "N/A"
        else:
            sharpe_chg_str = "—"

        # 漂移
        drift = r.get("drift_levels")
        if drift:
            DRIFT_ORDER = {
                "NONE": 0,
                "LOW": 1,
                "STABLE": 1,
                "MONITOR": 2,
                "MEDIUM": 2,
                "REVIEW": 3,
                "HIGH": 3,
                "ADJUST": 4,
            }
            max_drift = max(
                drift.values(), key=lambda x: DRIFT_ORDER.get(x, 0), default="—"
            )
        else:
            max_drift = "—"

        # Deploy
        dr = r.get("deploy_result", {})
        if dr.get("deploy_ready"):
            deploy_str = "🚀 DEPLOY"
            reason = "触发+安全通过"
            deploy_months.append(month)
        elif not dr.get("triggered"):
            deploy_str = "⏭️  SKIP"
            reason = dr.get("skip_reason", "无触发")[:40]
        elif dr.get("blocked_by"):
            deploy_str = "🚫 BLOCKED"
            reason = "; ".join(dr["blocked_by"])[:40]
        elif i == 0:
            deploy_str = "🆕 首版"
            reason = "首次运行"
        else:
            deploy_str = "?"
            reason = ""

        print(
            f"  {month:<10s} {sharpe:>8.4f} {trades:>7d} {decision:>8s} "
            f"{sharpe_chg_str:>10s} {max_drift:>8s} {deploy_str:<10s} {reason}"
        )

    print(f"  {'─' * 95}")

    total = len([r for r in monthly_results if "error" not in r])
    n_deploy = len(deploy_months)
    n_skip = len(
        [
            r
            for r in monthly_results
            if r.get("deploy_result", {}).get("triggered") is False
        ]
    )
    n_blocked = len(
        [
            r
            for r in monthly_results
            if r.get("deploy_result", {}).get("triggered")
            and not r.get("deploy_result", {}).get("deploy_ready")
        ]
    )

    print(f"\n  📈 总计 {total} 个月:")
    print(f"     🚀 DEPLOY:  {n_deploy} 个月 ({n_deploy / max(total, 1):.0%})")
    print(f"     ⏭️  SKIP:    {n_skip} 个月")
    print(f"     🚫 BLOCKED: {n_blocked} 个月")
    if deploy_months:
        print(f"     触发月份: {', '.join(deploy_months)}")
    print()


def _save_summary_json(
    strategy: str,
    test_dir: Path,
    end_dates: List[str],
    deploy_cfg: dict,
    monthly_results: List[Dict[str, Any]],
):
    """保存汇总 JSON."""
    summary_path = test_dir / "rolling_summary.json"
    summary = {
        "strategy": strategy,
        "end_dates": end_dates,
        "deploy_cfg": deploy_cfg,
        "months": [],
    }
    for r in monthly_results:
        entry = {
            "end_date": r.get("end_date"),
            "sharpe": r.get("sharpe"),
            "trades": r.get("trades"),
            "decision": r.get("decision"),
            "deploy_ready": r.get("deploy_result", {}).get("deploy_ready"),
            "triggered": r.get("deploy_result", {}).get("triggered"),
            "blocked_by": r.get("deploy_result", {}).get("blocked_by", []),
            "skip_reason": r.get("deploy_result", {}).get("skip_reason"),
            "drift_levels": r.get("drift_levels"),
            "elapsed_s": r.get("elapsed_s"),
        }
        if "error" in r:
            entry["error"] = r["error"]
        summary["months"].append(entry)

    summary_path.write_text(
        json.dumps(summary, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  💾 汇总已保存: {summary_path}")


def _print_cross_strategy_summary(
    strategies: List[str],
    all_summaries: Dict[str, List[Dict[str, Any]]],
):
    """多策略交叉汇总表."""
    print(f"\n\n{'=' * 80}")
    print(f"🎯 全策略滚动 Deploy 汇总")
    print(f"{'=' * 80}")
    print(
        f"  {'策略':<8s} {'月数':>4s} {'DEPLOY':>7s} {'SKIP':>6s} "
        f"{'BLOCKED':>8s} {'平均Sharpe':>10s} {'Deploy率':>8s}"
    )
    print(f"  {'─' * 60}")
    for s in strategies:
        results = all_summaries.get(s, [])
        valid = [r for r in results if "error" not in r]
        n_deploy = len(
            [r for r in valid if r.get("deploy_result", {}).get("deploy_ready")]
        )
        n_skip = len(
            [r for r in valid if r.get("deploy_result", {}).get("triggered") is False]
        )
        n_blocked = len(
            [
                r
                for r in valid
                if r.get("deploy_result", {}).get("triggered")
                and not r.get("deploy_result", {}).get("deploy_ready")
            ]
        )
        sharpes = [r["sharpe"] for r in valid if "sharpe" in r and r["sharpe"]]
        avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0
        rate = n_deploy / max(len(valid), 1)
        print(
            f"  {s:<8s} {len(valid):>4d} {n_deploy:>7d} {n_skip:>6d} "
            f"{n_blocked:>8d} {avg_sharpe:>10.4f} {rate:>8.0%}"
        )
    print(f"  {'─' * 60}")
    print()


# ====================================================================
# Main
# ====================================================================


def main():
    p = argparse.ArgumentParser(description="滚动 Deploy 门禁模拟")
    p.add_argument("--strategy", nargs="+", help="策略名 (如 fer bpc me, 支持多个)")
    p.add_argument("--all", action="store_true", help="执行 config 中定义的所有策略")
    p.add_argument("--months", type=int, default=10, help="模拟月数 (默认 10)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="pipeline 配置")
    p.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    p.add_argument("--resume", action="store_true", help="跳过已有结果的月份")
    p.add_argument("--ref-date", help="参考日期 (默认今天, 格式 YYYY-MM-DD)")
    args = p.parse_args()

    cfg = load_pipeline_config(Path(args.config))
    available_strategies = list(cfg.get("strategies", {}).keys())

    # 确定策略列表
    if args.all:
        strategies = available_strategies
    elif args.strategy:
        strategies = args.strategy
    else:
        p.error("必须指定 --strategy 或 --all")
        return

    # 校验
    for s in strategies:
        if s not in available_strategies:
            print(
                f"❌ 策略 '{s}' 不在 config 中, 可用: {', '.join(available_strategies)}"
            )
            sys.exit(1)

    print(f"\n{'#' * 78}")
    print(f"# 滚动 Deploy 门禁模拟")
    print(f"# 策略: {', '.join(s.upper() for s in strategies)}")
    print(f"# 月数: {args.months}")
    print(f"{'#' * 78}")

    # 每个策略单独跑
    all_summaries: Dict[str, List[Dict[str, Any]]] = {}
    for strategy in strategies:
        monthly_results = _run_strategy_rolling(
            strategy,
            cfg,
            args.months,
            args.ref_date,
            dry_run=args.dry_run,
            resume=args.resume,
        )
        all_summaries[strategy] = monthly_results

    # 多策略交叉汇总
    if len(strategies) > 1:
        _print_cross_strategy_summary(strategies, all_summaries)


if __name__ == "__main__":
    main()
