#!/usr/bin/env python3
"""
一键配置健康检查 — 自动发现数据 + 趋势分析 + 结论

统一替代 local_monitor_feature_drift / local_monitor_weekly /
local_monitor_monthly / monitor_retrain / test_rolling_deploy_gate 五个工具。

核心逻辑:
  1. 自动扫描 results/research_history/、results/train_final_* 与 results/<策略>/train_final_*
     发现最近 N 次训练的 report.json + training_baseline.json
  2. 如果数据不足 (< --min-months), 自动调用 auto_research_pipeline 补训
  3. 逐月对比: 特征漂移 + L1-L4 健康 + Sharpe/Trades 趋势
  4. 输出趋势表 + 明确结论: STABLE / ATTENTION / RETRAIN

工作流:
  check_need_retrain.py → 看到 RETRAIN
    → auto_research_pipeline.py → 修改配置 → commit → deploy

用法:
    # 单策略
    python scripts/check_need_retrain.py --strategy fer

    # 全部策略
    python scripts/check_need_retrain.py --all

    # 指定最少月数 (不足时自动补训)
    python scripts/check_need_retrain.py --strategy fer --min-months 6

    # 只看已有数据, 不自动补训
    python scripts/check_need_retrain.py --strategy fer --no-fill

    # dry-run (补训时只打印命令)
    python scripts/check_need_retrain.py --strategy fer --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "pipelines" / "pcm_orchestrate_2h.yaml"


# ====================================================================
# Config
# ====================================================================


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def get_available_strategies(cfg: dict) -> List[str]:
    return list(cfg.get("strategies", {}).keys())


# ====================================================================
# Auto-discover training data
# ====================================================================


def _normalize_date(d: str) -> str:
    """标准化日期格式: '2026-2-01' → '2026-02-01'."""
    try:
        return datetime.strptime(d.strip(), "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        try:
            return datetime.strptime(d.strip(), "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            return d


def _parse_end_date_from_report(report: dict) -> Optional[str]:
    """从 report.json 提取 end_date (标准化格式)."""
    dr = report.get("data_range", {})
    raw = dr.get("end_date")
    return _normalize_date(raw) if raw else None


def _parse_sharpe_from_report(report: dict) -> float:
    bm = report.get("backtest_metrics", {})
    return float(bm.get("sharpe_per_trade", 0))


def _parse_trades_from_report(report: dict) -> int:
    bm = report.get("backtest_metrics", {})
    return int(bm.get("total_trades", 0))


def discover_research_runs(
    strategy: str,
    cfg: dict,
) -> List[Dict[str, Any]]:
    """
    扫描三个来源, 返回按 end_date 去重排序的训练记录列表.

    来源 (按优先级):
      1. results/research_history/{strategy}/*/report.json
      2. results/rolling_deploy_test/{strategy}/month_*/report.json
      3. results/train_final_*/{strategy}/training_baseline.json
    """
    history_dir = PROJECT_ROOT / cfg["output"]["history_dir"]
    runs: Dict[str, Dict[str, Any]] = {}  # end_date → run_info

    # ── 来源 1: research_history ──
    strat_hist = history_dir / strategy
    if strat_hist.exists():
        for run_dir in sorted(strat_hist.iterdir()):
            if not run_dir.is_dir():
                continue
            report_path = run_dir / "report.json"
            if not report_path.exists():
                continue
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            end_date = _parse_end_date_from_report(report)
            sharpe = _parse_sharpe_from_report(report)
            trades = _parse_trades_from_report(report)
            if not end_date or trades == 0:
                continue
            # 按 end_date 去重, 保留最新 timestamp
            if end_date not in runs or run_dir.name > runs[end_date].get(
                "timestamp", ""
            ):
                runs[end_date] = {
                    "source": "research_history",
                    "timestamp": run_dir.name,
                    "end_date": end_date,
                    "sharpe": sharpe,
                    "trades": trades,
                    "report_path": str(report_path),
                    "dir": str(run_dir),
                }

    # ── 来源 2: rolling_deploy_test ──
    rolling_dir = PROJECT_ROOT / "results" / "rolling_deploy_test" / strategy
    if rolling_dir.exists():
        for month_dir in sorted(rolling_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            report_path = month_dir / "report.json"
            if not report_path.exists():
                continue
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            end_date = _parse_end_date_from_report(report)
            sharpe = _parse_sharpe_from_report(report)
            trades = _parse_trades_from_report(report)
            if not end_date or trades == 0:
                continue
            # 只在 research_history 中没有该月数据时使用
            if end_date not in runs:
                runs[end_date] = {
                    "source": "rolling_deploy_test",
                    "timestamp": month_dir.name,
                    "end_date": end_date,
                    "sharpe": sharpe,
                    "trades": trades,
                    "report_path": str(report_path),
                    "dir": str(month_dir),
                }

    # ── 来源 3: train_final_* (仅补充 baseline, 从 training_baseline.json 提取) ──
    # 新布局: results/<strategy>/train_final_* ; 旧布局: results/train_final_*
    _tf_roots: List[Path] = []
    for tf_dir in sorted(PROJECT_ROOT.glob("results/train_final_*")):
        if tf_dir.is_dir():
            _tf_roots.append(tf_dir)
    _strat_rf = PROJECT_ROOT / "results" / strategy
    if _strat_rf.is_dir():
        for tf_dir in sorted(_strat_rf.glob("train_final_*")):
            if tf_dir.is_dir():
                _tf_roots.append(tf_dir)

    for tf_dir in _tf_roots:
        strat_dir = tf_dir / strategy
        baseline_path = strat_dir / "training_baseline.json"
        if not baseline_path.exists():
            continue
        try:
            bl = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        hp = bl.get("holdout_period", {})
        end_date = hp.get("end")
        if not end_date:
            continue
        end_date = _normalize_date(end_date)
        # 从 layer_kpis 中提取 sharpe
        exec_kpi = bl.get("layer_kpis", {}).get("L7_execution", {})
        sharpe = float(exec_kpi.get("sharpe_per_trade", 0))
        trades = int(exec_kpi.get("total_trades", 0))
        if trades == 0 and sharpe == 0:
            # 尝试从 report.json fallback
            report_path = strat_dir / "report.json"
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    sharpe = _parse_sharpe_from_report(report)
                    trades = _parse_trades_from_report(report)
                except Exception:
                    pass
        if trades == 0:
            continue
        if end_date not in runs:
            runs[end_date] = {
                "source": "train_final",
                "timestamp": tf_dir.name,
                "end_date": end_date,
                "sharpe": sharpe,
                "trades": trades,
                "baseline_path": str(baseline_path),
                "dir": str(strat_dir),
            }

    # 按 end_date 排序
    return sorted(runs.values(), key=lambda x: x["end_date"])


# ====================================================================
# Find baseline pairs (for drift comparison)
# ====================================================================


def find_baseline_for_run(run: Dict[str, Any], strategy: str) -> Optional[Path]:
    """为某次 run 找到对应的 training_baseline.json."""
    # 1. 同目录下
    run_dir = Path(run["dir"])
    bl = run_dir / "training_baseline.json"
    if bl.exists():
        return bl

    # 2. 对应的 train_final 目录 (新/旧布局)
    _bl_roots: List[Path] = [
        p for p in PROJECT_ROOT.glob("results/train_final_*") if p.is_dir()
    ]
    _sroot = PROJECT_ROOT / "results" / strategy
    if _sroot.is_dir():
        _bl_roots.extend(p for p in _sroot.glob("train_final_*") if p.is_dir())
    for tf_dir in sorted(_bl_roots, key=lambda p: p.name, reverse=True):
        strat_bl = tf_dir / strategy / "training_baseline.json"
        if strat_bl.exists():
            # 检查 end_date 是否匹配
            try:
                d = json.loads(strat_bl.read_text(encoding="utf-8"))
                hp = d.get("holdout_period", {})
                if hp.get("end") == run["end_date"]:
                    return strat_bl
            except Exception:
                continue

    # 3. Fallback: 最新的 baseline (任何 end_date)
    for tf_dir in sorted(_bl_roots, key=lambda p: p.name, reverse=True):
        strat_bl = tf_dir / strategy / "training_baseline.json"
        if strat_bl.exists():
            return strat_bl

    return None


def find_predictions_for_run(run: Dict[str, Any], strategy: str) -> Optional[Path]:
    """为某次 run 找到 predictions.parquet."""
    run_dir = Path(run["dir"])
    # 同目录下
    pred = run_dir / "predictions.parquet"
    if pred.exists():
        return pred
    # 对应的 train_final 目录
    _pred_roots: List[Path] = [
        p for p in PROJECT_ROOT.glob("results/train_final_*") if p.is_dir()
    ]
    _sroot2 = PROJECT_ROOT / "results" / strategy
    if _sroot2.is_dir():
        _pred_roots.extend(p for p in _sroot2.glob("train_final_*") if p.is_dir())
    for tf_dir in sorted(_pred_roots, key=lambda p: p.name, reverse=True):
        p = tf_dir / strategy / "predictions.parquet"
        if p.exists():
            try:
                # 检查是否同 end_date
                bl_path = tf_dir / strategy / "training_baseline.json"
                if bl_path.exists():
                    d = json.loads(bl_path.read_text(encoding="utf-8"))
                    if (
                        _normalize_date(d.get("holdout_period", {}).get("end", ""))
                        == run["end_date"]
                    ):
                        return p
            except Exception:
                pass
    return None


# ====================================================================
# Auto-fill missing months
# ====================================================================


def generate_monthly_end_dates(n_months: int) -> List[str]:
    """生成最近 N 个月的 end_date (每月 1 号)."""
    now = datetime.now()
    dates = []
    for i in range(n_months, 0, -1):
        y = now.year
        m = now.month - i
        while m <= 0:
            m += 12
            y -= 1
        dates.append(f"{y:04d}-{m:02d}-01")
    return dates


def find_missing_months(
    runs: List[Dict[str, Any]],
    min_months: int,
) -> List[str]:
    """找出需要补训的月份."""
    needed = generate_monthly_end_dates(min_months)
    existing = {r["end_date"] for r in runs}
    return [d for d in needed if d not in existing]


def fill_missing_data(
    strategy: str,
    missing_months: List[str],
    config_path: Path,
    *,
    dry_run: bool = False,
) -> int:
    """对缺失月份调用 auto_research_pipeline 补训."""
    if not missing_months:
        return 0

    print(f"\n{'─'*60}")
    print(f"  自动补训: {strategy.upper()} × {len(missing_months)} 个月")
    print(f"  缺失: {', '.join(missing_months)}")
    print(f"{'─'*60}")

    filled = 0
    for end_date in missing_months:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "auto_research_pipeline.py"),
            "--strategy",
            strategy,
            "--end-date",
            end_date,
            "--no-adopt",
            "--config",
            str(config_path),
        ]
        if dry_run:
            print(f"  [DRY-RUN] {' '.join(cmd)}")
            filled += 1
        else:
            print(f"\n  ▶ 补训 end_date={end_date}")
            print(f"    {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if result.returncode == 0:
                filled += 1
                print(f"  ✅ {end_date} 完成")
            else:
                print(f"  ❌ {end_date} 失败 (exit {result.returncode})")

    return filled


# ====================================================================
# Health checks (lightweight, reuse existing monitor logic)
# ====================================================================


def check_feature_drift_lightweight(
    baseline_path: Path,
    data_path: Path,
) -> Dict[str, Any]:
    """轻量级特征漂移检查 (直接读 baseline 统计量)."""
    try:
        bl = json.loads(baseline_path.read_text(encoding="utf-8"))
        distributions = bl.get("feature_distributions", {})
        if not distributions:
            return {"status": "SKIP", "reason": "no distributions", "drift_rate": 0}

        import pandas as pd

        df = pd.read_parquet(data_path)

        n_checked = 0
        n_drifted = 0
        for feat, stats in distributions.items():
            if feat not in df.columns:
                continue
            n_checked += 1
            old_mean = stats.get("mean", 0)
            old_std = stats.get("std", 1)
            col = df[feat].dropna()
            if len(col) < 10:
                continue
            new_mean = float(col.mean())
            mean_shift = (
                abs(new_mean - old_mean) / max(old_std, 1e-8) if old_std > 1e-8 else 0
            )
            if mean_shift > 2.0:
                n_drifted += 1

        drift_rate = n_drifted / max(n_checked, 1)
        if drift_rate > 0.30:
            status = "HIGH"
        elif drift_rate > 0.15:
            status = "MEDIUM"
        elif drift_rate > 0.05:
            status = "LOW"
        else:
            status = "NONE"
        return {
            "status": status,
            "drift_rate": drift_rate,
            "checked": n_checked,
            "drifted": n_drifted,
        }
    except Exception as e:
        return {"status": "ERROR", "reason": str(e), "drift_rate": 0}


def check_data_age_days(end_date: str) -> int:
    """训练数据截止日期距今天数."""
    try:
        dt = datetime.strptime(end_date, "%Y-%m-%d")
        return (datetime.now() - dt).days
    except ValueError:
        return 9999


# ====================================================================
# Trend analysis
# ====================================================================


DRIFT_LEVEL_ORDER = {
    "NONE": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "ERROR": -1,
    "SKIP": -1,
}


def compute_trend(values: List[float]) -> str:
    """判断趋势: 上升/稳定/下降."""
    if len(values) < 2:
        return "—"
    # 简单线性回归斜率
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return "—"
    x, y = x[valid], y[valid]
    slope = np.polyfit(x, y, 1)[0]
    rel_slope = slope / max(abs(y.mean()), 1e-8)
    if rel_slope > 0.03:
        return "↗ 上升"
    elif rel_slope < -0.03:
        return "↘ 下降"
    return "→ 稳定"


def make_verdict(
    runs: List[Dict[str, Any]],
    drift_results: List[Dict[str, Any]],
    retrain_cfg: dict,
) -> Dict[str, Any]:
    """综合判定: STABLE / ATTENTION / RETRAIN."""
    reasons: List[str] = []

    if not runs:
        return {"verdict": "RETRAIN", "reasons": ["无训练数据"]}

    latest = runs[-1]

    # 1. 数据年龄
    max_age = retrain_cfg.get("max_data_age_days", 120)
    data_age = check_data_age_days(latest["end_date"])
    if data_age >= max_age:
        reasons.append(f"数据超龄: {data_age} 天 >= {max_age} 天")

    # 2. Sharpe 趋势
    sharpes = [r["sharpe"] for r in runs if r["sharpe"] > 0]
    if len(sharpes) >= 3:
        trend = compute_trend(sharpes)
        if "下降" in trend:
            # 检查下降幅度
            recent_avg = np.mean(sharpes[-2:])
            early_avg = np.mean(sharpes[:2])
            if early_avg > 0 and recent_avg / early_avg < 0.7:
                reasons.append(
                    f"Sharpe 持续下降: 近期={recent_avg:.4f} vs 早期={early_avg:.4f} ({recent_avg/early_avg:.0%})"
                )

    # 3. 特征漂移趋势
    if drift_results:
        latest_drift = drift_results[-1]
        if latest_drift.get("status") == "HIGH":
            reasons.append(f"特征漂移严重: {latest_drift.get('drift_rate', 0):.1%}")
        high_count = sum(1 for d in drift_results if d.get("status") == "HIGH")
        if high_count >= len(drift_results) * 0.5 and len(drift_results) >= 3:
            reasons.append(f"特征漂移持续 HIGH: {high_count}/{len(drift_results)} 个月")

    # 4. 定期检查
    schedule_days = retrain_cfg.get("schedule_days", 90)
    if data_age >= schedule_days:
        reasons.append(f"定期重训: 距上次 {data_age} 天 >= {schedule_days} 天")

    if len(reasons) >= 2:
        verdict = "RETRAIN"
    elif len(reasons) >= 1:
        verdict = "ATTENTION"
    else:
        verdict = "STABLE"

    return {"verdict": verdict, "reasons": reasons}


# ====================================================================
# Pretty print
# ====================================================================


def _drift_emoji(level: str) -> str:
    m = {"NONE": "🟢", "LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴"}
    return m.get(level, "⚪")


def _verdict_emoji(verdict: str) -> str:
    m = {"STABLE": "🟢", "ATTENTION": "🟡", "RETRAIN": "🔴"}
    return m.get(verdict, "⚪")


def print_strategy_report(
    strategy: str,
    runs: List[Dict[str, Any]],
    drift_results: List[Dict[str, Any]],
    verdict: Dict[str, Any],
):
    """打印单策略报告."""
    w = 72
    print(f"\n╔{'═'*w}╗")
    title = f"  {strategy.upper()} 配置健康检查 ({len(runs)} 次训练)"
    print(f"║{title}{' ' * max(0, w - len(title))}║")
    print(f"╚{'═'*w}╝")

    if not runs:
        print("  ❌ 无可用训练数据")
        return

    # ── 趋势表 ──
    print(
        f"\n  {'月份':<12s} {'Sharpe':>8s} {'Trades':>7s} {'漂移':>6s} {'数据龄':>7s}  {'来源'}"
    )
    print(f"  {'─'*65}")

    for i, run in enumerate(runs):
        end_date = run["end_date"]
        sharpe = run["sharpe"]
        trades = run["trades"]
        data_age = check_data_age_days(end_date)

        # Sharpe 变化标注
        sharpe_str = f"{sharpe:.4f}"
        if i > 0 and runs[i - 1]["sharpe"] > 0:
            chg = (sharpe - runs[i - 1]["sharpe"]) / abs(runs[i - 1]["sharpe"])
            if abs(chg) >= 0.05:
                sharpe_str += f" ({chg:+.0%})"

        # 漂移
        drift_info = drift_results[i] if i < len(drift_results) else {}
        drift_level = drift_info.get("status", "—")
        drift_str = (
            f"{_drift_emoji(drift_level)} {drift_level}"
            if drift_level in DRIFT_LEVEL_ORDER
            else f"  {drift_level}"
        )

        # 来源缩写
        src_map = {
            "research_history": "hist",
            "rolling_deploy_test": "roll",
            "train_final": "train",
        }
        src = src_map.get(run.get("source", ""), "?")

        # 数据年龄预警
        age_str = f"{data_age:>4d}天"
        if data_age >= 120:
            age_str += " ⚠"

        print(
            f"  {end_date:<12s} {sharpe_str:>14s} {trades:>7d} {drift_str:>10s} {age_str:>9s}  {src}"
        )

    # ── 趋势 ──
    sharpes = [r["sharpe"] for r in runs if r["sharpe"] > 0]
    if len(sharpes) >= 2:
        trend = compute_trend(sharpes)
        print(
            f"\n  Sharpe 趋势: {trend} (range: {min(sharpes):.4f} ~ {max(sharpes):.4f})"
        )

    # ── 结论 ──
    v = verdict["verdict"]
    emoji = _verdict_emoji(v)
    print(f"\n  {'━'*65}")
    print(f"  🎯 结论: {emoji} {v}")
    if verdict["reasons"]:
        for r in verdict["reasons"]:
            print(f"     → {r}")
    else:
        print(f"     配置健康, 无需改动")

    if v == "RETRAIN":
        print(f"\n  📋 下一步:")
        print(f"     python scripts/auto_research_pipeline.py --strategy {strategy}")
    elif v == "ATTENTION":
        print(f"\n  📋 建议:")
        print(f"     关注趋势变化, 可选择重训:")
        print(f"     python scripts/auto_research_pipeline.py --strategy {strategy}")


def print_cross_strategy_summary(
    all_verdicts: Dict[str, Dict[str, Any]],
):
    """多策略交叉汇总."""
    if len(all_verdicts) < 2:
        return

    print(f"\n{'╔' + '═'*50 + '╗'}")
    print(f"║  {'全策略配置健康汇总':<44s}║")
    print(f"{'╚' + '═'*50 + '╝'}")
    print(f"\n  {'策略':<8s} {'结论':<12s} {'原因数':>6s}  {'首要原因'}")
    print(f"  {'─'*55}")

    for strategy, verdict in all_verdicts.items():
        v = verdict["verdict"]
        emoji = _verdict_emoji(v)
        n_reasons = len(verdict.get("reasons", []))
        first_reason = verdict["reasons"][0] if verdict["reasons"] else "—"
        print(
            f"  {strategy.upper():<8s} {emoji} {v:<10s} {n_reasons:>4d}    {first_reason}"
        )

    retrain_list = [s for s, v in all_verdicts.items() if v["verdict"] == "RETRAIN"]
    if retrain_list:
        print(f"\n  需要重训: {', '.join(s.upper() for s in retrain_list)}")
        print(f"  一键重训: python scripts/auto_research_pipeline.py --all")
    else:
        print(f"\n  ✅ 所有策略配置健康")


# ====================================================================
# Save report
# ====================================================================


def save_report(
    strategy: str,
    runs: List[Dict[str, Any]],
    drift_results: List[Dict[str, Any]],
    verdict: Dict[str, Any],
    output_dir: Path,
):
    """保存 JSON 报告."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "strategy": strategy,
        "n_runs": len(runs),
        "runs": runs,
        "drift_results": drift_results,
        "verdict": verdict,
    }
    path = output_dir / f"health_check_{strategy}.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


# ====================================================================
# Run one strategy
# ====================================================================


def run_strategy_check(
    strategy: str,
    cfg: dict,
    *,
    min_months: int = 6,
    no_fill: bool = False,
    dry_run: bool = False,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """对单个策略执行完整健康检查."""
    print(f"\n{'='*60}")
    print(f"  检查策略: {strategy.upper()}")
    print(f"{'='*60}")

    # ── 1. 自动发现已有训练数据 ──
    runs = discover_research_runs(strategy, cfg)
    print(f"  发现 {len(runs)} 次有效训练结果")
    if runs:
        print(f"  时间范围: {runs[0]['end_date']} ~ {runs[-1]['end_date']}")

    # ── 2. 检查是否需要补训 ──
    if not no_fill and len(runs) < min_months:
        missing = find_missing_months(runs, min_months)
        if missing:
            print(f"\n  ⚠️  数据不足: 有 {len(runs)} 个月, 需要 {min_months} 个月")
            print(
                f"     缺失: {', '.join(missing[:6])}{'...' if len(missing) > 6 else ''}"
            )
            filled = fill_missing_data(strategy, missing, config_path, dry_run=dry_run)
            if filled > 0 and not dry_run:
                # 重新发现
                runs = discover_research_runs(strategy, cfg)
                print(f"  补训后: {len(runs)} 次有效训练结果")

    # ── 3. 特征漂移检查 (逐月对比) ──
    drift_results: List[Dict[str, Any]] = []
    if len(runs) >= 2:
        # 找第一个 run 的 baseline 作为参考
        first_baseline = find_baseline_for_run(runs[0], strategy)
        if first_baseline:
            for run in runs:
                pred = find_predictions_for_run(run, strategy)
                if pred:
                    drift = check_feature_drift_lightweight(first_baseline, pred)
                    drift_results.append(drift)
                else:
                    drift_results.append({"status": "SKIP", "reason": "no predictions"})
        else:
            # 没有 baseline, 跳过漂移检查
            drift_results = [{"status": "SKIP", "reason": "no baseline"}] * len(runs)
    elif len(runs) == 1:
        drift_results = [{"status": "NONE", "drift_rate": 0}]

    # ── 4. 综合判定 ──
    retrain_cfg = cfg.get("retrain_triggers", {})
    verdict = make_verdict(runs, drift_results, retrain_cfg)

    # ── 5. 输出 ──
    print_strategy_report(strategy, runs, drift_results, verdict)

    # ── 6. 保存 ──
    if output_dir:
        path = save_report(strategy, runs, drift_results, verdict, output_dir)
        print(f"\n  📄 报告: {path}")

    return verdict


# ====================================================================
# Main
# ====================================================================


def main() -> int:
    p = argparse.ArgumentParser(
        description="一键配置健康检查 — 自动发现数据 + 趋势分析 + 结论",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  单策略:           python scripts/check_need_retrain.py --strategy fer
  全部策略:         python scripts/check_need_retrain.py --all
  不自动补训:       python scripts/check_need_retrain.py --strategy fer --no-fill
  指定最少月数:     python scripts/check_need_retrain.py --strategy fer --min-months 8
  dry-run 补训:     python scripts/check_need_retrain.py --all --dry-run
        """,
    )
    p.add_argument("--strategy", nargs="+", help="策略名 (如 fer bpc me, 支持多个)")
    p.add_argument("--all", action="store_true", help="检查 config 中定义的所有策略")
    p.add_argument(
        "--min-months",
        type=int,
        default=6,
        help="最少需要的月数 (不足自动补训, 默认 6)",
    )
    p.add_argument("--no-fill", action="store_true", help="只看已有数据, 不自动补训")
    p.add_argument(
        "--dry-run", action="store_true", help="补训时只打印命令, 不实际执行"
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="配置文件路径")
    p.add_argument(
        "--output", default=None, help="输出 JSON 目录 (默认: reports/health_check/)"
    )
    args = p.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)

    # 确定策略列表
    if args.all:
        strategies = get_available_strategies(cfg)
    elif args.strategy:
        strategies = args.strategy
    else:
        p.error("请指定 --strategy 或 --all")
        return 1

    if not strategies:
        print("❌ 未找到策略配置")
        return 1

    output_dir = (
        Path(args.output) if args.output else PROJECT_ROOT / "reports" / "health_check"
    )

    print("=" * 60)
    print("🔍 配置健康检查")
    print("=" * 60)
    print(f"   策略: {', '.join(s.upper() for s in strategies)}")
    print(f"   最少月数: {args.min_months}")
    print(f"   自动补训: {'否' if args.no_fill else '是'}")
    if args.dry_run:
        print(f"   🏜️  Dry-run 模式")

    # ── 逐策略检查 ──
    all_verdicts: Dict[str, Dict[str, Any]] = {}

    for strategy in strategies:
        if strategy not in cfg.get("strategies", {}):
            print(f"\n  ⚠️  未知策略: {strategy}, 跳过")
            continue
        verdict = run_strategy_check(
            strategy,
            cfg,
            min_months=args.min_months,
            no_fill=args.no_fill,
            dry_run=args.dry_run,
            config_path=config_path,
            output_dir=output_dir,
        )
        all_verdicts[strategy] = verdict

    # ── 多策略汇总 ──
    print_cross_strategy_summary(all_verdicts)

    # ── Exit code ──
    has_retrain = any(v["verdict"] == "RETRAIN" for v in all_verdicts.values())
    has_attention = any(v["verdict"] == "ATTENTION" for v in all_verdicts.values())
    if has_retrain:
        return 2
    elif has_attention:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
