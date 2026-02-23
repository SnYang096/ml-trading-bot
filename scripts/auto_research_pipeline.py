#!/usr/bin/env python3
"""
自动研究流水线 — 一键执行全流程训练 + 结果快照 + 对比决策

功能:
  1. 自动检测最新数据日期, 计算 holdout 窗口 (end - 14 个月)
  2. 下载 + 转换最新月度 aggTrades 数据 (增量, 已有跳过)
  3. 按策略执行完整训练链: DataDownload → FeatureStore → Prepare
     → Prefilter → Direction → Gate → Evidence → EntryFilter
     → Execution → Backtest
  4. 所有阈值优化步骤带 --promote, 写入实验目录 (不覆盖生产 config)
  5. 保存结构化 report.json 到 results/research_history/{strategy}/{timestamp}/
  6. 与上次研究结果对比, 输出确定性决策: ADOPT / KEEP / ALERT
  7. ADOPT 时自动将实验 archetypes 复制回生产 config

  实验目录隔离:
    每次运行自动复制 config/strategies/{strategy}/ 到实验工作区,
    所有 --promote 写入实验副本, 生产 config 仅在 ADOPT 时更新。

用法:
    # 单策略
    python scripts/auto_research_pipeline.py --strategy fer

    # 全部策略
    python scripts/auto_research_pipeline.py --all

    # 指定 end-date (跳过自动检测)
    python scripts/auto_research_pipeline.py --strategy bpc --end-date 2026-01-01

    # 只运行对比 (不重新训练)
    python scripts/auto_research_pipeline.py --strategy fer --compare-only

    # dry-run (打印命令但不执行)
    python scripts/auto_research_pipeline.py --strategy fer --dry-run

    # 列出历史实验
    python scripts/auto_research_pipeline.py --strategy fer --list

    # 手动采纳某次实验
    python scripts/auto_research_pipeline.py --strategy fer --adopt 20260222_120000

    # 对比两次实验的 archetypes 差异
    python scripts/auto_research_pipeline.py --strategy fer --diff 20260220_100000 20260222_120000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

# ====================================================================
# Config
# ====================================================================

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "research_pipeline.yaml"


def load_pipeline_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ====================================================================
# Date helpers
# ====================================================================


def detect_latest_data_date(data_path: str, symbols: str) -> str:
    """扫描 parquet 文件名, 推断最新可用数据月份 (首日)."""
    import glob

    dp = Path(data_path)
    latest_year, latest_month = 2023, 1
    for sym in symbols.split(","):
        # 文件名格式: BTCUSDT_4h_2025.parquet 或 BTCUSDT/2025-12.parquet 等
        for f in dp.rglob(f"*{sym.strip()}*"):
            name = f.stem
            # 尝试提取 YYYY-MM 或 YYYY
            m = re.search(r"(\d{4})-(\d{2})", name)
            if m:
                y, mo = int(m.group(1)), int(m.group(2))
            else:
                m2 = re.search(r"(\d{4})", name)
                if m2:
                    y, mo = int(m2.group(1)), 12
                else:
                    continue
            if (y, mo) > (latest_year, latest_month):
                latest_year, latest_month = y, mo

    # 返回该月下一个月的第一天 (数据"到"这个月 → end-date = 下月 1 号)
    if latest_month == 12:
        end = datetime(latest_year + 1, 1, 1)
    else:
        end = datetime(latest_year, latest_month + 1, 1)
    return end.strftime("%Y-%m-%d")


def compute_holdout_start(end_date: str, holdout_months: int) -> str:
    """end_date - holdout_months → holdout_start_date."""
    end = datetime.strptime(end_date, "%Y-%m-%d")
    # 简单月份减法
    y = end.year
    m = end.month - holdout_months
    while m <= 0:
        m += 12
        y -= 1
    return datetime(y, m, 1).strftime("%Y-%m-%d")


# ====================================================================
# Step runner
# ====================================================================


def run_step(
    name: str,
    cmd: List[str],
    log_file: Path,
    *,
    dry_run: bool = False,
    cwd: Optional[Path] = None,
) -> Tuple[int, str]:
    """执行一个步骤, 输出到 stdout + log file."""
    cmd_str = " \\\n  ".join(cmd)
    header = f"\n{'='*70}\n[STEP] {name}\n{'='*70}\n$ {cmd_str}\n"
    print(header)

    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(header)

    if dry_run:
        print("  (dry-run, 跳过执行)")
        return 0, ""

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd or PROJECT_ROOT,
    )
    output = proc.stdout + proc.stderr

    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(output)
        lf.write(f"\n[EXIT CODE] {proc.returncode}\n")

    # 打印最后 30 行摘要
    lines = output.strip().split("\n")
    summary = "\n".join(lines[-30:]) if len(lines) > 30 else output
    print(summary)

    if proc.returncode != 0:
        print(f"\n❌ Step '{name}' FAILED (exit code {proc.returncode})")
    else:
        print(f"\n✅ Step '{name}' completed")

    return proc.returncode, output


def find_output_dir(output: str, strategy: str) -> Optional[str]:
    """从 mlbot train final 的 stdout 中解析输出目录."""
    # 尝试匹配 "Results saved to results/train_final_XXXXXXXX_..."
    m = re.search(r"(results/train_final_\S+/" + re.escape(strategy) + r")", output)
    if m:
        return m.group(1)
    # fallback: 扫描 results/ 找最新
    results_dir = PROJECT_ROOT / "results"
    candidates = sorted(
        results_dir.glob(f"train_final_*/{strategy}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0].relative_to(PROJECT_ROOT))
    return None


# ====================================================================
# Parse backtest output
# ====================================================================


def parse_backtest_stdout(output: str) -> Dict[str, Any]:
    """从 backtest_execution_layer.py stdout 提取指标."""
    metrics: Dict[str, Any] = {}

    m = re.search(r"Trades:\s*(\d+)", output)
    if m:
        metrics["total_trades"] = int(m.group(1))

    m = re.search(r"Mean R:\s*([\-\d.]+)", output)
    if m:
        metrics["mean_r"] = float(m.group(1))

    m = re.search(r"Win Rate:\s*([\d.]+)%", output)
    if m:
        metrics["win_rate"] = float(m.group(1)) / 100

    m = re.search(r"Sharpe \(per-trade\):\s*([\-\d.]+)", output)
    if m:
        metrics["sharpe_per_trade"] = float(m.group(1))

    m = re.search(r"Sharpe \(annualized\):\s*([\-\d.]+)", output)
    if m:
        metrics["sharpe_annualized"] = float(m.group(1))

    m = re.search(r"Sharpe \(daily.*?\):\s*([\-\d.]+)", output)
    if m:
        metrics["sharpe_daily"] = float(m.group(1))

    return metrics


# ====================================================================
# Snapshot & Compare
# ====================================================================


def snapshot_archetypes(strategy: str, strategy_config: dict, dest: Path):
    """复制当前 archetypes/ 配置到快照目录."""
    src = PROJECT_ROOT / strategy_config["config"] / "archetypes"
    if src.exists():
        dest.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dest / f.name)


def load_archetype_thresholds(strategy: str, strategy_config: dict) -> Dict[str, Any]:
    """读取当前 archetypes/*.yaml 中的关键阈值."""
    thresholds: Dict[str, Any] = {}
    arch_dir = PROJECT_ROOT / strategy_config["config"] / "archetypes"
    for name in ["gate.yaml", "evidence.yaml", "entry_filters.yaml", "execution.yaml"]:
        f = arch_dir / name
        if f.exists():
            thresholds[name] = yaml.safe_load(f.read_text(encoding="utf-8"))
    return thresholds


def find_previous_report(history_dir: Path, strategy: str) -> Optional[Dict[str, Any]]:
    """找到上一次研究的 report.json."""
    strat_dir = history_dir / strategy
    if not strat_dir.exists():
        return None
    runs = sorted(strat_dir.iterdir(), reverse=True)
    for run_dir in runs:
        report = run_dir / "report.json"
        if report.exists():
            return json.loads(report.read_text(encoding="utf-8"))
    return None


def check_deploy_gate(
    decision: str,
    comparison: Dict[str, Any],
    drift_levels: Optional[Dict[str, str]],
    deploy_cfg: dict,
) -> Dict[str, Any]:
    """检查是否满足 deploy 门禁条件.

    双层逻辑:
      1. 触发条件 (OR): 至少一个满足才「值得」deploy
         - Sharpe 显著提升 (>= trigger_sharpe_improve)
         - 漂移 >= HIGH 且 |Sharpe 变化| > min_sharpe_change (参数过时 + 性能有变)
      2. 安全门禁 (AND): 全部满足才「允许」deploy
         - ADOPT 决策
         - min_trades >= 阈值

    Returns: {"deploy_ready": bool, "triggers": [...], "safety": [...],
              "blocked_by": [...], "skip_reason": str|None}
    """
    triggers: List[Dict[str, Any]] = []
    safety: List[Dict[str, Any]] = []
    blocked: List[str] = []
    skip_reason: Optional[str] = None

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

    # ── 触发条件 (OR) ─────────────────────────────────────
    triggered = False

    # T1. Sharpe 提升
    sharpe_thresh = deploy_cfg.get("trigger_sharpe_improve", 0.05)
    prev_sharpe = comparison.get("previous_sharpe")
    cur_sharpe = comparison.get("current_sharpe", 0)
    if prev_sharpe is not None and prev_sharpe != 0:
        improve = (cur_sharpe - prev_sharpe) / abs(prev_sharpe)
        t1_ok = improve >= sharpe_thresh
        triggers.append(
            {
                "rule": "sharpe_improve",
                "value": f"{improve:+.1%}",
                "threshold": f">= {sharpe_thresh:.0%}",
                "pass": t1_ok,
            }
        )
        if t1_ok:
            triggered = True
    else:
        # 首次运行, 无对比基准 → 视为触发 (首版本必须 deploy)
        triggers.append({"rule": "sharpe_improve", "value": "首次运行", "pass": True})
        triggered = True

    # T2. 漂移级别 + Sharpe 稳定性保护
    trigger_drift = deploy_cfg.get("trigger_drift_level", "HIGH")
    min_sharpe_chg = deploy_cfg.get("min_sharpe_change", 0.03)
    if drift_levels:
        overall = max(
            drift_levels.values(), key=lambda x: DRIFT_ORDER.get(x, 0), default="NONE"
        )
        drift_ok = DRIFT_ORDER.get(overall, 0) >= DRIFT_ORDER.get(trigger_drift, 3)
        # 稳定性保护: 即使漂移达标, |Sharpe变化| 须 > min_sharpe_change 才触发
        sharpe_changed = True  # 默认有变化
        if prev_sharpe is not None and prev_sharpe != 0:
            abs_chg = abs(cur_sharpe - prev_sharpe) / abs(prev_sharpe)
            sharpe_changed = abs_chg > min_sharpe_chg
        t2_ok = drift_ok and sharpe_changed
        t2_note = f">= {trigger_drift}"
        if drift_ok and not sharpe_changed:
            t2_note += f" (Sharpe稳定, |变化|<={min_sharpe_chg:.0%}, 不触发)"
        triggers.append(
            {
                "rule": "drift_level",
                "value": overall,
                "threshold": t2_note,
                "pass": t2_ok,
            }
        )
        if t2_ok:
            triggered = True
    else:
        triggers.append({"rule": "drift_level", "value": "无历史对比", "pass": False})

    if not triggered:
        skip_reason = "无触发条件: Sharpe 提升不足 且 漂移较小 → 不需要 deploy"

    # ── 安全门禁 (AND) ────────────────────────────────────
    # S1. require_adopt
    if deploy_cfg.get("require_adopt", True):
        s1_ok = decision == "ADOPT"
        safety.append({"rule": "require_adopt", "value": decision, "pass": s1_ok})
        if not s1_ok:
            blocked.append(f"决策={decision}, 需要 ADOPT")

    # S2. min_trades
    min_trades = deploy_cfg.get("min_trades", 50)
    cur_trades = comparison.get("current_trades", 0)
    s2_ok = cur_trades >= min_trades
    safety.append(
        {
            "rule": "min_trades",
            "value": cur_trades,
            "threshold": min_trades,
            "pass": s2_ok,
        }
    )
    if not s2_ok:
        blocked.append(f"trades={cur_trades} < {min_trades}")

    # ── 最终判定 ──────────────────────────────────────────
    # 必须: 有触发 AND 安全门禁全过
    deploy_ready = triggered and len(blocked) == 0
    return {
        "deploy_ready": deploy_ready,
        "triggered": triggered,
        "triggers": triggers,
        "safety": safety,
        "blocked_by": blocked,
        "skip_reason": skip_reason,
        "require_human_confirm": deploy_cfg.get("require_human_confirm", True),
    }


def compare_runs(
    current: Dict[str, Any],
    previous: Optional[Dict[str, Any]],
    rules: dict,
) -> Dict[str, Any]:
    """确定性对比, 输出决策."""
    min_trades = rules.get("min_trades", 10)
    adopt_ratio = rules.get("sharpe_adopt_ratio", 0.7)
    reject_floor = rules.get("sharpe_reject_floor", 0.0)

    cur_metrics = current.get("backtest_metrics", {})
    cur_trades = cur_metrics.get("total_trades", 0)
    cur_sharpe = cur_metrics.get("sharpe_per_trade", 0.0)

    result = {
        "current_sharpe": cur_sharpe,
        "current_trades": cur_trades,
        "previous_run": None,
        "previous_sharpe": None,
        "sharpe_ratio": None,
        "decision": "ADOPT",
        "reasons": [],
    }

    # Rule 1: 交易数太少
    if cur_trades < min_trades:
        result["decision"] = "ERROR"
        result["reasons"].append(f"trades={cur_trades} < min={min_trades}")
        return result

    # Rule 2: Sharpe <= 0
    if cur_sharpe <= reject_floor:
        result["decision"] = "ALERT"
        result["reasons"].append(f"sharpe={cur_sharpe:.4f} <= floor={reject_floor}")

    # 首次运行
    if previous is None:
        result["reasons"].append("首次运行, 无历史对比")
        if result["decision"] != "ALERT":
            result["decision"] = "ADOPT"
        return result

    prev_metrics = previous.get("backtest_metrics", {})
    prev_sharpe = prev_metrics.get("sharpe_per_trade", 0.0)
    result["previous_run"] = previous.get("timestamp")
    result["previous_sharpe"] = prev_sharpe

    if prev_sharpe > 0:
        ratio = cur_sharpe / prev_sharpe
        result["sharpe_ratio"] = ratio

        if ratio >= adopt_ratio:
            if result["decision"] != "ALERT":
                result["decision"] = "ADOPT"
            result["reasons"].append(f"sharpe_ratio={ratio:.2f} >= {adopt_ratio}")
        else:
            result["decision"] = "ALERT"
            result["reasons"].append(
                f"sharpe_ratio={ratio:.2f} < {adopt_ratio} (显著衰减)"
            )
    else:
        result["reasons"].append(f"prev_sharpe={prev_sharpe:.4f} <= 0, 跳过比值")

    return result


# ====================================================================
# Data download & convert (Step 0)
# ====================================================================


def run_data_download(
    cfg: dict,
    *,
    end_date: str,
    symbols: str,
    log: Path,
    dry_run: bool = False,
) -> int:
    """Step 0: 下载 + 转换最新月度 aggTrades 数据 (增量).

    已有的月份自动跳过, 只下载新增月份.
    """
    dl_cfg = cfg.get("download", {})
    if not dl_cfg.get("enabled", True):
        print("\n⏭️  数据下载已禁用 (download.enabled=false), 跳过")
        return 0

    start_date = cfg["dates"]["start_date"]
    data_dir = dl_cfg.get("data_dir", "data/agg_data")
    parquet_dir = dl_cfg.get("parquet_dir", "data/parquet_data")

    # 从 start_date / end_date 推算 year-month
    sd = datetime.strptime(start_date, "%Y-%m-%d")
    ed = datetime.strptime(end_date, "%Y-%m-%d")

    # Step 0a: Download
    rc, _ = run_step(
        "Data Download",
        [
            "mlbot",
            "data",
            "download",
            "--no-docker",
            "--symbols",
            *[s.strip() for s in symbols.split(",")],
            "--start-year",
            str(sd.year),
            "--start-month",
            str(sd.month),
            "--end-year",
            str(ed.year),
            "--end-month",
            str(ed.month),
            "--data-dir",
            data_dir,
            "--parquet-dir",
            parquet_dir,
            "--yes",
        ],
        log,
        dry_run=dry_run,
    )

    if rc != 0 and not dry_run:
        print("  ⚠️  下载步骤失败, 尝试继续使用本地数据...")

    # Step 0b: Convert (ZIP → Parquet, 增量)
    rc, _ = run_step(
        "Data Convert",
        [
            "mlbot",
            "data",
            "convert",
            "--no-docker",
            "--input-dir",
            data_dir,
            "--output-dir",
            parquet_dir,
        ],
        log,
        dry_run=dry_run,
    )

    if rc != 0 and not dry_run:
        print("  ⚠️  转换步骤失败, 尝试继续使用已有数据...")

    return 0  # 不中断流水线, 即使下载失败也尝试用本地数据


# ====================================================================
# Pipeline: single strategy
# ====================================================================


def run_strategy_pipeline(
    strategy: str,
    cfg: dict,
    *,
    end_date: str,
    holdout_start: str,
    start_date: str,
    symbols: str,
    data_path: str,
    run_dir: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """执行单个策略的完整训练链."""
    scfg = cfg["strategies"][strategy]
    prod_config_dir = scfg["config"]
    timeframe = scfg["timeframe"]
    log = run_dir / "pipeline.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    # ── 实验目录隔离: config 副本到实验工作区 ──────────────────
    exp_strategies_root = run_dir / "strategies"
    exp_config_dir = exp_strategies_root / strategy
    shutil.copytree(
        PROJECT_ROOT / prod_config_dir,
        exp_config_dir,
        dirs_exist_ok=True,
    )
    config_dir = str(exp_config_dir)  # 后续命令全部用实验目录
    strategies_root = str(exp_strategies_root)
    print(f"\n📦 实验配置隔离: {exp_config_dir}")

    common_train_args = [
        "--symbol",
        symbols,
        "--timeframe",
        timeframe,
        "--data-path",
        data_path,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--holdout-start-date",
        holdout_start,
        "--holdout-end-date",
        end_date,
        "--seed",
        "42",
    ]

    # ── Step 0: Data Download + Convert (增量) ──
    run_data_download(
        cfg,
        end_date=end_date,
        symbols=symbols,
        log=log,
        dry_run=dry_run,
    )

    # ── Step 1: Feature Store (增量, 已有月份自动跳过) ──
    rc, _ = run_step(
        "Feature Store",
        [
            "mlbot",
            "feature-store",
            "build",
            "--no-docker",
            "--config",
            config_dir,
            "--symbols",
            symbols,
            "--timeframe",
            timeframe,
            "--start-date",
            start_date,
            "--end-date",
            end_date,
            "--warmup-months",
            "6",
        ],
        log,
        dry_run=dry_run,
    )
    if rc != 0 and not dry_run:
        return {"error": "feature_store_build_failed"}

    # ── Step 2: Prepare-only (features_labeled.parquet) ──
    rc, out = run_step(
        "Prepare Only",
        [
            "mlbot",
            "train",
            "final",
            "--no-docker",
            "--prepare-only",
            "--config",
            config_dir,
            "--features",
            f"{config_dir}/{scfg['features_gate']}",
            "--labels",
            f"{config_dir}/{scfg['labels_gate']}",
            *common_train_args,
        ],
        log,
        dry_run=dry_run,
    )

    prepare_dir = find_output_dir(out, strategy)
    if not prepare_dir and not dry_run:
        return {"error": "prepare_dir_not_found"}
    prepare_dir = prepare_dir or f"results/train_final_DRYRUN/{strategy}"

    # ── Step 3: Prefilter (--promote) ──
    if scfg.get("has_prefilter"):
        run_step(
            "Prefilter Analyze",
            [
                "python",
                "scripts/analyze_archetype_feature_stratification.py",
                "--logs",
                f"{prepare_dir}/features_labeled.parquet",
                "--strategy",
                strategy,
                "--config",
                f"{config_dir}/prefilter.yaml",
                "--select-recent",
                "6",
                "--promote",
            ],
            log,
            dry_run=dry_run,
        )

    # ── Step 4: Direction (--promote) ──
    if scfg.get("has_direction"):
        run_step(
            "Direction Validate",
            [
                "python",
                "z实验_005_统一研究/direction_strict_validation.py",
                "--logs",
                f"{prepare_dir}/features_labeled.parquet",
                "--strategy",
                strategy,
                "--strategies-root",
                strategies_root,
                "--compare-features",
                "--temporal",
                "--promote",
            ],
            log,
            dry_run=dry_run,
        )

    # ── Step 5: Gate 训练 ──
    prefilter_path = f"{config_dir}/archetypes/prefilter.yaml"
    gate_train_args = [
        "mlbot",
        "train",
        "final",
        "--no-docker",
        "--config",
        config_dir,
        "--features",
        f"{config_dir}/{scfg['features_gate']}",
        "--labels",
        f"{config_dir}/{scfg['labels_gate']}",
        *common_train_args,
    ]
    if Path(prefilter_path).exists():
        gate_train_args += ["--archetype-prefilter", prefilter_path]

    rc, out = run_step("Gate Train", gate_train_args, log, dry_run=dry_run)
    gate_dir = find_output_dir(out, strategy) or prepare_dir

    # Gate apply (用 gate_draft)
    gate_draft = f"{config_dir}/gate_draft.yaml"
    run_step(
        "Gate Apply",
        [
            "mlbot",
            "gate",
            "apply-archetype",
            "--logs",
            f"{gate_dir}/predictions.parquet",
            "--strategy",
            strategy,
            "--gate-path",
            gate_draft,
        ],
        log,
        dry_run=dry_run,
    )

    # Gate optimize (--promote)
    run_step(
        "Gate Optimize",
        [
            "python",
            "scripts/optimize_gate_unified.py",
            "--strategy",
            strategy,
            "--strategies-root",
            strategies_root,
            "--logs",
            f"{gate_dir}/logs_gated.parquet",
            "--output",
            f"{gate_dir}/gate_optimization.json",
            "--gate-path",
            gate_draft,
            "--promote",
        ],
        log,
        dry_run=dry_run,
    )

    # Re-apply with optimized gate
    run_step(
        "Gate Re-Apply",
        [
            "mlbot",
            "gate",
            "apply-archetype",
            "--logs",
            f"{gate_dir}/predictions.parquet",
            "--strategy",
            strategy,
            "--gate-path",
            f"{config_dir}/archetypes/gate.yaml",
        ],
        log,
        dry_run=dry_run,
    )

    # ── Step 6: Evidence 训练 ──
    evidence_train_args = [
        "mlbot",
        "train",
        "final",
        "--no-docker",
        "--config",
        config_dir,
        "--features",
        f"{config_dir}/{scfg['features_evidence']}",
        "--labels",
        f"{config_dir}/{scfg['labels_evidence']}",
        *common_train_args,
    ]
    if Path(prefilter_path).exists():
        evidence_train_args += ["--archetype-prefilter", prefilter_path]

    rc, out = run_step("Evidence Train", evidence_train_args, log, dry_run=dry_run)
    evidence_dir = find_output_dir(out, strategy) or gate_dir

    # Evidence gate apply
    run_step(
        "Evidence Gate Apply",
        [
            "mlbot",
            "gate",
            "apply-archetype",
            "--logs",
            f"{evidence_dir}/predictions.parquet",
            "--out",
            f"{evidence_dir}/logs_gated.parquet",
            "--gate-path",
            f"{config_dir}/archetypes/gate.yaml",
            "--strategy",
            strategy,
        ],
        log,
        dry_run=dry_run,
    )

    # Evidence optimize (--promote)
    run_step(
        "Evidence Optimize",
        [
            "python",
            "scripts/optimize_evidence_plateau.py",
            "--strategy",
            strategy,
            "--strategies-root",
            strategies_root,
            "--candidates",
            f"{evidence_dir}/evidence_candidates.yaml",
            "--predictions",
            f"{evidence_dir}/predictions.parquet",
            "--logs",
            f"{evidence_dir}/logs_gated.parquet",
            "--output",
            f"{evidence_dir}/evidence_optimization.json",
            "--promote",
        ],
        log,
        dry_run=dry_run,
    )

    # ── Step 7: Entry Filter (--promote) ──
    run_step(
        "Entry Filter Optimize",
        [
            "python",
            "scripts/optimize_entry_filter_plateau.py",
            "--logs",
            f"{evidence_dir}/predictions.parquet",
            "--strategy",
            strategy,
            "--strategies-root",
            strategies_root,
            "--research",
            "--promote",
        ],
        log,
        dry_run=dry_run,
    )

    # ── Step 8: Execution (--promote) ──
    run_step(
        "Execution Optimize",
        [
            "python",
            "scripts/optimize_execution_grid.py",
            "--logs",
            f"{evidence_dir}/logs_gated.parquet",
            "--strategy",
            strategy,
            "--strategies-root",
            strategies_root,
            "--output",
            f"{evidence_dir}/execution_grid.json",
            "--promote",
        ],
        log,
        dry_run=dry_run,
    )

    # ── Step 9: Backtest ──
    rc, bt_out = run_step(
        "Backtest",
        [
            "python",
            "scripts/backtest_execution_layer.py",
            "--logs",
            f"{evidence_dir}/predictions.parquet",
            "--strategy",
            strategy,
        ],
        log,
        dry_run=dry_run,
    )

    # ── 收集指标 ──
    backtest_metrics = (
        parse_backtest_stdout(bt_out)
        if not dry_run
        else {
            "total_trades": 0,
            "mean_r": 0,
            "win_rate": 0,
            "sharpe_per_trade": 0,
            "sharpe_annualized": 0,
            "sharpe_daily": 0,
        }
    )

    # ── Step 10: 导出训练基线 JSON ──
    if not dry_run:
        try:
            import importlib
            import sys as _sys

            # 确保项目根目录在 sys.path 中
            root_str = str(PROJECT_ROOT)
            if root_str not in _sys.path:
                _sys.path.insert(0, root_str)
            mod = importlib.import_module("scripts.export_training_baseline")
            mod.export_training_baseline(
                strategy=strategy,
                result_dir=Path(evidence_dir),
                gate_dir=Path(gate_dir),
                evidence_dir=Path(evidence_dir),
                backtest_metrics=backtest_metrics,
                config_root=strategies_root,
                training_period={"start": start_date, "end": holdout_start},
                holdout_period={"start": holdout_start, "end": end_date},
            )
        except Exception as exc:
            print(f"\n⚠️  Baseline export failed: {exc}")
    else:
        print("\n  Step 10: Export Training Baseline (dry-run, 跳过)")

    return {
        "gate_dir": gate_dir,
        "evidence_dir": evidence_dir,
        "backtest_metrics": backtest_metrics,
        "exp_config_dir": str(exp_config_dir),
        "prod_config_dir": prod_config_dir,
    }


# ====================================================================
# Save report
# ====================================================================


def save_report(
    strategy: str,
    cfg: dict,
    run_dir: Path,
    pipeline_result: Dict[str, Any],
    comparison: Dict[str, Any],
    *,
    start_date: str,
    end_date: str,
    holdout_start: str,
) -> Path:
    """保存结构化 report.json + archetypes 快照."""
    scfg = cfg["strategies"][strategy]
    timestamp = run_dir.name

    # 从实验目录读取 archetypes (已 promote 的版本)
    exp_config_dir = pipeline_result.get("exp_config_dir")
    if exp_config_dir:
        thresholds = {}
        arch_dir = Path(exp_config_dir) / "archetypes"
        for name in [
            "gate.yaml",
            "evidence.yaml",
            "entry_filters.yaml",
            "execution.yaml",
        ]:
            f = arch_dir / name
            if f.exists():
                thresholds[name] = yaml.safe_load(f.read_text(encoding="utf-8"))
    else:
        thresholds = load_archetype_thresholds(strategy, scfg)

    report = {
        "version": 2,
        "strategy": strategy,
        "timestamp": timestamp,
        "data_range": {
            "start_date": start_date,
            "end_date": end_date,
            "holdout_start": holdout_start,
            "holdout_months": cfg["dates"]["holdout_months"],
        },
        "backtest_metrics": pipeline_result.get("backtest_metrics", {}),
        "thresholds": thresholds,
        "comparison": comparison,
        "artifacts": {
            "gate_dir": pipeline_result.get("gate_dir"),
            "evidence_dir": pipeline_result.get("evidence_dir"),
            "exp_config_dir": exp_config_dir,
        },
    }

    # Save report.json
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )

    # 实验 archetypes 已在 run_dir/strategies/{strategy}/archetypes/ 中
    # 不再复制快照副本到 run_dir/archetypes/ (冗余)

    # Save comparison
    comp_path = run_dir / "comparison.json"
    comp_path.write_text(
        json.dumps(comparison, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    return report_path


def _patch_report_deploy(report_path: Path, deploy_result: Dict[str, Any]):
    """将 deploy 门禁结果追加到已保存的 report.json."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["deploy_gate"] = {
            "deploy_ready": deploy_result["deploy_ready"],
            "triggered": deploy_result.get("triggered", False),
            "triggers": deploy_result.get("triggers", []),
            "safety": deploy_result.get("safety", []),
            "blocked_by": deploy_result.get("blocked_by", []),
            "skip_reason": deploy_result.get("skip_reason"),
        }
        report_path.write_text(
            json.dumps(report, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass  # 非关键路径, 不影响主流程


# ====================================================================
# Main
# ====================================================================


def main():
    p = argparse.ArgumentParser(description="自动研究流水线 (实验隔离版)")
    p.add_argument("--strategy", help="策略名 (bpc/fer/me)")
    p.add_argument("--all", action="store_true", help="执行所有策略")
    p.add_argument("--end-date", help="数据截止日期 (默认自动检测)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="pipeline 配置文件")
    p.add_argument("--compare-only", action="store_true", help="只对比, 不重训")
    p.add_argument("--dry-run", action="store_true", help="打印命令但不执行")
    p.add_argument(
        "--no-adopt", action="store_true", help="禁止自动采纳, 仅保存实验结果"
    )
    p.add_argument(
        "--list",
        dest="list_experiments",
        action="store_true",
        help="列出历史实验及其 metrics",
    )
    p.add_argument(
        "--adopt",
        metavar="TIMESTAMP",
        help="手动采纳指定时间戳的实验 (如 20260222_120000)",
    )
    p.add_argument(
        "--diff",
        nargs=2,
        metavar="TS",
        help="对比两次实验的 archetypes 差异 (如 --diff TS1 TS2)",
    )
    args = p.parse_args()

    cfg = load_pipeline_config(Path(args.config))
    history_dir = PROJECT_ROOT / cfg["output"]["history_dir"]

    # ── 子命令: 列出历史实验 ──
    if args.list_experiments:
        if not args.strategy and not args.all:
            p.error("--list 需要指定 --strategy 或 --all")
        strats = list(cfg["strategies"].keys()) if args.all else [args.strategy]
        for s in strats:
            _cmd_list_experiments(history_dir, s)
        return

    # ── 子命令: 手动采纳实验 ──
    if args.adopt:
        if not args.strategy:
            p.error("--adopt 需要指定 --strategy")
        _cmd_adopt_experiment(history_dir, cfg, args.strategy, args.adopt)
        return

    # ── 子命令: 对比两次实验 ──
    if args.diff:
        if not args.strategy:
            p.error("--diff 需要指定 --strategy")
        _cmd_diff_experiments(history_dir, args.strategy, args.diff[0], args.diff[1])
        return

    if not args.strategy and not args.all:
        p.error("必须指定 --strategy 或 --all")

    dates = cfg["dates"]
    symbols = cfg["symbols"]
    data_path = cfg["data_path"]
    start_date = dates["start_date"]

    # ── 自动检测日期 ──
    if args.end_date:
        end_date = args.end_date
    else:
        end_date = detect_latest_data_date(data_path, symbols)
    holdout_start = compute_holdout_start(end_date, dates["holdout_months"])

    print("=" * 70)
    print("🚀 自动研究流水线")
    print("=" * 70)
    print(f"   数据范围:    {start_date} ~ {end_date}")
    print(f"   Train:       {start_date} ~ {holdout_start}")
    print(
        f"   Holdout:     {holdout_start} ~ {end_date} ({dates['holdout_months']} 个月)"
    )
    print(f"   Symbols:     {symbols}")
    print(f"   History:     {history_dir}")
    if args.dry_run:
        print("   Mode:        DRY RUN")
    print("=" * 70)

    # ── 确定策略列表 ──
    if args.all:
        strategies = list(cfg["strategies"].keys())
    else:
        strategies = [args.strategy]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_summary = []

    for strategy in strategies:
        if strategy not in cfg["strategies"]:
            print(f"\n❌ 未知策略: {strategy}, 跳过")
            continue

        run_dir = history_dir / strategy / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"# 策略: {strategy.upper()}")
        print(f"# 输出: {run_dir}")
        print(f"{'#'*70}")

        if args.compare_only:
            # 只对比
            prev = find_previous_report(history_dir, strategy)
            if prev:
                print(f"\n📊 上次运行: {prev.get('timestamp')}")
                print(
                    f"   Sharpe: {prev.get('backtest_metrics', {}).get('sharpe_per_trade', 'N/A')}"
                )
            else:
                print("\n📊 无历史记录")
            continue

        # ── 执行流水线 ──
        pipeline_result = run_strategy_pipeline(
            strategy,
            cfg,
            end_date=end_date,
            holdout_start=holdout_start,
            start_date=start_date,
            symbols=symbols,
            data_path=data_path,
            run_dir=run_dir,
            dry_run=args.dry_run,
        )

        if "error" in pipeline_result:
            print(f"\n❌ Pipeline failed: {pipeline_result['error']}")
            results_summary.append(
                {
                    "strategy": strategy,
                    "decision": "ERROR",
                    "reason": pipeline_result["error"],
                }
            )
            continue

        # ── 对比决策 ──
        prev = find_previous_report(history_dir, strategy)
        comparison = compare_runs(
            {"backtest_metrics": pipeline_result["backtest_metrics"]},
            prev,
            cfg.get("comparison", {}),
        )

        # ── 保存 ──
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

        # ── 打印决策 ──
        decision = comparison["decision"]
        emoji = {"ADOPT": "✅", "ALERT": "⚠️", "KEEP": "🔄", "ERROR": "❌"}.get(
            decision, "❓"
        )

        bt = pipeline_result["backtest_metrics"]
        print(f"\n{'='*70}")
        print(f"📊 {strategy.upper()} 研究结果")
        print(f"{'='*70}")
        print(f"   Trades:      {bt.get('total_trades', 'N/A')}")
        print(f"   Sharpe:      {bt.get('sharpe_per_trade', 'N/A')}")
        print(f"   Win Rate:    {bt.get('win_rate', 'N/A')}")
        print(f"   Mean R:      {bt.get('mean_r', 'N/A')}")
        if prev:
            prev_bt = prev.get("backtest_metrics", {})
            print(f"\n   上次 Sharpe:  {prev_bt.get('sharpe_per_trade', 'N/A')}")
            print(f"   变化比:       {comparison.get('sharpe_ratio', 'N/A')}")
        print(f"\n   {emoji} 决策: {decision}")
        for r in comparison.get("reasons", []):
            print(f"      → {r}")
        print(f"\n   📁 Report: {report_path}")
        print(f"   📦 实验配置: {run_dir}/strategies/{strategy}/archetypes/")

        # ── 漂移报告 (当存在上次实验时自动输出) ──
        drift_levels = None
        if prev and not args.dry_run:
            prev_ts = prev.get("timestamp", "")
            prev_dir = history_dir / strategy / prev_ts
            prev_arch = _find_arch_dir(prev_dir, strategy)
            cur_arch = _find_arch_dir(run_dir, strategy)
            if prev_arch and cur_arch:
                drift_levels = _print_drift_report(
                    strategy,
                    prev_ts,
                    timestamp,
                    prev_arch,
                    cur_arch,
                    prev.get("backtest_metrics", {}),
                    bt,
                )

        # ── Deploy 门禁检查 ──
        deploy_cfg = cfg.get("deploy_gate", {})
        deploy_result = check_deploy_gate(
            decision, comparison, drift_levels, deploy_cfg
        )
        deploy_ready = deploy_result["deploy_ready"]

        # 打印 deploy 状态
        if deploy_ready:
            print(f"\n   🚀 Deploy: ✅ 值得且允许 deploy")
            if deploy_result.get("require_human_confirm"):
                print(
                    f"      运行: python scripts/deploy_config_to_live.py --diff --strategy {strategy}"
                )
                print(
                    f"      确认后: python scripts/deploy_config_to_live.py --deploy --strategy {strategy}"
                )
        elif not deploy_result.get("triggered"):
            print(
                f"\n   ⏭️  Deploy: SKIP — {deploy_result.get('skip_reason', '无触发条件')}"
            )
        else:
            print(f"\n   🚫 Deploy: ❌ 有触发但安全门禁未通过")
            for b in deploy_result["blocked_by"]:
                print(f"      ❌ {b}")

        # 写入 report.json
        _patch_report_deploy(report_path, deploy_result)

        # ── 自动采纳 ──
        prod_config_dir = pipeline_result.get("prod_config_dir")
        exp_cfg_dir = pipeline_result.get("exp_config_dir")
        if (
            decision == "ADOPT"
            and not args.no_adopt
            and prod_config_dir
            and exp_cfg_dir
        ):
            _adopt_experiment_config(Path(exp_cfg_dir), prod_config_dir)
        elif decision == "ADOPT" and args.no_adopt:
            print(f"\n   ⏭️  --no-adopt: 跳过自动采纳, 可后续手动:")
            print(
                f"      python scripts/auto_research_pipeline.py --strategy {strategy} --adopt {timestamp}"
            )

        results_summary.append(
            {
                "strategy": strategy,
                "decision": decision,
                "sharpe": bt.get("sharpe_per_trade"),
                "trades": bt.get("total_trades"),
            }
        )

    # ── 汇总 ──
    print(f"\n{'='*70}")
    print("📋 汇总")
    print(f"{'='*70}")
    for r in results_summary:
        emoji = {"ADOPT": "✅", "ALERT": "⚠️", "KEEP": "🔄", "ERROR": "❌"}.get(
            r["decision"], "❓"
        )
        print(
            f"   {emoji} {r['strategy']:>6s}: {r['decision']:<8s} sharpe={r.get('sharpe', 'N/A')} trades={r.get('trades', 'N/A')}"
        )


# ====================================================================
# 实验管理子命令
# ====================================================================


def _adopt_experiment_config(exp_config_dir: Path, prod_config_dir: str) -> bool:
    """将实验 archetypes 复制回生产 config."""
    exp_arch = exp_config_dir / "archetypes"
    prod_arch = PROJECT_ROOT / prod_config_dir / "archetypes"

    if not exp_arch.exists():
        print(f"   ❌ 实验 archetypes 不存在: {exp_arch}")
        return False

    prod_arch.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in exp_arch.iterdir():
        if f.is_file():
            shutil.copy2(f, prod_arch / f.name)
            copied += 1

    # 也复制 gate_draft.yaml (如果有)
    exp_draft = exp_config_dir / "gate_draft.yaml"
    if exp_draft.exists():
        shutil.copy2(exp_draft, PROJECT_ROOT / prod_config_dir / "gate_draft.yaml")

    print(f"   ✅ Adopted: {copied} files → {prod_arch}")
    return True


def _cmd_list_experiments(history_dir: Path, strategy: str):
    """列出指定策略的所有历史实验."""
    strat_dir = history_dir / strategy
    if not strat_dir.exists():
        print(f"\n📋 {strategy.upper()}: 无历史实验")
        return

    runs = sorted(strat_dir.iterdir())
    print(f"\n📋 {strategy.upper()} 历史实验 ({len(runs)} 次):")
    print(f"{'─'*80}")
    print(f"  {'时间戳':<22s} {'Sharpe':>10s} {'Trades':>8s} {'决策':>8s}  备注")
    print(f"{'─'*80}")

    for run_dir in runs:
        report_file = run_dir / "report.json"
        if not report_file.exists():
            print(f"  {run_dir.name:<22s}  (无 report.json)")
            continue

        report = json.loads(report_file.read_text(encoding="utf-8"))
        bt = report.get("backtest_metrics", {})
        comp = report.get("comparison", {})
        decision = comp.get("decision", "?")
        sharpe = bt.get("sharpe_per_trade", "N/A")
        trades = bt.get("total_trades", "N/A")
        dr = report.get("data_range", {})
        note = f"{dr.get('start_date', '?')}~{dr.get('end_date', '?')}"

        emoji = {"ADOPT": "✅", "ALERT": "⚠️", "KEEP": "🔄", "ERROR": "❌"}.get(
            decision, "❓"
        )
        sharpe_str = (
            f"{sharpe:.4f}" if isinstance(sharpe, (int, float)) else str(sharpe)
        )
        print(
            f"  {run_dir.name:<22s} {sharpe_str:>10s} {str(trades):>8s} {emoji}{decision:>6s}  {note}"
        )


def _cmd_adopt_experiment(history_dir: Path, cfg: dict, strategy: str, timestamp: str):
    """手动采纳指定实验."""
    run_dir = history_dir / strategy / timestamp
    if not run_dir.exists():
        print(f"❌ 实验不存在: {run_dir}")
        # 列出可用的
        strat_dir = history_dir / strategy
        if strat_dir.exists():
            available = [d.name for d in sorted(strat_dir.iterdir()) if d.is_dir()]
            if available:
                print(f"   可用: {', '.join(available[-5:])}")
        return

    scfg = cfg["strategies"][strategy]
    exp_config_dir = run_dir / "strategies" / strategy
    if not exp_config_dir.exists():
        # 旧版实验 (无隔离), 尝试从 archetypes 快照恢复
        arch_snapshot = run_dir / "archetypes"
        if arch_snapshot.exists():
            prod_arch = PROJECT_ROOT / scfg["config"] / "archetypes"
            for f in arch_snapshot.iterdir():
                if f.is_file():
                    shutil.copy2(f, prod_arch / f.name)
            print(f"✅ Adopted (from snapshot): {prod_arch}")
        else:
            print(f"❌ 实验目录中找不到 strategies/ 或 archetypes/ 快照")
        return

    _adopt_experiment_config(exp_config_dir, scfg["config"])


def _cmd_diff_experiments(history_dir: Path, strategy: str, ts1: str, ts2: str):
    """对比两次实验 — 输出结构化漂移报告."""
    dir1 = history_dir / strategy / ts1
    dir2 = history_dir / strategy / ts2

    for d, ts in [(dir1, ts1), (dir2, ts2)]:
        if not d.exists():
            print(f"❌ 实验不存在: {d}")
            return

    arch1 = _find_arch_dir(dir1, strategy)
    arch2 = _find_arch_dir(dir2, strategy)
    if not arch1 or not arch2:
        print("❌ 至少一个实验缺少 archetypes 数据")
        return

    rpt1 = _load_report_metrics(dir1)
    rpt2 = _load_report_metrics(dir2)

    _print_drift_report(strategy, ts1, ts2, arch1, arch2, rpt1, rpt2)


def _find_arch_dir(run_dir: Path, strategy: str) -> Optional[Path]:
    """查找 archetypes 目录 (优先实验隔离版, fallback 快照)."""
    exp_arch = run_dir / "strategies" / strategy / "archetypes"
    if exp_arch.exists():
        return exp_arch
    snap_arch = run_dir / "archetypes"
    if snap_arch.exists():
        return snap_arch
    return None


def _load_report_metrics(run_dir: Path) -> Dict[str, Any]:
    rpt = run_dir / "report.json"
    if rpt.exists():
        r = json.loads(rpt.read_text(encoding="utf-8"))
        return r.get("backtest_metrics", {})
    return {}


# ── 漂移报告核心 ─────────────────────────────────────────────────


def _pct_change(old: float, new: float) -> str:
    if old == 0:
        return "N/A"
    pct = (new - old) / abs(old) * 100
    return f"{pct:+.1f}%"


def _drift_level(changes: List[str]) -> str:
    """从子项漂移标记列表中取最高."""
    order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
    level = max(changes, key=lambda x: order.get(x, 0), default="NONE")
    return level


def _drift_emoji(level: str) -> str:
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "NONE": "⚪"}.get(level, "❓")


def _analyze_prefilter(y1: dict, y2: dict) -> Tuple[List[str], str]:
    """分析 prefilter.yaml 漂移."""
    lines: List[str] = []
    drifts: List[str] = []

    r1 = y1.get("rules", [])
    r2 = y2.get("rules", [])

    # 提取所有 feature->value 对
    def _extract_features(rules: list) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for rule in rules:
            if isinstance(rule, dict):
                if "any_of" in rule:
                    for sub in rule["any_of"]:
                        out[sub.get("feature", "?")] = sub.get("value", 0)
                elif "feature" in rule:
                    out[rule["feature"]] = rule.get("value", 0)
        return out

    f1 = _extract_features(r1)
    f2 = _extract_features(r2)
    feats1 = set(f1.keys())
    feats2 = set(f2.keys())

    if feats1 == feats2:
        lines.append(f"   Rules 特征: 不变 ({', '.join(sorted(feats1))})")
    else:
        added = feats2 - feats1
        removed = feats1 - feats2
        if added:
            lines.append(f"   Rules 新增特征: {', '.join(sorted(added))}")
        if removed:
            lines.append(f"   Rules 移除特征: {', '.join(sorted(removed))}")
        drifts.append("HIGH")

    # 共有特征的阈值漂移
    for feat in sorted(feats1 & feats2):
        v1, v2 = f1[feat], f2[feat]
        if v1 != v2:
            lines.append(f"   阈值漂移: {feat} {v1} → {v2} ({_pct_change(v1, v2)})")
            pct = abs(v2 - v1) / max(abs(v1), 1e-9) * 100
            drifts.append("MEDIUM" if pct < 20 else "HIGH")
        else:
            drifts.append("NONE")

    if not drifts:
        drifts.append("NONE")
    level = _drift_level(drifts)
    return lines, level


def _analyze_gate(y1: dict, y2: dict) -> Tuple[List[str], str]:
    """分析 gate.yaml 漂移."""
    lines: List[str] = []
    drifts: List[str] = []

    hg1 = y1.get("hard_gates", [])
    hg2 = y2.get("hard_gates", [])
    ids1 = {r.get("id", f"rule_{i}"): r for i, r in enumerate(hg1)}
    ids2 = {r.get("id", f"rule_{i}"): r for i, r in enumerate(hg2)}
    set1, set2 = set(ids1.keys()), set(ids2.keys())

    lines.append(f"   规则数: {len(hg1)} → {len(hg2)}")
    added = set2 - set1
    removed = set1 - set2
    common = set1 & set2

    if added:
        lines.append(f"   新增规则: {', '.join(sorted(added))}")
        drifts.append("MEDIUM")
    if removed:
        lines.append(f"   移除规则: {', '.join(sorted(removed))}")
        drifts.append("HIGH" if len(removed) > 2 else "MEDIUM")

    # 共有规则阈值对比
    changed_count = 0
    for rid in sorted(common):
        r1, r2 = ids1[rid], ids2[rid]
        w1, w2 = r1.get("when", {}), r2.get("when", {})

        # 提取阈值
        def _get_threshold(when: dict) -> Optional[float]:
            for feat, conds in when.items():
                if isinstance(conds, dict):
                    for k, v in conds.items():
                        if k.startswith("value_") and isinstance(v, (int, float)):
                            return float(v)
            return None

        t1, t2 = _get_threshold(w1), _get_threshold(w2)
        if t1 is not None and t2 is not None and t1 != t2:
            lines.append(f"   {rid}: 阈值 {t1:.4f} → {t2:.4f} ({_pct_change(t1, t2)})")
            changed_count += 1
            pct = abs(t2 - t1) / max(abs(t1), 1e-9) * 100
            drifts.append("MEDIUM" if pct < 30 else "HIGH")

    if changed_count == 0 and not added and not removed:
        lines.append("   阈值: 全部不变")
        drifts.append("NONE")

    level = _drift_level(drifts) if drifts else "NONE"
    return lines, level


def _analyze_evidence(y1: dict, y2: dict) -> Tuple[List[str], str]:
    """分析 evidence.yaml 漂移."""
    lines: List[str] = []
    drifts: List[str] = []

    def _get_features(y: dict) -> Dict[str, dict]:
        feats = y.get("features", y.get("evidence_features", []))
        if isinstance(feats, list):
            return {
                f.get("name", f.get("feature", f"feat_{i}")): f
                for i, f in enumerate(feats)
            }
        return {}

    f1, f2 = _get_features(y1), _get_features(y2)
    set1, set2 = set(f1.keys()), set(f2.keys())

    if set1 == set2:
        lines.append(f"   特征集合: 不变 ({len(set1)} 个)")
    else:
        added = set2 - set1
        removed = set1 - set2
        if added:
            lines.append(f"   新增特征: {', '.join(sorted(added))}")
            drifts.append("MEDIUM")
        if removed:
            lines.append(f"   移除特征: {', '.join(sorted(removed))}")
            drifts.append("MEDIUM")

    # 共有特征阈值对比
    for fname in sorted(set1 & set2):
        e1, e2 = f1[fname], f2[fname]
        for key in ["threshold", "weight", "min_score", "value"]:
            v1 = e1.get(key)
            v2 = e2.get(key)
            if v1 is not None and v2 is not None and v1 != v2:
                lines.append(
                    f"   {fname}.{key}: {v1} → {v2} ({_pct_change(float(v1), float(v2))})"
                )
                drifts.append("LOW")

    if not drifts:
        drifts.append("NONE")
    return lines, _drift_level(drifts)


def _analyze_execution(y1: dict, y2: dict) -> Tuple[List[str], str]:
    """分析 execution.yaml 漂移."""
    lines: List[str] = []
    drifts: List[str] = []

    for section in ["stop_loss", "take_profit", "trailing_stop"]:
        s1 = y1.get(section, {})
        s2 = y2.get(section, {})
        if not isinstance(s1, dict) or not isinstance(s2, dict):
            continue
        all_keys = sorted(set(list(s1.keys()) + list(s2.keys())))
        for k in all_keys:
            v1, v2 = s1.get(k), s2.get(k)
            if v1 == v2:
                continue
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                lines.append(
                    f"   {section}.{k}: {v1} → {v2} ({_pct_change(float(v1), float(v2))})"
                )
                pct = abs(v2 - v1) / max(abs(v1), 1e-9) * 100
                drifts.append("LOW" if pct < 15 else "MEDIUM")
            elif v1 != v2:
                lines.append(f"   {section}.{k}: {v1} → {v2}")
                drifts.append("LOW")

    if not drifts:
        lines.append("   参数: 全部不变")
        drifts.append("NONE")
    return lines, _drift_level(drifts)


def _analyze_direction(y1: dict, y2: dict) -> Tuple[List[str], str]:
    """分析 direction.yaml 漂移."""
    lines: List[str] = []
    drifts: List[str] = []

    # primary feature
    p1 = y1.get("primary", y1.get("direction_feature", ""))
    p2 = y2.get("primary", y2.get("direction_feature", ""))
    if p1 == p2:
        lines.append(f"   主特征: {p1} (不变)")
    else:
        lines.append(f"   主特征: {p1} → {p2}")
        drifts.append("HIGH")

    # fallback features
    eval1 = y1.get("last_evaluation", {})
    eval2 = y2.get("last_evaluation", {})
    fb1 = [f.get("feature", "") for f in eval1.get("fallback", [])]
    fb2 = [f.get("feature", "") for f in eval2.get("fallback", [])]
    fb_common = len(set(fb1) & set(fb2))
    fb_total = max(len(set(fb1) | set(fb2)), 1)
    if fb1 == fb2:
        lines.append(f"   Fallback 候选: 不变 ({len(fb1)} 个)")
    else:
        lines.append(f"   Fallback 候选: {fb_common}/{fb_total} 个重合")
        overlap = fb_common / fb_total
        drifts.append("LOW" if overlap > 0.6 else "MEDIUM")

    # n_rows change
    nr1 = eval1.get("n_rows", 0)
    nr2 = eval2.get("n_rows", 0)
    if nr1 and nr2 and nr1 != nr2:
        lines.append(f"   数据量: {nr1:,} → {nr2:,} ({_pct_change(nr1, nr2)})")

    if not drifts:
        drifts.append("NONE")
    return lines, _drift_level(drifts)


def _analyze_entry_filters(y1: dict, y2: dict) -> Tuple[List[str], str]:
    """分析 entry_filters.yaml 漂移."""
    lines: List[str] = []
    drifts: List[str] = []

    filters1 = y1.get("filters", [])
    filters2 = y2.get("filters", [])
    ids1 = {f.get("id", f"f{i}"): f for i, f in enumerate(filters1)}
    ids2 = {f.get("id", f"f{i}"): f for i, f in enumerate(filters2)}
    set1, set2 = set(ids1.keys()), set(ids2.keys())

    lines.append(f"   Filter 数: {len(filters1)} → {len(filters2)}")
    added = set2 - set1
    removed = set1 - set2
    if added:
        lines.append(f"   新增: {', '.join(sorted(added))}")
        drifts.append("MEDIUM")
    if removed:
        lines.append(f"   移除: {', '.join(sorted(removed))}")
        drifts.append("MEDIUM")

    # 共有 filter 的 enabled/threshold 对比
    for fid in sorted(set1 & set2):
        ef1, ef2 = ids1[fid], ids2[fid]
        en1, en2 = ef1.get("enabled", True), ef2.get("enabled", True)
        if en1 != en2:
            lines.append(f"   {fid}: enabled {en1} → {en2}")
            drifts.append("MEDIUM")
        # threshold
        for key in ["threshold", "value", "min_value", "max_value"]:
            v1, v2 = ef1.get(key), ef2.get(key)
            if v1 is not None and v2 is not None and v1 != v2:
                lines.append(f"   {fid}.{key}: {v1} → {v2}")
                drifts.append("LOW")

    if not drifts:
        drifts.append("NONE")
    return lines, _drift_level(drifts)


def _analyze_generic(y1: dict, y2: dict) -> Tuple[List[str], str]:
    """通用 YAML 对比 (holding.yaml 等)."""
    if y1 == y2:
        return ["   无变化"], "NONE"
    lines: List[str] = []
    _flat_diff(y1, y2, lines, prefix="   ")
    level = "LOW" if len(lines) <= 3 else "MEDIUM"
    return lines, level


def _flat_diff(
    d1: dict, d2: dict, lines: List[str], prefix: str = "", max_lines: int = 10
):
    """递归扁平化 diff, 最多 max_lines 行."""
    all_keys = sorted(set(list(d1.keys()) + list(d2.keys())))
    for k in all_keys:
        if len(lines) >= max_lines:
            lines.append(f"{prefix}... (更多差异省略)")
            return
        v1, v2 = d1.get(k), d2.get(k)
        if v1 == v2:
            continue
        if k not in d1:
            lines.append(f"{prefix}+ {k}")
        elif k not in d2:
            lines.append(f"{prefix}- {k}")
        elif isinstance(v1, dict) and isinstance(v2, dict):
            _flat_diff(v1, v2, lines, prefix, max_lines)
        elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            lines.append(
                f"{prefix}{k}: {v1} → {v2} ({_pct_change(float(v1), float(v2))})"
            )
        else:
            # 对于 list 等复杂类型, 只显示有/无变化
            lines.append(f"{prefix}{k}: 已变更")


_FILE_ANALYZERS = {
    "prefilter.yaml": _analyze_prefilter,
    "gate.yaml": _analyze_gate,
    "evidence.yaml": _analyze_evidence,
    "execution.yaml": _analyze_execution,
    "direction.yaml": _analyze_direction,
    "entry_filters.yaml": _analyze_entry_filters,
}


def _print_drift_report(
    strategy: str,
    ts1: str,
    ts2: str,
    arch1: Path,
    arch2: Path,
    metrics1: Dict[str, Any],
    metrics2: Dict[str, Any],
) -> Dict[str, str]:
    """输出结构化漂移报告, 返回 {filename: drift_level}."""
    w = 72
    print(f"\n{'╔' + '═' * w + '╗'}")
    print(f"║  {strategy.upper()} Archetype 漂移报告{' ' * (w - len(strategy) - 22)}║")
    print(f"║  旧: {ts1}   新: {ts2}{' ' * (w - len(ts1) - len(ts2) - 12)}║")
    print(f"{'╚' + '═' * w + '╝'}")

    # ── Metrics 对比 ──
    print(f"\n📊 Metrics 对比")
    print(f"   {'─' * 56}")
    print(f"   {'指标':<16s} {'旧':>12s} {'新':>12s} {'变化':>10s}")
    print(f"   {'─' * 56}")
    for key, label, fmt in [
        ("sharpe_per_trade", "Sharpe", ".4f"),
        ("total_trades", "Trades", ".0f"),
        ("win_rate", "Win Rate", ".2%"),
        ("mean_r", "Mean R", ".4f"),
    ]:
        v1 = metrics1.get(key)
        v2 = metrics2.get(key)
        if v1 is not None and v2 is not None:
            s1 = f"{v1:{fmt}}" if isinstance(v1, (int, float)) else str(v1)
            s2 = f"{v2:{fmt}}" if isinstance(v2, (int, float)) else str(v2)
            chg = (
                _pct_change(float(v1), float(v2))
                if isinstance(v1, (int, float)) and v1 != 0
                else ""
            )
            print(f"   {label:<16s} {s1:>12s} {s2:>12s} {chg:>10s}")
    print(f"   {'─' * 56}")

    # ── 逐文件分析 ──
    file_drifts: Dict[str, str] = {}
    all_files = sorted(
        set(
            [f.name for f in arch1.iterdir() if f.is_file()]
            + [f.name for f in arch2.iterdir() if f.is_file()]
        )
    )

    for fname in all_files:
        f1_path, f2_path = arch1 / fname, arch2 / fname
        if not f1_path.exists():
            print(f"\n📄 {fname}: 仅存在于新版 ⚡")
            file_drifts[fname] = "HIGH"
            continue
        if not f2_path.exists():
            print(f"\n📄 {fname}: 新版中已移除 ⚡")
            file_drifts[fname] = "HIGH"
            continue

        text1, text2 = f1_path.read_text(encoding="utf-8"), f2_path.read_text(
            encoding="utf-8"
        )
        if text1 == text2:
            print(f"\n📄 {fname}: 无变化 ✅")
            file_drifts[fname] = "NONE"
            continue

        try:
            y1 = yaml.safe_load(text1) or {}
            y2 = yaml.safe_load(text2) or {}
        except Exception:
            print(f"\n📄 {fname}: 有差异 (YAML 解析失败)")
            file_drifts[fname] = "MEDIUM"
            continue

        analyzer = _FILE_ANALYZERS.get(fname, _analyze_generic)
        detail_lines, level = analyzer(y1, y2)
        emoji = _drift_emoji(level)
        print(f"\n📄 {fname}: {emoji} {level}")
        for line in detail_lines:
            print(line)
        file_drifts[fname] = level

    # ── 综合判定 ──
    overall = _drift_level(list(file_drifts.values()))
    overall_emoji = _drift_emoji(overall)

    # 决定建议
    sharpe1 = metrics1.get("sharpe_per_trade", 0)
    sharpe2 = metrics2.get("sharpe_per_trade", 0)
    sharpe_stable = abs(sharpe2 - sharpe1) / max(abs(sharpe1), 1e-9) < 0.05  # < 5% 变化

    if overall == "NONE" or (overall == "LOW" and sharpe_stable):
        advice = "STABLE — 参数稳定, 可直接 ADOPT"
    elif overall in ("LOW", "MEDIUM") and sharpe_stable:
        advice = "MONITOR — Sharpe 稳定但参数有漂移, 建议检查变动项后 ADOPT"
    elif overall == "MEDIUM" and not sharpe_stable:
        advice = "REVIEW — 参数与 Sharpe 同时漂移, 需人工审查变动原因"
    else:  # HIGH
        high_files = [f for f, l in file_drifts.items() if l == "HIGH"]
        advice = f"ADJUST — 大幅漂移 ({', '.join(high_files)}), 需人工审查并可能回退"

    print(f"\n{'━' * 74}")
    print(f"🎯 综合判定")
    print(f"   总体漂移: {overall_emoji} {overall}")
    print(f"   建议:     {advice}")
    print(f"{'━' * 74}")

    return file_drifts


if __name__ == "__main__":
    main()
