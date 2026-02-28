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
    seed: int = 42,
    dry_run: bool = False,
    use_1min: bool = False,
    live_root: str = "live/highcap",
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

    # ── 加载 per-strategy KPI gates ──
    kpi_gates = scfg.get("kpi_gates", {})
    prefilter_gates = kpi_gates.get("prefilter", {})
    gate_gates = kpi_gates.get("gate", {})
    backtest_gates = kpi_gates.get("backtest", {})
    execution_gates = kpi_gates.get("execution", {})

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
        str(seed),
        # NOTE: --deterministic NOT passed → multi-thread for speed.
        # Multi-seed search provides controlled exploration;
        # non-determinism from threads is acceptable "free" variation.
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
        prefilter_cmd = [
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
        ]
        # 从 kpi_gates 注入 prefilter 约束
        if prefilter_gates.get("min_pass_rate"):
            prefilter_cmd += [
                "--min-prefilter-pass-rate",
                str(prefilter_gates["min_pass_rate"]),
            ]
        if prefilter_gates.get("min_rows"):
            prefilter_cmd += [
                "--min-prefilter-rows",
                str(prefilter_gates["min_rows"]),
            ]
        run_step(
            "Prefilter Analyze",
            prefilter_cmd,
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
    # ⚠️ Gate Train 现在使用 prefilter (见 BPC pipeline 文档):
    #   - 只在 archetype 适用样本上学习 → 专注策略特有特征
    #   - 避免学习 "archetype vs 非 archetype" 而不是 "好 archetype vs 坏 archetype"
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
        "--archetype-prefilter",
        f"{config_dir}/archetypes/prefilter.yaml",
        *common_train_args,
    ]

    rc, out = run_step("Gate Train", gate_train_args, log, dry_run=dry_run)
    gate_dir = find_output_dir(out, strategy) or prepare_dir

    # ── Early termination: if Gate Train failed, downstream steps are useless ──
    gate_pred = Path(f"{gate_dir}/predictions.parquet")
    if rc != 0 or (not dry_run and not gate_pred.exists()):
        print(
            f"\n\u274c Gate Train 失败或未产出 predictions.parquet"
            f" (rc={rc}, exists={gate_pred.exists() if not dry_run else 'N/A'})"
        )
        print(
            "   可能原因: prefilter 过滤过严 (样本量不足), "
            "或训练参数错误. 请检查上方日志."
        )
        if not dry_run:
            return {"error": "gate_train_failed", "gate_dir": gate_dir}

    # Gate apply (用 gate_draft 作为中间件)
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

    # Gate optimize (--promote, 在 gate 应用后的数据上做 plateau 验证)
    gate_optimize_cmd = [
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
    ]
    # 从 kpi_gates 注入 gate 约束
    if gate_gates.get("min_combined_pass_rate"):
        gate_optimize_cmd += [
            "--min-combined-pass-rate",
            str(gate_gates["min_combined_pass_rate"]),
        ]
    run_step(
        "Gate Optimize",
        gate_optimize_cmd,
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
    # ⚠️ Evidence Train 不使用 prefilter (见 BPC pipeline 文档):
    #   - 全量训练 → 更多候选 → 更好的概率校准
    #   - Evidence Optimize 在 logs_gated.parquet（已过 gate）上做 plateau 验证
    #   - 因此不影响生产一致性：只在 gated 人群有效的候选才能 promote
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

    rc, out = run_step("Evidence Train", evidence_train_args, log, dry_run=dry_run)
    evidence_dir = find_output_dir(out, strategy) or gate_dir

    # ── Early termination: if Evidence Train failed ──
    ev_pred = Path(f"{evidence_dir}/predictions.parquet")
    if rc != 0 or (not dry_run and not ev_pred.exists()):
        print(
            f"\n\u274c Evidence Train 失败或未产出 predictions.parquet"
            f" (rc={rc}, exists={ev_pred.exists() if not dry_run else 'N/A'})"
        )
        print("   可能原因: prefilter 过滤过严, 或 Gate 配置问题. " "请检查上方日志.")
        if not dry_run:
            return {
                "error": "evidence_train_failed",
                "gate_dir": gate_dir,
                "evidence_dir": evidence_dir,
            }

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
    # 必须用 logs_gated.parquet：Entry Filter 的阈值必须在 gate 过滤后的
    # 分布上优化，否则会和 gate 产生 distribution mismatch
    run_step(
        "Entry Filter Optimize",
        [
            "python",
            "scripts/optimize_entry_filter_plateau.py",
            "--logs",
            f"{evidence_dir}/logs_gated.parquet",
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
    # Execution 小样本保护: gate 筛选后交易数不足时跳过 grid search,
    # 保留现有 execution.yaml 避免在噪声数据上 "优化" 出极端参数.
    exec_min_trades = execution_gates.get("min_trades", 0)
    skip_execution_opt = False
    if exec_min_trades > 0 and not dry_run:
        gated_path = Path(f"{evidence_dir}/logs_gated.parquet")
        if gated_path.exists():
            try:
                import pandas as _pd

                _gated = _pd.read_parquet(gated_path)
                if "gate_decision" in _gated.columns:
                    _n_allow = int((_gated["gate_decision"] == "allow").sum())
                elif "gate_passed" in _gated.columns:
                    _n_allow = int(_gated["gate_passed"].sum())
                else:
                    _n_allow = len(_gated)
                if _n_allow < exec_min_trades:
                    skip_execution_opt = True
                    print(
                        f"\n  ⏭️  Step 8 SKIP: gate allows {_n_allow} trades "
                        f"< min_trades={exec_min_trades} → 保留现有 execution.yaml"
                    )
            except Exception as exc:
                print(f"\n  ⚠️  Execution min_trades check failed: {exc}")

    if not skip_execution_opt:
        exec_cmd = [
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
        ]
        if use_1min:
            exec_cmd.extend(["--use-1min", "--data-path", data_path])
        run_step(
            "Execution Optimize",
            exec_cmd,
            log,
            dry_run=dry_run,
        )

    # ── Step 9: Backtest ──
    # 必须用 logs_gated.parquet：回测输入必须和实盘一致（经过 gate 过滤），
    # 否则报告的 Sharpe/trades 虚高（gate 会 veto 大量信号）
    # 必须传 --strategies-root: 确保读取本次实验的 entry_filters.yaml 而不是旧的
    experiment_map_path = f"{run_dir}/trading_map_{strategy}.html"
    bt_cmd = [
        "python",
        "scripts/backtest_execution_layer.py",
        "--logs",
        f"{evidence_dir}/logs_gated.parquet",
        "--strategy",
        strategy,
        "--strategies-root",
        strategies_root,
        "--output",
        experiment_map_path,
    ]
    if use_1min:
        bt_cmd.extend(["--use-1min", "--data-path", data_path])
    rc, bt_out = run_step(
        "Backtest",
        bt_cmd,
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


def _patch_report_pcm(report_path: Path, pcm_result: Dict[str, Any]):
    """将 PCM 联合回测结果追加到已保存的 report.json."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["pcm_joint"] = {
            "pcm_decision": pcm_result.get("pcm_decision"),
            "sharpe_daily": pcm_result.get("sharpe_daily"),
            "conflict_rate": pcm_result.get("conflict_rate"),
            "strategies_count": pcm_result.get("strategies_count"),
            "strategies": pcm_result.get("strategies", []),
            "total_trades": pcm_result.get("total_trades"),
        }
        report_path.write_text(
            json.dumps(report, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def _parse_pcm_stdout(output: str) -> Dict[str, Any]:
    """从 PCM 联合回测 stdout 提取指标."""
    result: Dict[str, Any] = {}

    m = re.search(r"Trades:\s*(\d+)", output)
    if m:
        result["total_trades"] = int(m.group(1))

    m = re.search(r"Sharpe \(daily.*?\):\s*([\-\d.]+)", output)
    if m:
        result["sharpe_daily"] = float(m.group(1))

    m = re.search(r"Conflict rate:\s*([\d.]+)%", output)
    if m:
        result["conflict_rate"] = float(m.group(1)) / 100

    m = re.search(r"Mean R:\s*([\-\d.]+)", output)
    if m:
        result["mean_r"] = float(m.group(1))

    m = re.search(r"Win Rate:\s*([\d.]+)%", output)
    if m:
        result["win_rate"] = float(m.group(1)) / 100

    return result


def _run_pcm_joint_backtest(
    results_summary: List[Dict[str, Any]],
    history_dir: Path,
    timestamp: str,
    *,
    dry_run: bool = False,
    use_1min: bool = False,
    live_root: str = "live/highcap",
    data_path: str = "data/parquet_data",
) -> Optional[Dict[str, Any]]:
    """Step 9.5: 全策略完成后, 执行 PCM 联合回测.

    Returns dict with pcm_decision, sharpe_daily, conflict_rate, etc.
    Returns None if <2 strategies have predictions.
    """
    # 收集有 gated logs 的策略（必须用 gate 过滤后的数据，和单策略回测一致）
    pcm_specs = []
    for r in results_summary:
        ev_dir = r.get("evidence_dir")
        if not ev_dir:
            continue
        gated_path = Path(ev_dir) / "logs_gated.parquet"
        if gated_path.exists() or dry_run:
            pcm_specs.append((r["strategy"], str(gated_path)))

    if len(pcm_specs) < 2:
        if len(results_summary) >= 2:
            print(f"\n{'='*70}")
            print("[Step 9.5] PCM 联合回测: ⏭️  SKIP")
            print(f"   找到 {len(pcm_specs)} 个策略有 logs_gated (需 ≥2)")
            print(f"{'='*70}")
        return None

    # 构建 --pcm 参数
    pcm_args = [f"{name}:{path}" for name, path in pcm_specs]
    strategy_names = [name for name, _ in pcm_specs]

    # PCM 交易地图输出路径 (保存到第一个策略的实验目录)
    first_strat = strategy_names[0]
    # 多 seed 模式下 run_dir_name 含 _s{seed} 后缀，必须用实际目录名
    first_run_dir_name = next(
        (
            r.get("run_dir_name", timestamp)
            for r in results_summary
            if r["strategy"] == first_strat
        ),
        timestamp,
    )
    pcm_map_path = str(
        history_dir / first_strat / first_run_dir_name / "pcm_trading_map.html"
    )

    # 日志文件
    pcm_log = history_dir / first_strat / first_run_dir_name / "pcm_joint.log"

    cmd = (
        [
            "python",
            "scripts/backtest_execution_layer.py",
            "--pcm",
        ]
        + pcm_args
        + [
            "--output",
            pcm_map_path,
        ]
    )
    if use_1min:
        cmd.extend(["--use-1min", "--data-path", data_path])

    rc, pcm_out = run_step(
        "PCM Joint Backtest (Step 9.5)",
        cmd,
        pcm_log,
        dry_run=dry_run,
    )

    if dry_run:
        return {
            "pcm_decision": "DRY_RUN",
            "strategies": strategy_names,
            "strategies_count": len(strategy_names),
        }

    # 解析输出
    metrics = _parse_pcm_stdout(pcm_out)
    sharpe_daily = metrics.get("sharpe_daily", 0)
    conflict_rate = metrics.get("conflict_rate", 0)
    total_trades = metrics.get("total_trades", 0)

    # PCM 决策逻辑
    pcm_decision = "PASS"
    pcm_reasons = []

    if rc != 0:
        pcm_decision = "ERROR"
        pcm_reasons.append(f"backtest exit code={rc}")
    elif total_trades < 10:
        pcm_decision = "ERROR"
        pcm_reasons.append(f"trades={total_trades} < 10")
    else:
        if conflict_rate > 0.15:
            pcm_decision = "ALERT"
            pcm_reasons.append(f"conflict_rate={conflict_rate:.2%} > 15%")
        if sharpe_daily < 1.0:
            if pcm_decision != "ALERT":
                pcm_decision = "ALERT"
            pcm_reasons.append(f"sharpe_daily={sharpe_daily:.2f} < 1.0")

    # 打印决策
    pcm_emoji = {"PASS": "\u2705", "ALERT": "\u26a0\ufe0f", "ERROR": "\u274c"}.get(
        pcm_decision, "\u2753"
    )
    print(f"\n   {pcm_emoji} PCM 决策: {pcm_decision}")
    for reason in pcm_reasons:
        print(f"      → {reason}")
    if pcm_decision == "PASS":
        print(
            f"      → sharpe_daily={sharpe_daily:.2f}, conflict_rate={conflict_rate:.2%}"
        )
    print(f"   📄 交易地图: {pcm_map_path}")

    return {
        "pcm_decision": pcm_decision,
        "pcm_reasons": pcm_reasons,
        "sharpe_daily": sharpe_daily,
        "conflict_rate": conflict_rate,
        "total_trades": total_trades,
        "mean_r": metrics.get("mean_r"),
        "win_rate": metrics.get("win_rate"),
        "strategies": strategy_names,
        "strategies_count": len(strategy_names),
        "trading_map": pcm_map_path,
    }


# ====================================================================
# Multi-seed search helpers
# ====================================================================


def _extract_gate_rules(run_dir: Path, strategy: str) -> List[str]:
    """从 seed trial 的 gate.yaml 提取 hard_gate 特征名."""
    arch_dir = run_dir / "strategies" / strategy / "archetypes"
    gate_path = arch_dir / "gate.yaml"
    if not gate_path.exists():
        return []
    gt = yaml.safe_load(gate_path.read_text(encoding="utf-8")) or {}
    rules = []
    for r in gt.get("hard_gates", []):
        if r.get("frozen"):
            continue  # 跳过 frozen (prefilter 注入的)
        feat = r.get("feature", r.get("id", "?"))
        rules.append(feat)
    return rules


def _select_best_seed(
    seed_trials: List[dict],
    min_trades: int = 0,
    selection: str = "best_sharpe",
) -> dict:
    """从多 seed 结果中选最佳.

    seed_trials: [{seed, run_dir, result, metrics, gate_rules}, ...]
    Returns the best trial dict.
    """
    # 筛选: 有 backtest_metrics 且无 error
    valid = [
        t
        for t in seed_trials
        if "error" not in t["result"]
        and t["metrics"].get("total_trades", 0) >= max(min_trades, 1)
    ]
    if not valid:
        # 退而求其次: 任何有 metrics 的都行
        valid = [t for t in seed_trials if "error" not in t["result"]]
    if not valid:
        # 全部失败, 返回第一个
        return seed_trials[0]

    # 排序: sharpe_per_trade 降序
    valid.sort(
        key=lambda t: t["metrics"].get("sharpe_per_trade", -999),
        reverse=True,
    )
    return valid[0]


def _print_seed_diagnostics(
    strategy: str,
    seed_trials: List[dict],
    best_trial: dict,
) -> None:
    """打印多 seed 搜索诊断表."""
    print(f"\n{'─'*60}")
    print(f"🔍 {strategy.upper()} Seed 搜索结果 ({len(seed_trials)} seeds):")
    print(f"{'─'*60}")
    print(f"  {'seed':>6s}  {'Sharpe':>8s}  {'trades':>7s}  {'win%':>6s}  gate_rules")
    print(f"  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*20}")
    for t in seed_trials:
        m = t["metrics"]
        sharpe = m.get("sharpe_per_trade", 0)
        trades = m.get("total_trades", 0)
        win = m.get("win_rate", 0)
        rules = t.get("gate_rules", [])
        marker = " 🏆" if t["seed"] == best_trial["seed"] else ""
        err = " ❌" if "error" in t["result"] else ""
        print(
            f"  {t['seed']:>6d}  {sharpe:>8.4f}  {trades:>7.0f}  "
            f"{win*100:>5.1f}%  {', '.join(rules) if rules else '(error)'}{marker}{err}"
        )

    # 稳定性诊断: 统计每个特征被选中的次数
    from collections import Counter

    feat_counts = Counter()
    for t in seed_trials:
        if "error" not in t["result"]:
            for f in t.get("gate_rules", []):
                feat_counts[f] += 1
    n_valid = sum(1 for t in seed_trials if "error" not in t["result"])
    if feat_counts and n_valid > 1:
        print(f"\n  📊 Gate 特征稳定性:")
        for feat, cnt in feat_counts.most_common():
            pct = cnt / n_valid * 100
            bar = "█" * int(pct / 10)
            print(f"     {feat:<30s} {cnt}/{n_valid} ({pct:.0f}%) {bar}")
    print()


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
    p.add_argument(
        "--use-1min",
        action="store_true",
        help="使用 1min bar 精细模拟止损/移动止损 (匹配实盘精度)",
    )
    p.add_argument(
        "--live-root",
        default="live/highcap",
        help="1min bar 数据根目录 (default: live/highcap)",
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
    # ── 多 Seed 配置 ──
    training_cfg = cfg.get("training", {})
    seeds = training_cfg.get("seeds", [42])
    seed_selection = training_cfg.get("seed_selection", "best_sharpe")
    print(f"   Seeds:       {seeds}")
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

        print(f"\n{'#'*70}")
        print(f"# 策略: {strategy.upper()}")
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

        # ── 多 Seed 搜索 ──
        scfg = cfg["strategies"][strategy]
        exec_min_trades = (
            scfg.get("kpi_gates", {}).get("backtest", {}).get("min_trades", 0)
        )
        multi_seed = len(seeds) > 1
        seed_trials = []

        for seed_idx, seed in enumerate(seeds):
            if multi_seed:
                seed_run_dir = history_dir / strategy / f"{timestamp}_s{seed}"
                print(f"\n  🌱 Seed {seed} ({seed_idx+1}/{len(seeds)})")
            else:
                seed_run_dir = history_dir / strategy / timestamp
            seed_run_dir.mkdir(parents=True, exist_ok=True)

            result = run_strategy_pipeline(
                strategy,
                cfg,
                end_date=end_date,
                holdout_start=holdout_start,
                start_date=start_date,
                symbols=symbols,
                data_path=data_path,
                run_dir=seed_run_dir,
                seed=seed,
                dry_run=args.dry_run,
                use_1min=args.use_1min,
                live_root=args.live_root,
            )

            metrics = result.get("backtest_metrics", {})
            gate_rules = (
                _extract_gate_rules(seed_run_dir, strategy) if not args.dry_run else []
            )
            seed_trials.append(
                {
                    "seed": seed,
                    "run_dir": seed_run_dir,
                    "result": result,
                    "metrics": metrics,
                    "gate_rules": gate_rules,
                }
            )

        # ── 选优 ──
        if multi_seed:
            best = _select_best_seed(
                seed_trials, min_trades=exec_min_trades, selection=seed_selection
            )
            _print_seed_diagnostics(strategy, seed_trials, best)
        else:
            best = seed_trials[0]

        run_dir = best["run_dir"]
        pipeline_result = best["result"]

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

        # ── 分层汇总表 (每层规则/阈值 + 与上次对比) ──
        cur_arch = _find_arch_dir(run_dir, strategy)
        prev_arch = None
        prev_bt = {}
        if prev and not args.dry_run:
            prev_ts = prev.get("timestamp", "")
            prev_dir = history_dir / strategy / prev_ts
            prev_arch = _find_arch_dir(prev_dir, strategy)
            prev_bt = prev.get("backtest_metrics", {})
        if cur_arch:
            print_layer_summary(
                strategy,
                run_dir.name,
                cur_arch,
                bt,
                prev_arch_dir=prev_arch,
                prev_metrics=prev_bt if prev_bt else None,
            )

        print(f"\n   {emoji} 决策: {decision}")
        for r in comparison.get("reasons", []):
            print(f"      → {r}")
        print(f"   📁 Report: {report_path}")
        print(f"   📦 实验配置: {run_dir}/strategies/{strategy}/archetypes/")

        # ── 漂移报告 (当存在上次实验时自动输出) ──
        drift_levels = None
        if prev and not args.dry_run and prev_arch and cur_arch:
            drift_levels = _print_drift_report(
                strategy,
                prev.get("timestamp", ""),
                run_dir.name,
                prev_arch,
                cur_arch,
                prev_bt,
                bt,
            )

        # ── Deploy 门禁检查 ──
        deploy_cfg = cfg.get("deploy_gate", {})
        # per-strategy kpi_gates.deploy 覆盖全局默认
        deploy_kpi = cfg["strategies"][strategy].get("kpi_gates", {}).get("deploy", {})
        if deploy_kpi.get("min_trades") is not None:
            deploy_cfg = {**deploy_cfg, "min_trades": deploy_kpi["min_trades"]}
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
                f"      python scripts/auto_research_pipeline.py --strategy {strategy} --adopt {run_dir.name}"
            )

        results_summary.append(
            {
                "strategy": strategy,
                "decision": decision,
                "sharpe": bt.get("sharpe_per_trade"),
                "trades": bt.get("total_trades"),
                "evidence_dir": pipeline_result.get("evidence_dir"),
                "run_dir_name": run_dir.name,
                "seed": best["seed"] if multi_seed else seeds[0],
            }
        )

    # ── Step 9.5: PCM 联合回测 (仅 --all 且 ≥2 策略成功时) ──
    pcm_result = None
    if args.all and not args.compare_only:
        pcm_result = _run_pcm_joint_backtest(
            results_summary,
            history_dir,
            timestamp,
            dry_run=args.dry_run,
            use_1min=args.use_1min,
            live_root=args.live_root,
            data_path=cfg["data_path"],
        )

    # ── 汇总 ──
    print(f"\n{'='*70}")
    print("📋 汇总")
    print(f"{'='*70}")
    for r in results_summary:
        emoji = {"ADOPT": "✅", "ALERT": "⚠️", "KEEP": "🔄", "ERROR": "❌"}.get(
            r["decision"], "❓"
        )
        seed_str = f" seed={r['seed']}" if r.get("seed") is not None else ""
        print(
            f"   {emoji} {r['strategy']:>6s}: {r['decision']:<8s} sharpe={r.get('sharpe', 'N/A')} trades={r.get('trades', 'N/A')}{seed_str}"
        )
    if pcm_result:
        pcm_emoji = {"PASS": "✅", "ALERT": "⚠️", "ERROR": "❌"}.get(
            pcm_result.get("pcm_decision", "?"), "❓"
        )
        print(
            f"\n   {pcm_emoji}    PCM: {pcm_result.get('pcm_decision', '?'):<8s} "
            f"sharpe_daily={pcm_result.get('sharpe_daily', 'N/A')} "
            f"conflict_rate={pcm_result.get('conflict_rate', 'N/A')} "
            f"strategies={pcm_result.get('strategies_count', 0)}"
        )
        # 保存 pcm_stats.json 到每个策略的实验目录
        for r in results_summary:
            rdn = r.get("run_dir_name", timestamp)
            strat_run = history_dir / r["strategy"] / rdn
            if strat_run.exists():
                _patch_report_pcm(strat_run / "report.json", pcm_result)


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
    """列出指定策略的历史实验。

    Multi-seed 实验组（同一 timestamp, 不同 _s{N} 后缀）只显示胜出的 seed，
    其余 seed 折叠为一行摘要，避免列表污染。
    """
    import re
    import time

    strat_dir = history_dir / strategy
    if not strat_dir.exists():
        print(f"\n📋 {strategy.upper()}: 无历史实验")
        return

    runs = sorted(d for d in strat_dir.iterdir() if d.is_dir())

    # 分组: base_timestamp -> [(run_dir, seed_num)]
    # e.g. 20260226_211920_s1, _s2, _s42 => base=20260226_211920
    seed_re = re.compile(r"^(.+?)_s(\d+)$")
    groups = {}  # base_ts -> [(run_dir, seed_num)]
    standalone = []  # 无 seed 后缀的单独实验
    for run_dir in runs:
        m = seed_re.match(run_dir.name)
        if m:
            base_ts, seed_num = m.group(1), int(m.group(2))
            groups.setdefault(base_ts, []).append((run_dir, seed_num))
        else:
            standalone.append(run_dir)

    # 构建显示列表: (sort_key, lines)
    display_items = []

    for run_dir in standalone:
        lines = _format_experiment_line(run_dir)
        display_items.append((run_dir.name, lines))

    for base_ts, members in groups.items():
        members.sort(key=lambda x: x[1])
        if len(members) == 1:
            # 只有一个 seed，当单独实验显示
            lines = _format_experiment_line(members[0][0])
            display_items.append((base_ts, lines))
            continue

        # 多 seed: 找胜出的 (有 report + 最高 sharpe)
        best_dir, best_seed, best_sharpe = None, None, -999
        n_total = len(members)
        n_no_report = 0
        n_error = 0
        for run_dir, seed_num in members:
            report_file = run_dir / "report.json"
            if not report_file.exists():
                n_no_report += 1
                continue
            try:
                report = json.loads(report_file.read_text(encoding="utf-8"))
                bt = report.get("backtest_metrics", {})
                comp = report.get("comparison", {})
                decision = comp.get("decision", "?")
                sharpe = bt.get("sharpe_per_trade", 0) or 0
                if decision == "ERROR":
                    n_error += 1
                if isinstance(sharpe, (int, float)) and sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_dir = run_dir
                    best_seed = seed_num
            except Exception:
                n_no_report += 1

        if best_dir is not None:
            lines = _format_experiment_line(best_dir)
            seed_note = f"  └─ multi-seed: winner=s{best_seed}/{n_total} seeds"
            if n_no_report > 0:
                seed_note += f", {n_no_report} incomplete"
            if n_error > 0:
                seed_note += f", {n_error} error"
            lines.append(seed_note)
            display_items.append((base_ts, lines))
        else:
            # 所有 seed 都没有有效 report
            first_dir = members[0][0]
            created_time = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(first_dir.stat().st_ctime)
            )
            lines = [
                f"  {base_ts:<22s} {created_time:<15s}  (无有效 report, {n_total} seeds 全部失败)"
            ]
            display_items.append((base_ts, lines))

    display_items.sort(key=lambda x: x[0])
    n_logical = len(display_items)
    print(f"\n📋 {strategy.upper()} 历史实验 ({n_logical} 次):")
    print(f"{'─'*100}")
    print(
        f"  {'时间戳':<22s} {'创建时间':<15s} {'Sharpe':>10s} {'Trades':>8s} {'决策':>8s}  备注"
    )
    print(f"{'─'*100}")
    for _, lines in display_items:
        for line in lines:
            print(line)


def _format_experiment_line(run_dir: Path) -> list:
    """格式化单个实验目录为显示行."""
    import time

    created_time = time.strftime(
        "%Y-%m-%d %H:%M", time.localtime(run_dir.stat().st_ctime)
    )
    report_file = run_dir / "report.json"
    if not report_file.exists():
        return [f"  {run_dir.name:<22s} {created_time:<15s}  (无 report.json)"]

    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
    except Exception:
        return [f"  {run_dir.name:<22s} {created_time:<15s}  (report.json 损坏)"]

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
    sharpe_str = f"{sharpe:.4f}" if isinstance(sharpe, (int, float)) else str(sharpe)
    return [
        f"  {run_dir.name:<22s} {created_time:<15s} {sharpe_str:>10s} {str(trades):>8s} {emoji}{decision:>6s}  {note}"
    ]


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


# ── 分层汇总表 ─────────────────────────────────────────────────────


def _read_yaml_safe(path: Path) -> dict:
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def _extract_layer_info(arch_dir: Path) -> Dict[str, Any]:
    """Extract per-layer summary from archetypes directory."""
    info: Dict[str, Any] = {}

    # Prefilter
    pf = _read_yaml_safe(arch_dir / "prefilter.yaml")
    pf_rules = pf.get("rules", [])
    pf_descs = []
    for r in pf_rules:
        if "any_of" in r:
            subs = [
                f"{s.get('feature', '?')}{s.get('operator', '')}{s.get('value', '')}"
                for s in r["any_of"]
            ]
            pf_descs.append("(" + " OR ".join(subs) + ")")
        else:
            pf_descs.append(
                f"{r.get('feature', '?')} {r.get('operator', '')} {r.get('value', '')}"
            )
    info["prefilter"] = {"count": len(pf_rules), "rules": pf_descs}

    # Direction
    dr = _read_yaml_safe(arch_dir / "direction.yaml")
    dr_rules = dr.get("direction_rules", [])
    dr_descs = [f"{r.get('feature', '?')}(w={r.get('weight', 1)})" for r in dr_rules]
    info["direction"] = {"count": len(dr_rules), "rules": dr_descs}

    # Gate
    gt = _read_yaml_safe(arch_dir / "gate.yaml")
    hg = gt.get("hard_gates", [])
    gr = gt.get("guardrails", [])
    hg_descs = []
    for r in hg:
        rid = r.get("id", "")
        w = r.get("when", {})
        if "all_of" in w:
            # compound gate (e.g. OR prefilter negated)
            parts = []
            for sub in w["all_of"]:
                if isinstance(sub, dict):
                    for sf, sc in sub.items():
                        if isinstance(sc, dict):
                            for sop, sv in sc.items():
                                parts.append(f"{sf} {sop.replace('value_','')} {sv}")
            hg_descs.append(" AND ".join(parts) if parts else rid)
        else:
            for feat, cond in w.items():
                if isinstance(cond, dict):
                    for op, val in cond.items():
                        op_s = op.replace("value_", "")
                        hg_descs.append(f"{feat} {op_s} {val}")
                else:
                    hg_descs.append(f"{feat}: {cond}")
    info["gate"] = {"hard_gates": len(hg), "guardrails": len(gr), "rules": hg_descs}

    # Evidence
    ev = _read_yaml_safe(arch_dir / "evidence.yaml")
    ev_axes = ev.get("evidence_axes", ev.get("evidence", []))
    ev_descs = []
    for r in ev_axes:
        feat = r.get("feature", "?")
        direction = r.get("direction", "")
        qm = r.get("quantile_mapping", {})
        bins = qm.get("bins", [])
        if bins:
            ev_descs.append(f"{feat}({direction}, bins={len(bins)})")
        else:
            ev_descs.append(f"{feat}({direction})")
    info["evidence"] = {"count": len(ev_axes), "rules": ev_descs}

    # Entry Filters
    ef = _read_yaml_safe(arch_dir / "entry_filters.yaml")
    ef_filters = ef.get("filters", [])
    ef_descs = [f.get("id", f.get("name", "?")) for f in ef_filters]
    info["entry_filters"] = {"count": len(ef_filters), "rules": ef_descs}

    # Execution
    ex = _read_yaml_safe(arch_dir / "execution.yaml")
    ex_summary = {}
    if ex.get("stop_loss"):
        sl = ex["stop_loss"]
        init_r = sl.get("initial_r", sl.get("r_multiple", sl.get("atr_multiple", "?")))
        trail = sl.get("trailing", {})
        act_r = trail.get("activation_r", "")
        trail_r = trail.get("trail_r", "")
        sl_s = f"{init_r}R"
        if act_r:
            sl_s += f"(act={act_r},trail={trail_r})"
        ex_summary["SL"] = sl_s
    if ex.get("take_profit") and ex["take_profit"].get("enabled", True):
        tp = ex["take_profit"]
        tp_val = tp.get("r_multiple", tp.get("target_r", tp.get("atr_multiple", "?")))
        ex_summary["TP"] = f"{tp_val}R"
    if ex.get("holding"):
        h = ex["holding"]
        mb = h.get("max_holding_bars", h.get("max_bars", None))
        ts = h.get("time_stop_bars", None)
        if ts:
            ex_summary["time_stop"] = f"{ts}bars"
        elif mb:
            ex_summary["max_bars"] = mb
    if ex.get("tiers", {}).get("enabled"):
        ex_summary["tiers"] = len(ex["tiers"].get("levels", []))
    # fallback: generic params
    ex_params = ex.get("params", ex.get("execution_params", {}))
    if ex_params:
        ex_summary.update({k: v for k, v in list(ex_params.items())[:3]})
    info["execution"] = ex_summary

    return info


def _fmt_rules(rules: list, max_show: int = 3) -> str:
    if not rules:
        return ""
    shown = rules[:max_show]
    rest = len(rules) - max_show
    s = ", ".join(str(r) for r in shown)
    if rest > 0:
        s += f" (+{rest}更多)"
    return s


def _delta_str(cur: int, prev: int) -> str:
    if prev == cur:
        return "—"
    diff = cur - prev
    return f"{diff:+d}" if diff != 0 else "—"


def print_layer_summary(
    strategy: str,
    timestamp: str,
    arch_dir: Path,
    backtest_metrics: Dict[str, Any],
    prev_arch_dir: Optional[Path] = None,
    prev_metrics: Optional[Dict[str, Any]] = None,
):
    """Pipeline 结束时打印分层汇总表."""
    cur = _extract_layer_info(arch_dir)
    prev = _extract_layer_info(prev_arch_dir) if prev_arch_dir else None
    w = 74
    sep = "─" * w

    print(f"\n{'═' * w}")
    print(f"  {strategy.upper()} 分层配置汇总  ({timestamp})")
    print(f"{'═' * w}")

    # ── Prefilter ──
    pf = cur["prefilter"]
    line = f"  L2 Prefilter     {pf['count']} rule(s)"
    if prev:
        pp = prev["prefilter"]
        line += f"  ← prev {pp['count']}  {_delta_str(pf['count'], pp['count'])}"
    print(line)
    if pf["rules"]:
        print(f"                   {_fmt_rules(pf['rules'])}")

    # ── Direction ──
    dr = cur["direction"]
    line = f"  L3 Direction     {dr['count']} feature(s)"
    if prev:
        pd_ = prev["direction"]
        line += f"  ← prev {pd_['count']}  {_delta_str(dr['count'], pd_['count'])}"
    print(line)
    if dr["rules"]:
        print(f"                   {_fmt_rules(dr['rules'])}")

    # ── Gate ──
    gt = cur["gate"]
    line = f"  L4 Gate          {gt['hard_gates']} hard_gate(s), {gt['guardrails']} guardrail(s)"
    if prev:
        pg = prev["gate"]
        line += f"  ← prev {pg['hard_gates']}+{pg['guardrails']}  {_delta_str(gt['hard_gates'], pg['hard_gates'])}"
    print(line)
    if gt["rules"]:
        print(f"                   {_fmt_rules(gt['rules'])}")

    # ── Evidence ──
    ev = cur["evidence"]
    line = f"  L5 Evidence      {ev['count']} axis/axes"
    if prev:
        pe = prev["evidence"]
        line += f"  ← prev {pe['count']}  {_delta_str(ev['count'], pe['count'])}"
    print(line)
    if ev["rules"]:
        print(f"                   {_fmt_rules(ev['rules'], max_show=5)}")

    # ── Entry Filters ──
    ef = cur["entry_filters"]
    line = f"  L6 Entry Filter  {ef['count']} filter(s)"
    if prev:
        pef = prev["entry_filters"]
        line += f"  ← prev {pef['count']}  {_delta_str(ef['count'], pef['count'])}"
    print(line)
    if ef["rules"]:
        print(f"                   {_fmt_rules(ef['rules'])}")

    # ── Execution ──
    ex = cur.get("execution", {})
    ex_parts = [f"{k}={v}" for k, v in list(ex.items())[:5]] if ex else ["(默认)"]
    print(f"  L7 Execution     {', '.join(ex_parts)}")

    # ── Backtest ──
    print(f"  {sep}")
    bt = backtest_metrics
    sharpe_pt = bt.get("sharpe_per_trade", "N/A")
    sharpe_d = bt.get("sharpe_daily", "")
    trades = bt.get("total_trades", "N/A")
    winr = bt.get("win_rate", "N/A")
    mean_r = bt.get("mean_r", "N/A")
    sharpe_s = (
        f"{sharpe_pt:.4f}" if isinstance(sharpe_pt, (int, float)) else str(sharpe_pt)
    )
    daily_s = f" (daily {sharpe_d:.2f})" if isinstance(sharpe_d, (int, float)) else ""
    winr_s = f"{winr:.1%}" if isinstance(winr, (int, float)) else str(winr)
    mean_r_s = f"{mean_r:.4f}" if isinstance(mean_r, (int, float)) else str(mean_r)
    line = f"  Backtest         Sharpe={sharpe_s}{daily_s}  Trades={trades}  Win={winr_s}  MeanR={mean_r_s}"
    print(line)
    if prev_metrics:
        p_sharpe = prev_metrics.get("sharpe_per_trade")
        p_trades = prev_metrics.get("total_trades")
        p_winr = prev_metrics.get("win_rate")
        parts = []
        if isinstance(p_sharpe, (int, float)):
            parts.append(f"Sharpe={p_sharpe:.4f}")
            if isinstance(sharpe_pt, (int, float)) and p_sharpe != 0:
                pct = (sharpe_pt - p_sharpe) / abs(p_sharpe) * 100
                parts.append(f"Δ={pct:+.1f}%")
        if p_trades is not None:
            parts.append(f"Trades={p_trades}")
        if isinstance(p_winr, (int, float)):
            parts.append(f"Win={p_winr:.1%}")
        if parts:
            print(f"     prev:         {', '.join(parts)}")

    print(f"{'═' * w}")


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
