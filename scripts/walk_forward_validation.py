#!/usr/bin/env python3
"""
Walk-Forward Validation: 量化 Sharpe 过拟合程度

原理 (Anchored Walk-Forward):
  将数据按时间分成 N 个不重叠的 fold (每 fold = holdout_months 个月):

  Fold 1: train → 2024-06, holdout 2024-07~2025-01  → IS (calibration)
  Fold 2: train → 2025-01, holdout 2025-01~2025-07  → IS + OOS 对比
  Fold 3: train → 2025-07, holdout 2025-07~2026-01  → IS + OOS 对比

  IS Sharpe = 每个 fold 自己优化后的回测 Sharpe (当前管线输出)
  OOS Sharpe = 用前一个 fold 的冻结配置, 在当前 fold 的模型预测上回测

  IS/OOS 比率 = Sharpe 衰减比, 接近 1.0 = 参数稳健, <0.5 = 严重过拟合

用法:
  # Phase 1: 运行各 fold (耗时, 支持 resume)
  python scripts/walk_forward_validation.py --strategy me --folds 3

  # Phase 1: dry-run 预览 fold 配置
  python scripts/walk_forward_validation.py --strategy me --folds 3 --dry-run

  # Phase 2: 只做 OOS 对比 (已有 fold 结果时)
  python scripts/walk_forward_validation.py --strategy me --folds 3 --oos-only

  # 单 seed (更快, 用于快速验证)
  python scripts/walk_forward_validation.py --strategy me --folds 3 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.auto_research_pipeline import (
    DEFAULT_CONFIG,
    compute_holdout_start,
    detect_latest_data_date,
    load_pipeline_config,
    parse_backtest_stdout,
    run_strategy_pipeline,
)
from scripts.backtest_execution_layer import (
    compute_sharpe,
    load_execution_config,
    simulate_rr_execution,
    _estimate_span_years,
)

WF_ROOT = PROJECT_ROOT / "results" / "walk_forward"


# ====================================================================
# Date helpers
# ====================================================================


def generate_fold_dates(
    latest_end_date: str,
    holdout_months: int,
    n_folds: int,
    start_date: str,
) -> List[Dict[str, str]]:
    """Generate non-overlapping fold windows, working backwards from latest_end_date.

    Each fold's holdout = holdout_months.  Folds don't overlap.
    """
    from dateutil.relativedelta import relativedelta

    end = datetime.strptime(latest_end_date, "%Y-%m-%d")
    folds = []
    for i in range(n_folds - 1, -1, -1):
        fold_end = end - relativedelta(months=holdout_months * i)
        fold_holdout_start = fold_end - relativedelta(months=holdout_months)
        # 确保 holdout_start > start_date (否则训练数据不足)
        if fold_holdout_start <= datetime.strptime(start_date, "%Y-%m-%d"):
            continue
        folds.append(
            {
                "fold_id": len(folds) + 1,
                "end_date": fold_end.strftime("%Y-%m-%d"),
                "holdout_start": fold_holdout_start.strftime("%Y-%m-%d"),
            }
        )
    return folds


# ====================================================================
# Phase 1: Run folds
# ====================================================================


def run_fold(
    strategy: str,
    cfg: dict,
    fold: Dict[str, str],
    wf_dir: Path,
    *,
    seed: int = 42,
    dry_run: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run a single WF fold (full pipeline with specific end_date)."""
    fold_id = fold["fold_id"]
    end_date = fold["end_date"]
    holdout_start = fold["holdout_start"]
    start_date = cfg["dates"]["start_date"]
    symbols = cfg["symbols"]
    data_path = cfg.get("data_path", "data/parquet_data")

    fold_dir = wf_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 70}")
    print(
        f"📊 Fold {fold_id}: train → {holdout_start}, holdout {holdout_start} → {end_date}"
    )
    print(f"   seed={seed}, output: {fold_dir}")
    print(f"{'═' * 70}")

    if dry_run:
        print("   [DRY-RUN] 跳过实际运行")
        return None

    t0 = time.time()
    result = run_strategy_pipeline(
        strategy,
        cfg,
        end_date=end_date,
        holdout_start=holdout_start,
        start_date=start_date,
        symbols=symbols,
        data_path=data_path,
        run_dir=fold_dir,
        seed=seed,
        dry_run=False,
    )
    elapsed = time.time() - t0

    # 保存 fold metadata
    fold_meta = {
        "fold_id": fold_id,
        "end_date": end_date,
        "holdout_start": holdout_start,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 1),
        "backtest_metrics": result.get("backtest_metrics", {}),
        "evidence_dir": result.get("evidence_dir", ""),
        "exp_config_dir": result.get("exp_config_dir", ""),
    }
    (fold_dir / "fold_meta.json").write_text(
        json.dumps(fold_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n   ✅ Fold {fold_id} 完成 ({elapsed:.0f}s)")
    return fold_meta


# ====================================================================
# Phase 2: OOS backtest with frozen configs
# ====================================================================


def run_oos_backtest(
    strategy: str,
    current_fold_dir: Path,
    prev_fold_dir: Path,
    wf_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Apply previous fold's frozen configs to current fold's predictions → true OOS.

    Steps:
    1. Read predictions.parquet from current fold (model trained up to current end_date)
    2. Apply prev fold's gate.yaml → logs_gated_oos.parquet
    3. Apply prev fold's entry_filters.yaml + execution.yaml → backtest
    """
    # ── Load fold metadata ──
    cur_meta_path = current_fold_dir / "fold_meta.json"
    prev_meta_path = prev_fold_dir / "fold_meta.json"
    if not cur_meta_path.exists() or not prev_meta_path.exists():
        print(f"   ⚠️  缺少 fold_meta.json, 跳过 OOS")
        return None

    cur_meta = json.loads(cur_meta_path.read_text())
    prev_meta = json.loads(prev_meta_path.read_text())

    cur_fold_id = cur_meta["fold_id"]
    prev_fold_id = prev_meta["fold_id"]

    print(f"\n{'─' * 70}")
    print(
        f"🔒 OOS Backtest: Fold {cur_fold_id} predictions × Fold {prev_fold_id} frozen configs"
    )
    print(f"{'─' * 70}")

    # ── Locate files ──
    cur_evidence_dir = cur_meta.get("evidence_dir", "")
    cur_predictions = Path(cur_evidence_dir) / "predictions.parquet"
    if not cur_predictions.exists():
        # fallback: look in fold dir
        for p in current_fold_dir.rglob("predictions.parquet"):
            cur_predictions = p
            break
    if not cur_predictions.exists():
        print(f"   ❌ 找不到 predictions.parquet: {cur_predictions}")
        return None

    prev_config_dir = prev_meta.get("exp_config_dir", "")
    prev_gate_yaml = Path(prev_config_dir) / "archetypes" / "gate.yaml"
    if not prev_gate_yaml.exists():
        print(f"   ❌ 找不到 frozen gate: {prev_gate_yaml}")
        return None

    # ── Step 1: Apply frozen gate ──
    oos_dir = wf_dir / f"oos_fold_{cur_fold_id}"
    oos_dir.mkdir(parents=True, exist_ok=True)
    oos_gated = oos_dir / "logs_gated_oos.parquet"

    print(f"   1️⃣  Apply frozen gate: {prev_gate_yaml.name}")
    rc = subprocess.run(
        [
            "mlbot",
            "gate",
            "apply-archetype",
            "--logs",
            str(cur_predictions),
            "--out",
            str(oos_gated),
            "--gate-path",
            str(prev_gate_yaml),
            "--strategy",
            strategy,
        ],
        capture_output=True,
        text=True,
    )
    if rc.returncode != 0 or not oos_gated.exists():
        print(f"   ❌ Gate apply 失败: {rc.stderr[:200]}")
        return None

    # ── Step 2: Load gated data + apply frozen entry filter + backtest ──
    print(f"   2️⃣  Apply frozen entry_filter + execution → backtest")
    prev_strategies_root = str(Path(prev_config_dir).parent)

    # Run backtest using backtest_execution_layer.py with frozen configs
    bt_cmd = [
        "python",
        "scripts/backtest_execution_layer.py",
        "--logs",
        str(oos_gated),
        "--strategy",
        strategy,
        "--strategies-root",
        prev_strategies_root,
        "--output",
        str(oos_dir / f"trading_map_oos_{strategy}.html"),
    ]
    bt_result = subprocess.run(bt_cmd, capture_output=True, text=True)
    if bt_result.returncode != 0:
        print(f"   ❌ Backtest 失败: {bt_result.stderr[:300]}")
        return None

    oos_metrics = parse_backtest_stdout(bt_result.stdout)
    print(
        f"   📊 OOS: Sharpe={oos_metrics.get('sharpe_per_trade', 0):.4f}, "
        f"Trades={oos_metrics.get('total_trades', 0)}, "
        f"Win={oos_metrics.get('win_rate', 0):.1%}"
    )

    # Save OOS results
    oos_result = {
        "fold_id": cur_fold_id,
        "frozen_from_fold": prev_fold_id,
        "oos_metrics": oos_metrics,
        "is_metrics": cur_meta.get("backtest_metrics", {}),
    }
    (oos_dir / "oos_result.json").write_text(
        json.dumps(oos_result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return oos_result


# ====================================================================
# Summary
# ====================================================================


def print_summary(
    strategy: str,
    folds: List[Dict],
    fold_results: List[Dict],
    oos_results: List[Dict],
):
    """Print WF summary table."""
    print(f"\n{'=' * 78}")
    print(f"📊 Walk-Forward Validation Summary: {strategy.upper()}")
    print(f"{'=' * 78}")

    # ── Fold results (IS) ──
    print(f"\n{'─' * 78}")
    print(
        f"{'Fold':>5s}  {'Holdout':>20s}  {'Sharpe_pt':>10s}  {'Trades':>7s}  {'WinRate':>8s}  {'MeanR':>8s}"
    )
    print(f"{'─' * 78}")

    is_sharpes = []
    for fr in fold_results:
        if fr is None:
            continue
        m = fr.get("backtest_metrics", {})
        sharpe = m.get("sharpe_per_trade", 0)
        trades = m.get("total_trades", 0)
        win = m.get("win_rate", 0)
        mean_r = m.get("mean_r", 0)
        fid = fr["fold_id"]
        fold_info = next((f for f in folds if f["fold_id"] == fid), {})
        h_range = (
            f"{fold_info.get('holdout_start', '?')}→{fold_info.get('end_date', '?')}"
        )
        print(
            f"  {fid:>3d}  {h_range:>20s}  {sharpe:>10.4f}  {trades:>7d}  {win:>7.1%}  {mean_r:>8.3f}"
        )
        is_sharpes.append(sharpe)

    if is_sharpes:
        print(f"{'─' * 78}")
        print(f"  {'IS Mean':>24s}  {np.mean(is_sharpes):>10.4f}")
        print(f"  {'IS Std':>24s}  {np.std(is_sharpes):>10.4f}")

    # ── OOS results ──
    if oos_results:
        print(f"\n{'─' * 78}")
        print(
            f"{'Fold':>5s}  {'Frozen':>7s}  {'IS Sharpe':>10s}  {'OOS Sharpe':>11s}  "
            f"{'Decay':>7s}  {'OOS Trades':>11s}"
        )
        print(f"{'─' * 78}")

        oos_sharpes = []
        decay_ratios = []
        for oos in oos_results:
            if oos is None:
                continue
            fid = oos["fold_id"]
            frozen = oos["frozen_from_fold"]
            is_s = oos["is_metrics"].get("sharpe_per_trade", 0)
            oos_s = oos["oos_metrics"].get("sharpe_per_trade", 0)
            oos_t = oos["oos_metrics"].get("total_trades", 0)
            decay = oos_s / is_s if is_s != 0 else float("nan")
            print(
                f"  {fid:>3d}  F{frozen:>5d}  {is_s:>10.4f}  {oos_s:>11.4f}  "
                f"{decay:>6.0%}  {oos_t:>11d}"
            )
            oos_sharpes.append(oos_s)
            if not np.isnan(decay):
                decay_ratios.append(decay)

        if oos_sharpes:
            print(f"{'─' * 78}")
            print(f"  {'OOS Mean':>18s}  {'':>10s}  {np.mean(oos_sharpes):>11.4f}")
            if decay_ratios:
                mean_decay = np.mean(decay_ratios)
                print(f"  {'Avg Decay':>18s}  {'':>10s}  {'':>11s}  {mean_decay:>6.0%}")

                # ── Verdict ──
                print(f"\n{'═' * 78}")
                if mean_decay >= 0.7:
                    verdict = "✅ 参数稳健 (OOS/IS ≥ 70%)"
                elif mean_decay >= 0.5:
                    verdict = "⚠️  中等衰减 (OOS/IS 50-70%), 可接受但需关注"
                elif mean_decay >= 0.3:
                    verdict = "🟡 较大衰减 (OOS/IS 30-50%), 优化层可能过拟合"
                else:
                    verdict = "🔴 严重过拟合 (OOS/IS < 30%), 需简化优化层"
                print(f"  📋 判定: {verdict}")
                print(f"     IS Mean Sharpe:  {np.mean(is_sharpes):.4f}")
                print(f"     OOS Mean Sharpe: {np.mean(oos_sharpes):.4f}")
                print(f"     Decay Ratio:     {mean_decay:.0%}")
                print(f"{'═' * 78}")
    else:
        print(
            f"\n  ℹ️  需要至少 2 个 fold 才能计算 OOS (当前 fold 结果: {len(fold_results)})"
        )


# ====================================================================
# Main
# ====================================================================


def main():
    p = argparse.ArgumentParser(
        description="Walk-Forward Validation: 量化 Sharpe 过拟合程度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--strategy", required=True, help="策略名 (bpc/fer/me)")
    p.add_argument("--folds", type=int, default=3, help="Fold 数量 (默认 3)")
    p.add_argument("--end-date", help="最新 end_date (默认自动检测)")
    p.add_argument(
        "--seed", type=int, default=42, help="训练 seed (默认 42, 单 seed 加速)"
    )
    p.add_argument("--resume", action="store_true", help="跳过已完成的 fold")
    p.add_argument("--dry-run", action="store_true", help="只打印 fold 配置, 不执行")
    p.add_argument(
        "--oos-only",
        action="store_true",
        help="跳过 Phase 1 (fold 运行), 只做 Phase 2 (OOS 对比)",
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="pipeline 配置文件")
    args = p.parse_args()

    cfg = load_pipeline_config(Path(args.config))
    strategy = args.strategy
    holdout_months = cfg["dates"]["holdout_months"]
    start_date = cfg["dates"]["start_date"]
    symbols = cfg["symbols"]
    data_path = cfg.get("data_path", "data/parquet_data")

    # ── Detect end_date ──
    if args.end_date:
        latest_end_date = args.end_date
    else:
        latest_end_date = detect_latest_data_date(data_path, symbols)
        print(f"📅 自动检测最新数据: {latest_end_date}")

    # ── Generate fold dates ──
    folds = generate_fold_dates(latest_end_date, holdout_months, args.folds, start_date)
    if len(folds) < 2:
        print(f"❌ 数据不足: 需要至少 2 个 fold, 但只能生成 {len(folds)} 个")
        print(
            f"   数据范围: {start_date} ~ {latest_end_date}, holdout_months={holdout_months}"
        )
        print(f"   建议: 减少 --folds 或 减小 holdout_months")
        return 1

    # ── Print fold plan ──
    wf_dir = WF_ROOT / strategy
    print(f"\n{'=' * 70}")
    print(f"🔬 Walk-Forward Validation: {strategy.upper()}")
    print(f"{'=' * 70}")
    print(f"   数据范围:       {start_date} ~ {latest_end_date}")
    print(f"   Holdout:        {holdout_months} 个月")
    print(f"   Folds:          {len(folds)}")
    print(f"   Seed:           {args.seed}")
    print(f"   Output:         {wf_dir}")
    print()

    for f in folds:
        train_end = f["holdout_start"]
        print(
            f"   Fold {f['fold_id']}: train → {train_end}, "
            f"holdout {f['holdout_start']} → {f['end_date']}"
        )

    if args.dry_run:
        print(f"\n{'=' * 70}")
        print("   [DRY-RUN] 以上为 fold 配置预览, 不执行")
        return 0

    # ── Phase 1: Run folds ──
    fold_results: List[Optional[Dict]] = []
    if not args.oos_only:
        wf_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 70}")
        print(f"📦 Phase 1: 运行 {len(folds)} 个 fold")
        print(f"{'=' * 70}")

        for fold in folds:
            fold_dir = wf_dir / f"fold_{fold['fold_id']}"
            # Resume: skip if fold_meta.json exists
            if args.resume and (fold_dir / "fold_meta.json").exists():
                meta = json.loads((fold_dir / "fold_meta.json").read_text())
                print(f"\n   ⏭️  Fold {fold['fold_id']} 已完成, 跳过 (--resume)")
                fold_results.append(meta)
                continue

            result = run_fold(
                strategy,
                cfg,
                fold,
                wf_dir,
                seed=args.seed,
                dry_run=False,
            )
            fold_results.append(result)
    else:
        # Load existing fold results
        for fold in folds:
            fold_dir = wf_dir / f"fold_{fold['fold_id']}"
            meta_path = fold_dir / "fold_meta.json"
            if meta_path.exists():
                fold_results.append(json.loads(meta_path.read_text()))
            else:
                fold_results.append(None)

    # ── Phase 2: OOS backtest ──
    print(f"\n{'=' * 70}")
    print(f"🔒 Phase 2: OOS Backtest (冻结配置)")
    print(f"{'=' * 70}")

    oos_results: List[Optional[Dict]] = []
    for i in range(1, len(folds)):
        cur_fold_dir = wf_dir / f"fold_{folds[i]['fold_id']}"
        prev_fold_dir = wf_dir / f"fold_{folds[i - 1]['fold_id']}"

        if not cur_fold_dir.exists() or not prev_fold_dir.exists():
            print(
                f"\n   ⚠️  Fold {folds[i]['fold_id']} 或 Fold {folds[i-1]['fold_id']} 目录不存在, 跳过"
            )
            oos_results.append(None)
            continue

        oos = run_oos_backtest(strategy, cur_fold_dir, prev_fold_dir, wf_dir)
        oos_results.append(oos)

    # ── Summary ──
    valid_fold_results = [fr for fr in fold_results if fr is not None]
    valid_oos_results = [oos for oos in oos_results if oos is not None]
    print_summary(strategy, folds, valid_fold_results, valid_oos_results)

    # ── Save summary ──
    summary = {
        "strategy": strategy,
        "folds": folds,
        "fold_results": [
            {k: v for k, v in fr.items() if k != "evidence_dir"}
            for fr in valid_fold_results
        ],
        "oos_results": valid_oos_results,
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = wf_dir / "wf_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n📄 Summary saved: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
