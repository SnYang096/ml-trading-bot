#!/usr/bin/env python3
"""
实盘监控 + 自动重训触发器

与 实盘监控系统设计.md 的对接:
  - Part A 本地研究监控: 下载实盘数据 → 计算指标 → 检测漂移
  - Part B 实盘监控联动: 灯号异常 → 触发本地验证 → 决定是否重训
  - 重训决策树: 触发条件满足 → 调用 auto_research_pipeline.py

触发条件 (config/research_pipeline.yaml → retrain_triggers):
  1. schedule_days:      定期重训 (默认 90 天)
  2. sharpe_decay_ratio:  实盘滚动 Sharpe / 回测基线 < 50%
  3. consecutive_losses:  连续亏损次数 >= 8
  4. max_data_age_days:   训练数据距今 > 120 天

数据来源 (优先级):
  A. order_management.db → positions 表 (已关仓交易)
  B. 实盘执行日志 JSONL  (fallback)
  C. 下载的 live parquet  (通过 download_monitor_data.sh)

用法:
    # 检查所有策略的触发条件 (只报告, 不重训)
    python scripts/monitor_retrain.py --check-only

    # 检查并在需要时自动触发重训
    python scripts/monitor_retrain.py

    # 强制重训指定策略
    python scripts/monitor_retrain.py --force --strategy fer

    # 只检查单个策略
    python scripts/monitor_retrain.py --check-only --strategy me

    # 指定 DB 路径
    python scripts/monitor_retrain.py --db data/order_management.db

Cron 示例 (每周一 UTC 6:00):
    0 6 * * 1  cd /home/yin/trading/ml_trading_bot && python scripts/monitor_retrain.py >> logs/monitor_retrain.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "research_pipeline.yaml"


# ====================================================================
# Config
# ====================================================================


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ====================================================================
# Research history helpers
# ====================================================================


def get_last_research(history_dir: Path, strategy: str) -> Optional[Dict[str, Any]]:
    """加载最近一次研究的 report.json."""
    strat_dir = history_dir / strategy
    if not strat_dir.exists():
        return None
    runs = sorted([d for d in strat_dir.iterdir() if d.is_dir()], reverse=True)
    for run_dir in runs:
        report = run_dir / "report.json"
        if report.exists():
            return json.loads(report.read_text(encoding="utf-8"))
    return None


def days_since_last_train(report: Optional[Dict[str, Any]]) -> int:
    """计算上次研究距今天数."""
    if not report:
        return 9999
    ts = report.get("timestamp", "")
    if not ts:
        return 9999
    try:
        # timestamp 格式: YYYYMMDD_HHMMSS
        dt = datetime.strptime(ts[:8], "%Y%m%d")
        return (datetime.now() - dt).days
    except (ValueError, IndexError):
        return 9999


def get_baseline_sharpe(report: Optional[Dict[str, Any]]) -> float:
    """从上次研究报告中获取回测 Sharpe."""
    if not report:
        return 0.0
    metrics = report.get("backtest_metrics", {})
    return float(metrics.get("sharpe_per_trade", 0.0))


def get_data_end_date(report: Optional[Dict[str, Any]]) -> Optional[str]:
    """从上次研究报告中获取训练数据截止日期."""
    if not report:
        return None
    return report.get("data_range", {}).get("end_date")


# ====================================================================
# Load live trades — 从 order_management.db 或 JSONL
# ====================================================================


def load_live_trades_from_db(
    db_path: Path,
    strategy: Optional[str] = None,
    days: int = 90,
) -> List[Dict[str, Any]]:
    """
    从 order_management.db positions 表加载已关仓交易.

    Schema (src/order_management/database/schema.sql):
      positions: position_id, symbol, side, entry_time, exit_time,
                 entry_price, exit_price, realized_pnl, status,
                 strategy_id, archetype, ...
    """
    if not db_path.exists():
        print(f"  ⚠️  DB 不存在: {db_path}")
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        query = """
            SELECT position_id, symbol, side, entry_time, exit_time,
                   entry_price, exit_price, realized_pnl, status,
                   strategy_id, archetype
            FROM positions
            WHERE status = 'closed'
              AND exit_time >= ?
        """
        params: list = [cutoff]

        if strategy:
            # strategy_id 可能存储为 "bpc" / "fer" / "me" / "lv"
            # 也可能是 archetype 名称, 需要模糊匹配
            query += " AND (LOWER(strategy_id) = ? OR LOWER(archetype) LIKE ?)"
            params.extend([strategy.lower(), f"%{strategy.lower()}%"])

        query += " ORDER BY exit_time ASC"
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError as e:
        print(f"  ⚠️  DB 查询失败: {e}")
        return []
    finally:
        conn.close()


def load_live_trades_from_jsonl(
    log_dir: Path,
    strategy: Optional[str] = None,
    days: int = 90,
) -> List[Dict[str, Any]]:
    """
    Fallback: 从 JSONL 执行日志加载交易.

    日志文件预期路径:
      data/live_execution_log.jsonl 或
      live/highcap/data/execution_log.jsonl
    """
    trades = []
    cutoff = datetime.now() - timedelta(days=days)

    candidates = [
        log_dir / "live_execution_log.jsonl",
        log_dir / "execution_log.jsonl",
        PROJECT_ROOT / "data" / "live_execution_log.jsonl",
        PROJECT_ROOT / "live" / "highcap" / "data" / "execution_log.jsonl",
    ]

    log_file = None
    for f in candidates:
        if f.exists():
            log_file = f
            break

    if not log_file:
        return []

    with open(log_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 过滤: 只要已关仓的
            if record.get("status") not in ("closed", "CLOSED"):
                continue

            exit_time = record.get("exit_time") or record.get("close_time") or ""
            try:
                dt = datetime.fromisoformat(
                    exit_time.replace("Z", "+00:00").replace("+00:00", "")
                )
            except (ValueError, AttributeError):
                continue

            if dt < cutoff:
                continue

            if strategy:
                rec_strategy = (
                    record.get("strategy_id")
                    or record.get("strategy")
                    or record.get("archetype")
                    or ""
                ).lower()
                if strategy.lower() not in rec_strategy:
                    continue

            trades.append(
                {
                    "position_id": record.get("position_id", ""),
                    "symbol": record.get("symbol", ""),
                    "side": record.get("side", ""),
                    "entry_time": record.get("entry_time", ""),
                    "exit_time": exit_time,
                    "entry_price": record.get("entry_price", 0),
                    "exit_price": record.get("exit_price", 0),
                    "realized_pnl": record.get("realized_pnl", 0),
                    "strategy_id": record.get("strategy_id", ""),
                    "archetype": record.get("archetype", ""),
                }
            )

    return trades


def load_live_trades(
    db_path: Path,
    log_dir: Path,
    strategy: Optional[str] = None,
    days: int = 90,
) -> List[Dict[str, Any]]:
    """加载实盘交易记录, DB 优先, JSONL fallback."""
    trades = load_live_trades_from_db(db_path, strategy=strategy, days=days)
    if trades:
        print(f"  📊 从 DB 加载 {len(trades)} 笔交易 (近 {days} 天)")
        return trades

    trades = load_live_trades_from_jsonl(log_dir, strategy=strategy, days=days)
    if trades:
        print(f"  📊 从 JSONL 加载 {len(trades)} 笔交易 (近 {days} 天)")
        return trades

    print(f"  ⚠️  无实盘交易记录 (DB: {db_path}, JSONL: {log_dir})")
    return []


# ====================================================================
# Metrics computation
# ====================================================================


def compute_live_sharpe(trades: List[Dict[str, Any]], window_days: int = 30) -> float:
    """
    计算实盘滚动 Sharpe (per-trade).

    只使用最近 window_days 内的交易.
    Sharpe = mean(pnl) / std(pnl) * sqrt(N_per_year)
    """
    if not trades:
        return 0.0

    cutoff = datetime.now() - timedelta(days=window_days)
    recent_pnl = []
    for t in trades:
        exit_time = t.get("exit_time", "")
        try:
            dt = datetime.fromisoformat(
                str(exit_time).replace("Z", "").replace("+00:00", "")
            )
        except (ValueError, AttributeError):
            continue
        if dt >= cutoff:
            pnl = float(t.get("realized_pnl", 0) or 0)
            recent_pnl.append(pnl)

    if len(recent_pnl) < 3:
        return 0.0

    arr = np.array(recent_pnl, dtype=float)
    mean_pnl = arr.mean()
    std_pnl = arr.std(ddof=1)

    if std_pnl < 1e-12:
        return 0.0

    # Per-trade Sharpe, 与 backtest_execution_layer 对齐
    return float(mean_pnl / std_pnl)


def compute_consecutive_losses(trades: List[Dict[str, Any]]) -> int:
    """计算当前最新的连续亏损次数 (从最后一笔向前计)."""
    if not trades:
        return 0

    # 按 exit_time 排序 (升序), 取最后部分
    sorted_trades = sorted(trades, key=lambda t: t.get("exit_time", ""))
    count = 0
    for t in reversed(sorted_trades):
        pnl = float(t.get("realized_pnl", 0) or 0)
        if pnl < 0:
            count += 1
        else:
            break
    return count


def compute_data_age_days(report: Optional[Dict[str, Any]]) -> int:
    """计算训练数据截止日期距今天数."""
    end_date = get_data_end_date(report)
    if not end_date:
        return 9999
    try:
        dt = datetime.strptime(end_date, "%Y-%m-%d")
        return (datetime.now() - dt).days
    except ValueError:
        return 9999


# ====================================================================
# Trigger evaluation
# ====================================================================


def check_triggers(
    strategy: str,
    trades: List[Dict[str, Any]],
    report: Optional[Dict[str, Any]],
    triggers_cfg: dict,
) -> Dict[str, Any]:
    """
    检查 4 个重训触发条件.

    返回 {triggered: bool, reasons: [...], details: {...}}
    """
    reasons: List[str] = []
    details: Dict[str, Any] = {}

    # ── 1. 定期重训 ──
    schedule_days = triggers_cfg.get("schedule_days", 90)
    days_elapsed = days_since_last_train(report)
    details["days_since_last_train"] = days_elapsed
    details["schedule_threshold"] = schedule_days
    if days_elapsed >= schedule_days:
        reasons.append(f"定期重训: 距上次研究 {days_elapsed} 天 >= {schedule_days} 天")

    # ── 2. Sharpe 衰减 ──
    sharpe_decay_ratio = triggers_cfg.get("sharpe_decay_ratio", 0.5)
    baseline_sharpe = get_baseline_sharpe(report)
    live_sharpe = compute_live_sharpe(trades, window_days=30)
    details["baseline_sharpe"] = baseline_sharpe
    details["live_sharpe_30d"] = live_sharpe

    if baseline_sharpe > 0:
        ratio = live_sharpe / baseline_sharpe
        details["sharpe_ratio"] = ratio
        if ratio < sharpe_decay_ratio:
            reasons.append(
                f"Sharpe 衰减: live={live_sharpe:.3f} / baseline={baseline_sharpe:.3f} "
                f"= {ratio:.2f} < {sharpe_decay_ratio}"
            )
    elif baseline_sharpe <= 0 and live_sharpe <= 0:
        details["sharpe_ratio"] = None
        reasons.append(
            f"Sharpe 双负: baseline={baseline_sharpe:.3f}, live={live_sharpe:.3f}"
        )

    # ── 3. 连续亏损 ──
    max_consec = triggers_cfg.get("consecutive_losses", 8)
    consec = compute_consecutive_losses(trades)
    details["consecutive_losses"] = consec
    details["consecutive_threshold"] = max_consec
    if consec >= max_consec:
        reasons.append(f"连续亏损: {consec} 次 >= {max_consec}")

    # ── 4. 数据过期 ──
    max_age = triggers_cfg.get("max_data_age_days", 120)
    data_age = compute_data_age_days(report)
    details["data_age_days"] = data_age
    details["data_age_threshold"] = max_age
    if data_age >= max_age:
        reasons.append(f"数据过期: 训练数据距今 {data_age} 天 >= {max_age} 天")

    triggered = len(reasons) > 0
    return {
        "strategy": strategy,
        "triggered": triggered,
        "trigger_count": len(reasons),
        "reasons": reasons,
        "details": details,
    }


# ====================================================================
# Retrain action
# ====================================================================


def trigger_retrain(
    strategy: str,
    trigger_result: Dict[str, Any],
    *,
    config_path: Path,
    dry_run: bool = False,
) -> int:
    """调用 auto_research_pipeline.py 执行重训."""
    print(f"\n{'='*60}")
    print(f"🔄 触发重训: {strategy.upper()}")
    print(f"{'='*60}")
    for r in trigger_result.get("reasons", []):
        print(f"   → {r}")
    print()

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "auto_research_pipeline.py"),
        "--strategy",
        strategy,
        "--config",
        str(config_path),
    ]

    if dry_run:
        print(f"  [DRY-RUN] 命令: {' '.join(cmd)}")
        return 0

    print(f"  执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


# ====================================================================
# Download live data
# ====================================================================


def download_live_data(days: int = 30) -> int:
    """调用 download_monitor_data.sh 从服务器下载数据."""
    script = PROJECT_ROOT / "live" / "scripts" / "download_monitor_data.sh"
    if not script.exists():
        print("  ⚠️  下载脚本不存在, 跳过远程数据下载")
        return 1

    print(f"\n📥 下载最近 {days} 天的实盘数据...")
    result = subprocess.run(
        ["bash", str(script), "--days", str(days)],
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode


# ====================================================================
# Report
# ====================================================================


def generate_monitoring_report(
    all_results: List[Dict[str, Any]],
    history_dir: Path,
) -> Path:
    """生成监控检查报告 (JSON)."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "strategies": all_results,
        "summary": {
            "total_checked": len(all_results),
            "triggered": sum(1 for r in all_results if r.get("triggered")),
            "strategies_triggered": [
                r["strategy"] for r in all_results if r.get("triggered")
            ],
        },
    }

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "monitor_retrain_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    return report_path


def print_check_summary(results: List[Dict[str, Any]]):
    """打印检查结果概览."""
    print(f"\n{'='*60}")
    print("📋 监控检查汇总")
    print(f"{'='*60}")

    for r in results:
        strategy = r["strategy"]
        triggered = r["triggered"]
        details = r.get("details", {})

        status = "🔴 需重训" if triggered else "🟢 正常"
        print(f"\n  {status}  {strategy.upper()}")
        print(f"     距上次研究:    {details.get('days_since_last_train', 'N/A')} 天")
        print(f"     回测 Sharpe:   {details.get('baseline_sharpe', 'N/A')}")
        print(f"     实盘 Sharpe:   {details.get('live_sharpe_30d', 'N/A')}")
        print(f"     连续亏损:      {details.get('consecutive_losses', 'N/A')} 次")
        print(f"     数据年龄:      {details.get('data_age_days', 'N/A')} 天")

        if triggered:
            for reason in r.get("reasons", []):
                print(f"     ⚡ {reason}")

    triggered_count = sum(1 for r in results if r["triggered"])
    total = len(results)
    print(f"\n  总计: {total} 个策略, {triggered_count} 个需重训")


# ====================================================================
# Main
# ====================================================================


def main():
    p = argparse.ArgumentParser(
        description="实盘监控 + 自动重训触发器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  检查所有策略:     python scripts/monitor_retrain.py --check-only
  自动触发重训:     python scripts/monitor_retrain.py
  强制重训 FER:     python scripts/monitor_retrain.py --force --strategy fer
  下载数据后检查:   python scripts/monitor_retrain.py --download --check-only
        """,
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="只检查触发条件, 不实际重训",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="强制触发重训 (忽略触发条件)",
    )
    p.add_argument(
        "--strategy",
        help="只检查/重训指定策略 (bpc/fer/me/lv)",
    )
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Pipeline 配置文件路径",
    )
    p.add_argument(
        "--db",
        help="order_management.db 路径 (默认: data/order_management.db)",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="先从服务器下载最新数据再检查",
    )
    p.add_argument(
        "--download-days",
        type=int,
        default=30,
        help="下载最近 N 天数据 (默认 30)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="打印重训命令但不执行",
    )
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    triggers_cfg = cfg.get("retrain_triggers", {})
    history_dir = PROJECT_ROOT / cfg["output"]["history_dir"]

    # DB 路径
    db_path = (
        Path(args.db) if args.db else PROJECT_ROOT / "data" / "order_management.db"
    )
    # 也支持实盘部署路径
    if not db_path.exists():
        alt_db = PROJECT_ROOT / "live" / "highcap" / "data" / "order_management.db"
        if alt_db.exists():
            db_path = alt_db

    log_dir = PROJECT_ROOT / "data"

    # 确定策略列表
    if args.strategy:
        strategies = [args.strategy]
    else:
        strategies = list(cfg.get("strategies", {}).keys())

    print("=" * 60)
    print("🔍 实盘监控 + 重训触发器")
    print("=" * 60)
    print(f"   配置:     {args.config}")
    print(f"   DB:       {db_path}")
    print(f"   策略:     {', '.join(strategies)}")
    print(f"   模式:     {'只检查' if args.check_only else '检查+重训'}")
    if args.force:
        print(f"   ⚡ 强制重训模式")
    if args.dry_run:
        print(f"   🏜️  Dry-run 模式")
    print("=" * 60)

    # ── 可选: 先下载数据 ──
    if args.download:
        download_live_data(args.download_days)

    # ── 检查每个策略 ──
    all_results: List[Dict[str, Any]] = []

    for strategy in strategies:
        if strategy not in cfg.get("strategies", {}):
            print(f"\n  ⚠️  未知策略: {strategy}, 跳过")
            continue

        print(f"\n{'─'*60}")
        print(f"  检查策略: {strategy.upper()}")
        print(f"{'─'*60}")

        # 1. 加载上次研究报告
        report = get_last_research(history_dir, strategy)
        if report:
            print(f"  📁 上次研究: {report.get('timestamp', 'N/A')}")
            print(f"     回测 Sharpe: {get_baseline_sharpe(report):.4f}")
        else:
            print(f"  📁 无历史研究记录")

        # 2. 加载实盘交易
        trades = load_live_trades(db_path, log_dir, strategy=strategy, days=90)

        # 3. 检查触发条件
        if args.force:
            trigger_result = {
                "strategy": strategy,
                "triggered": True,
                "trigger_count": 1,
                "reasons": ["强制触发 (--force)"],
                "details": {
                    "days_since_last_train": days_since_last_train(report),
                    "baseline_sharpe": get_baseline_sharpe(report),
                    "live_sharpe_30d": compute_live_sharpe(trades, window_days=30),
                    "consecutive_losses": compute_consecutive_losses(trades),
                    "data_age_days": compute_data_age_days(report),
                },
            }
        else:
            trigger_result = check_triggers(
                strategy,
                trades,
                report,
                triggers_cfg,
            )

        all_results.append(trigger_result)

    # ── 输出检查结果 ──
    print_check_summary(all_results)

    # ── 保存报告 ──
    report_path = generate_monitoring_report(all_results, history_dir)
    print(f"\n  📄 报告: {report_path}")

    # ── 触发重训 ──
    if args.check_only:
        print("\n  (--check-only 模式, 不触发重训)")
        return

    triggered_strategies = [r for r in all_results if r["triggered"]]
    if not triggered_strategies:
        print("\n  ✅ 所有策略正常, 无需重训")
        return

    retrain_results: List[Dict[str, Any]] = []
    for result in triggered_strategies:
        strategy = result["strategy"]
        rc = trigger_retrain(
            strategy,
            result,
            config_path=Path(args.config),
            dry_run=args.dry_run,
        )
        retrain_results.append(
            {
                "strategy": strategy,
                "exit_code": rc,
                "success": rc == 0,
            }
        )

    # ── 重训汇总 ──
    print(f"\n{'='*60}")
    print("📋 重训汇总")
    print(f"{'='*60}")
    for r in retrain_results:
        emoji = "✅" if r["success"] else "❌"
        print(f"  {emoji} {r['strategy']:>6s}: exit_code={r['exit_code']}")


if __name__ == "__main__":
    main()
