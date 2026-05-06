from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .context import PROJECT_ROOT


def run_step(
    name: str,
    cmd: List[str],
    log_file: Path,
    *,
    dry_run: bool = False,
    cwd: Optional[Path] = None,
    env_extra: Optional[Dict[str, str]] = None,
) -> Tuple[int, str]:
    cmd_str = " \\\n  ".join(cmd)
    header = f"\n{'='*70}\n[STEP] {name}\n{'='*70}\n$ {cmd_str}\n"
    if env_extra:
        _envs = " ".join(f"{k}={v}" for k, v in env_extra.items())
        header += f"$ env: {_envs}\n"
    print(header)

    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(header)

    if dry_run:
        print("  (dry-run, 跳过执行)")
        return 0, ""

    _env = None
    if env_extra:
        _env = {**os.environ, **env_extra}
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd or PROJECT_ROOT,
        env=_env,
    )
    output = proc.stdout + proc.stderr

    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(output)
        lf.write(f"\n[EXIT CODE] {proc.returncode}\n")

    lines = output.strip().split("\n")
    # 事件回测报告较长；末尾含「加仓诊断」+ path_efficiency 文末汇总，多打几行以免被截断
    tail_n = 95 if "Event Backtest" in name else 30
    summary = "\n".join(lines[-tail_n:]) if len(lines) > tail_n else output
    print(summary)

    if proc.returncode != 0:
        print(f"\n❌ Step '{name}' FAILED (exit code {proc.returncode})")
    else:
        print(f"\n✅ Step '{name}' completed")

    return proc.returncode, output


def find_output_dir(output: str, strategy: str) -> Optional[str]:
    """从 stdout 或磁盘解析 Gate Train 产物目录 ``.../<strategy>``。

    支持两种布局：
    - 新: ``results/<strategy>/train_final_* /<strategy>``
    - 旧: ``results/train_final_* /<strategy>``
    """
    esc = re.escape(strategy)
    for pat in (
        rf"(results/{esc}/train_final_\S+/{esc})",
        rf"(results/train_final_\S+/{esc})",
    ):
        m = re.search(pat, output)
        if m:
            return m.group(1)
    results_dir = PROJECT_ROOT / "results"
    candidates = sorted(
        results_dir.glob(f"{strategy}/train_final_*/{strategy}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            results_dir.glob(f"train_final_*/{strategy}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if candidates:
        return str(candidates[0].relative_to(PROJECT_ROOT))
    return None


def parse_backtest_stdout(output: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    m = re.search(r"Trades:\s*(\d+)", output) or re.search(r"交易数:\s*(\d+)", output)
    if m:
        metrics["total_trades"] = int(m.group(1))
    m = re.search(r"Mean R:\s*([\-\d.]+)", output)
    if m:
        metrics["mean_r"] = float(m.group(1))
    m = re.search(r"Win Rate:\s*([\d.]+)%", output) or re.search(
        r"胜率:\s*([\d.]+)%", output
    )
    if m:
        metrics["win_rate"] = float(m.group(1)) / 100
    m = re.search(r"Sharpe \(per-trade\):\s*([\-\d.]+)", output) or re.search(
        r"Sharpe \(R\):\s*([\-\d.]+)", output
    )
    if m:
        metrics["sharpe_per_trade"] = float(m.group(1))
    m = re.search(r"Total R:\s*([\-\d.]+)", output)
    if m:
        metrics["total_r"] = float(m.group(1))
    m = re.search(r"Max DD \(R\):\s*([\-\d.]+)", output)
    if m:
        metrics["max_drawdown_r"] = float(m.group(1))
    m = re.search(r"Final:\s*\$[\d.]+\s*\(([\+\-\d.]+)%\)", output)
    if m:
        metrics["equity_return_pct"] = float(m.group(1))
    m = re.search(r"Max DD:\s*([\d.]+)%", output)
    if m:
        metrics["max_drawdown_pct"] = float(m.group(1)) / 100
    return metrics
