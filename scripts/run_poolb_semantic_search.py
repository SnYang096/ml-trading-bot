"""
Run a full 2-stage loop for multiple strategies:

1) Generate Pool-B YAML via `mlbot analyze factor-eval`
2) Run `mlbot diagnose feature-group-search` using:
   - semantic groups (auto groups-yaml by strategy dir name)
   - Pool-B YAML as additional singleton candidate groups
3) Generate a markdown summary report.

Key behavior:
- `--tag` affects ALL outputs: Pool-B dir, search output dir, writeback YAML, and report.

This script is meant to be re-runnable (idempotent-ish):
- If Pool-B YAML exists and --regen-poolb is not set, it will be reused.
- If feature-group-search result exists and --rerun-search is not set, it will be reused.

Example:
  python3 scripts/run_poolb_semantic_search.py \
    --symbol BTCUSDT --timeframe 240T \
    --start-date 2023-01-01 --end-date 2025-12-31 \
    --seeds 1,2,3,4,5 --objective Sharpe_mean \
    --max-steps 10 --test-size 0.3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parent.parent


DEFAULT_STRATEGIES = [
    "sr_reversal_rr_reg_long",
    "sr_breakout",
    "compression_breakout",
    "trend_following",
]


@dataclass(frozen=True)
class RunSpec:
    strategy: str
    strategy_dir: Path
    pool_b_dir: Path
    pool_b_yaml: Path
    fgs_out_dir: Path
    writeback_yaml: Path
    search_algo: str


def _run(
    cmd: Sequence[str], *, cwd: Path, env: Optional[Dict[str, str]] = None
) -> None:
    print("\n" + "=" * 100)
    print("CMD:", " ".join(cmd))
    print("=" * 100)
    subprocess.run(list(cmd), cwd=str(cwd), env=env, check=True)


def _today_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _pool_b_dir(strategy: str, tag: str) -> Path:
    # Tag-aware Pool-B dir to avoid overwriting the default pool_b YAML across runs.
    # Convention:
    #   results/pools/<strategy>/pool_b/<tag>/features_pool_b.yaml
    return ROOT / "results" / "pools" / strategy / "pool_b" / tag


def _build_runs(*, strategies: List[str], tag: str, search_algo: str) -> List[RunSpec]:
    out: List[RunSpec] = []
    for s in strategies:
        strategy_dir = ROOT / "config" / "strategies" / s
        pool_b_dir = _pool_b_dir(s, tag)
        pool_b_yaml = pool_b_dir / "features_pool_b.yaml"
        fgs_out_dir = (
            ROOT
            / "results"
            / "feature_group_search"
            / f"{s}_{search_algo}_poolb_semantic_{tag}"
        )
        writeback_yaml = (
            ROOT
            / "config"
            / "strategies"
            / s
            / f"features_suggested_{search_algo}_poolb_semantic_{tag}.yaml"
        )
        out.append(
            RunSpec(
                strategy=s,
                strategy_dir=strategy_dir,
                pool_b_dir=pool_b_dir,
                pool_b_yaml=pool_b_yaml,
                fgs_out_dir=fgs_out_dir,
                writeback_yaml=writeback_yaml,
                search_algo=search_algo,
            )
        )
    return out


def _ensure_strategy_dir_exists(spec: RunSpec) -> None:
    if not spec.strategy_dir.exists():
        raise FileNotFoundError(f"Strategy dir not found: {spec.strategy_dir}")


def generate_pool_b(
    spec: RunSpec,
    *,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    regen_poolb: bool,
) -> Path:
    _ensure_strategy_dir_exists(spec)
    if spec.pool_b_yaml.exists() and not regen_poolb:
        print(f"✅ Pool-B exists, reuse: {spec.pool_b_yaml}")
        return spec.pool_b_yaml

    spec.pool_b_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: explicitly set output-dir/export-yaml to the tag-aware pool_b paths,
    # instead of relying on factor-eval's convention defaults.
    cmd = [
        sys.executable,
        "-m",
        "src.cli.main",
        "analyze",
        "factor-eval",
        "--strategy-config",
        str(spec.strategy_dir),
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--remove-correlated",
        "--filter-by-best-lag",
        "--output-dir",
        str(spec.pool_b_dir),
        "--export-yaml",
        str(spec.pool_b_yaml),
        "--no-docker",
    ]
    _run(cmd, cwd=ROOT)

    if not spec.pool_b_yaml.exists():
        raise RuntimeError(f"Pool-B YAML was not generated: {spec.pool_b_yaml}")
    print(f"✅ Pool-B generated: {spec.pool_b_yaml}")
    return spec.pool_b_yaml


def run_feature_group_search(
    spec: RunSpec,
    *,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    test_size: float,
    seeds: str,
    objective: str,
    min_trades: int,
    max_steps: int,
    search_algo: str,
    halving_stages: str,
    halving_top_fraction: float,
    halving_min_survivors: int,
    pipeline_survivors: int,
    beam_width: int,
    sffs_max_backward_per_step: int,
    expand_semantic_singletons: bool,
    rerun_search: bool,
) -> Path:
    _ensure_strategy_dir_exists(spec)
    result_json = spec.fgs_out_dir / "feature_group_search_result.json"
    if result_json.exists() and not rerun_search:
        print(f"✅ feature-group-search result exists, reuse: {result_json}")
        return result_json

    spec.fgs_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "src.cli.main",
        "diagnose",
        "feature-group-search",
        "--base-strategy-config",
        str(spec.strategy_dir),
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--test-size",
        str(test_size),
        "--seeds",
        seeds,
        "--objective",
        objective,
        "--min-trades",
        str(min_trades),
        "--max-steps",
        str(max_steps),
        "--search-algo",
        str(search_algo),
        "--halving-stages",
        str(halving_stages),
        "--halving-top-fraction",
        str(halving_top_fraction),
        "--halving-min-survivors",
        str(halving_min_survivors),
        "--pipeline-survivors",
        str(pipeline_survivors),
        "--beam-width",
        str(beam_width),
        "--sffs-max-backward-per-step",
        str(sffs_max_backward_per_step),
        "--pool-b-yaml",
        str(spec.pool_b_yaml),
        # Let the tool pick groups-yaml automatically (strategy semantic yaml if present).
        "--invert-candidates-yaml",
        str(spec.pool_b_yaml),
        "--writeback-yaml",
        str(spec.writeback_yaml),
        "--output-dir",
        str(spec.fgs_out_dir),
        "--no-docker",
    ]
    if expand_semantic_singletons:
        cmd.append("--expand-semantic-singletons")
    _run(cmd, cwd=ROOT)

    if not result_json.exists():
        raise RuntimeError(
            f"feature-group-search did not produce result: {result_json}"
        )
    print(f"✅ feature-group-search done: {result_json}")
    return result_json


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def _seed1_stats(fgs_out_dir: Path, *, history: List[dict]) -> Dict[str, object]:
    # Follow the last accepted step for seed_1 stats.
    if history:
        last = history[-1]
        step_dir = f"step{last['step']}_add_{last['added_group']}"
    else:
        step_dir = "baseline"
    candidates = list(
        (fgs_out_dir / "runs" / step_dir / "seed_1").glob("*/results.json")
    )
    if not candidates:
        return {}
    payload = json.loads(candidates[0].read_text())
    bt = payload.get("backtest", {}) or {}
    return {
        "seed1_results_path": str(candidates[0]),
        "n_train_samples": payload.get("n_train_samples"),
        "n_test_samples": payload.get("n_test_samples"),
        "n_features": payload.get("n_features"),
        "backtest_sharpe": bt.get("sharpe"),
        "total_trades": bt.get("total_trades"),
    }


def write_report(
    *,
    specs: List[RunSpec],
    result_json_paths: Dict[str, Path],
    tag: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    seeds: str,
    objective: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(
        f"# Feature Group Search Summary (Pool-B + semantic, greedy multi-seed) — {tag}"
    )
    lines.append("")
    lines.append(
        "This report summarizes runs that include **semantic groups + Pool-B singletons**."
    )
    lines.append("")
    lines.append("## Runs included")
    lines.append("")
    for s in specs:
        r = result_json_paths.get(s.strategy)
        if r:
            lines.append(f"- **{s.strategy}**: `{r.relative_to(ROOT)}`")
            lines.append(f"  - pool_b: `{s.pool_b_yaml.relative_to(ROOT)}`")
            lines.append(f"  - writeback: `{s.writeback_yaml.relative_to(ROOT)}`")
    lines.append("")
    lines.append("Common params:")
    lines.append(f"- symbol: `{symbol}`")
    lines.append(f"- timeframe: `{timeframe}`")
    lines.append(f"- date range: `{start_date} .. {end_date}`")
    lines.append(f"- seeds: `{seeds}`")
    lines.append(f"- objective: `{objective}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")

    for s in specs:
        rpath = result_json_paths.get(s.strategy)
        if not rpath or not rpath.exists():
            continue
        d = json.loads(rpath.read_text())
        baseline = (d.get("baseline") or {}).get("score")
        history = d.get("history") or []
        selected_groups = d.get("selected_groups") or []
        final_features = d.get("final_features") or []
        stop_reason = d.get("stop_reason")

        lines.append(f"### {s.strategy}")
        lines.append("")
        lines.append(
            f"- **Selected groups**: `{', '.join(selected_groups)}`"
            if selected_groups
            else "- **Selected groups**: *(none)*"
        )
        lines.append(f"- **Final requested_features ({len(final_features)} nodes)**:")
        if final_features:
            lines.append(f"  - `{', '.join(final_features)}`")
        else:
            lines.append("  - *(none)*")
        lines.append(f"- **{objective} (multi-seed)**:")
        lines.append(f"  - baseline: **{_fmt(baseline)}**")
        for h in history:
            lines.append(f"  - +`{h['added_group']}`: **{_fmt(h.get('score'))}**")
        if stop_reason:
            lines.append(f"- **stop_reason**: `{stop_reason}`")

        s1 = _seed1_stats(s.fgs_out_dir, history=history)
        if s1:
            lines.append(
                f"- **Seed-1 training stats** (`{Path(s1['seed1_results_path']).relative_to(ROOT)}`)"
            )
            lines.append(
                f"  - `n_train_samples={s1.get('n_train_samples')}`, `n_test_samples={s1.get('n_test_samples')}`"
            )
            lines.append(f"  - `n_features={s1.get('n_features')}`")
            lines.append(
                f"  - backtest: `sharpe={_fmt(s1.get('backtest_sharpe'))}`, `total_trades={s1.get('total_trades')}`"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Report written: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    p.add_argument("--tag", default=_today_tag())
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--timeframe", default="240T")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seeds", default="1,2,3,4,5")
    p.add_argument("--objective", default="Sharpe_mean")
    p.add_argument("--min-trades", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument(
        "--search-algo",
        default="greedy",
        choices=["greedy", "halving", "beam", "sffs", "pipeline"],
        help="feature-group-search algo; pipeline is SH->Beam->SFFS.",
    )
    p.add_argument(
        "--halving-stages",
        default="1,3,5",
        help="Halving budgets in seeds-counts, e.g. 1,3,5",
    )
    p.add_argument("--halving-top-fraction", type=float, default=0.25)
    p.add_argument("--halving-min-survivors", type=int, default=5)
    p.add_argument("--pipeline-survivors", type=int, default=30)
    p.add_argument("--beam-width", type=int, default=3)
    p.add_argument("--sffs-max-backward-per-step", type=int, default=2)
    p.add_argument(
        "--expand-semantic-singletons",
        action="store_true",
        help="Expand semantic score blocks into singleton output-column groups for selection (finer-grained).",
    )
    p.add_argument(
        "--regen-poolb", action="store_true", help="Force regenerate Pool-B YAML"
    )
    p.add_argument(
        "--rerun-search",
        action="store_true",
        help="Force rerun feature-group-search even if result exists",
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate report (requires results.json present)",
    )
    args = p.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    tag = args.tag

    specs = _build_runs(strategies=strategies, tag=tag, search_algo=args.search_algo)
    result_json_paths: Dict[str, Path] = {}

    if not args.report_only:
        for spec in specs:
            generate_pool_b(
                spec,
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date,
                regen_poolb=args.regen_poolb,
            )
            result_json = run_feature_group_search(
                spec,
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date,
                test_size=args.test_size,
                seeds=args.seeds,
                objective=args.objective,
                min_trades=args.min_trades,
                max_steps=args.max_steps,
                search_algo=args.search_algo,
                halving_stages=args.halving_stages,
                halving_top_fraction=args.halving_top_fraction,
                halving_min_survivors=args.halving_min_survivors,
                pipeline_survivors=args.pipeline_survivors,
                beam_width=args.beam_width,
                sffs_max_backward_per_step=args.sffs_max_backward_per_step,
                expand_semantic_singletons=args.expand_semantic_singletons,
                rerun_search=args.rerun_search,
            )
            result_json_paths[spec.strategy] = result_json
    else:
        for spec in specs:
            result_json_paths[spec.strategy] = (
                spec.fgs_out_dir / "feature_group_search_result.json"
            )

    report_path = (
        ROOT
        / "docs"
        / "architecture"
        / "reports"
        / f"feature_group_search_summary_{tag}_poolb_semantic.md"
    )
    write_report(
        specs=specs,
        result_json_paths=result_json_paths,
        tag=tag,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        seeds=args.seeds,
        objective=args.objective,
        out_path=report_path,
    )


if __name__ == "__main__":
    main()
