"""
Run the **best-practice** end-to-end loop for multiple strategies:

1) Generate Pool-B YAML via `mlbot analyze factor-eval`
2) Run `mlbot diagnose feature-group-search` in strict stages:
   - preset A (fast proxy) -> export shortlist groups.yaml
   - preset B (medium)     -> export shortlist groups.yaml
   - preset C (full verify) using B-shortlist
3) Generate a markdown summary report (includes A/B/C paths, final = C)

Each feature-group-search run uses:
   - semantic groups (auto groups-yaml by strategy dir name)
   - Pool-B YAML as additional singleton candidate groups

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
        out.append(
            RunSpec(
                strategy=s,
                strategy_dir=strategy_dir,
                pool_b_dir=pool_b_dir,
                pool_b_yaml=pool_b_yaml,
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
    min_trades: int,
    search_algo: str,
    out_dir: Path,
    writeback_yaml: Path,
    groups_yaml: Path | None,
    expand_semantic_singletons: bool,
    rerun_search: bool,
    preset: str,
) -> Path:
    _ensure_strategy_dir_exists(spec)
    result_json = out_dir / "feature_group_search_result.json"
    if result_json.exists() and not rerun_search:
        print(f"✅ feature-group-search result exists, reuse: {result_json}")
        return result_json

    out_dir.mkdir(parents=True, exist_ok=True)

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
        "--min-trades",
        str(min_trades),
        "--search-algo",
        str(search_algo),
        "--preset",
        str(preset),
        "--pool-b-yaml",
        str(spec.pool_b_yaml),
        # Let the tool pick groups-yaml automatically (strategy semantic yaml if present).
        "--invert-candidates-yaml",
        str(spec.pool_b_yaml),
        "--writeback-yaml",
        str(writeback_yaml),
        "--output-dir",
        str(out_dir),
        "--no-docker",
    ]
    if groups_yaml is not None:
        cmd.extend(["--groups-yaml", str(groups_yaml)])
    if expand_semantic_singletons:
        cmd.append("--expand-semantic-singletons")
    _run(cmd, cwd=ROOT)

    if not result_json.exists():
        raise RuntimeError(
            f"feature-group-search did not produce result: {result_json}"
        )
    print(f"✅ feature-group-search done: {result_json}")
    return result_json


def export_shortlist_groups(
    *,
    spec: RunSpec,
    result_json: Path,
    out_yaml: Path,
    expand_semantic_singletons: bool,
    mode: str = "prefilter_survivors",
    max_groups: int = 30,
) -> Path:
    """
    Export shortlist groups YAML from a previous feature-group-search result JSON.
    This is used to make A output become B input, and B output become C input.
    """
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "src.cli.main",
        "diagnose",
        "export-fgs-shortlist",
        "--base-strategy-config",
        str(spec.strategy_dir),
        "--result-json",
        str(result_json),
        "--output-yaml",
        str(out_yaml),
        "--mode",
        str(mode),
        "--pool-b-yaml",
        str(spec.pool_b_yaml),
        "--max-groups",
        str(max_groups),
    ]
    if expand_semantic_singletons:
        cmd.append("--expand-semantic-singletons")
    _run(cmd, cwd=ROOT)
    if not out_yaml.exists():
        raise RuntimeError(f"Shortlist groups YAML was not generated: {out_yaml}")
    return out_yaml


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
    result_json_paths: Dict[str, Dict[str, Path]],
    tag: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(
        f"# Feature Group Search Summary (Pool-B + semantic, staged A→B→C) — {tag}"
    )
    lines.append("")
    lines.append(
        "This report summarizes runs that include **semantic groups + Pool-B singletons**."
    )
    lines.append("")
    lines.append("## Runs included")
    lines.append("")
    for s in specs:
        stage_paths = result_json_paths.get(s.strategy) or {}
        if stage_paths:
            lines.append(f"- **{s.strategy}**:")
            lines.append(f"  - pool_b: `{s.pool_b_yaml.relative_to(ROOT)}`")
            for stage in ["A", "B", "C"]:
                rp = stage_paths.get(stage)
                if rp:
                    lines.append(f"  - stage_{stage}_result: `{rp.relative_to(ROOT)}`")
            # Convention: final writeback is stage C YAML
            final_writeback = (
                ROOT
                / "config"
                / "strategies"
                / s.strategy
                / f"features_suggested_{s.search_algo}_poolb_semantic_{tag}_C.yaml"
            )
            lines.append(f"  - final_writeback: `{final_writeback.relative_to(ROOT)}`")
    lines.append("")
    lines.append("Common params:")
    lines.append(f"- symbol: `{symbol}`")
    lines.append(f"- timeframe: `{timeframe}`")
    lines.append(f"- date range: `{start_date} .. {end_date}`")
    lines.append(
        "- stages: `A (CV_mean, 2 seeds) -> B (CV_mean, 3 seeds) -> C (Sharpe_mean, 5 seeds)`"
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")

    for s in specs:
        lines.append(f"### {s.strategy}")
        lines.append("")
        stage_paths = result_json_paths.get(s.strategy) or {}
        for stage in ["A", "B", "C"]:
            rpath = stage_paths.get(stage)
            if not rpath or not rpath.exists():
                continue
            d = json.loads(rpath.read_text())
            objective = d.get("objective")
            baseline = (d.get("baseline") or {}).get("score")
            history = d.get("history") or []
            selected_groups = d.get("selected_groups") or []
            final_features = d.get("final_features") or []
            stop_reason = d.get("stop_reason")
            lines.append(f"- **Stage {stage}**: `{rpath.relative_to(ROOT)}`")
            lines.append(f"  - objective: `{objective}`")
            lines.append(f"  - baseline_score: **{_fmt(baseline)}**")
            if history:
                last = history[-1]
                lines.append(
                    f"  - last_score: **{_fmt(last.get('score'))}** (step={last.get('step')})"
                )
            lines.append(
                f"  - selected_groups: `{', '.join(selected_groups)}`"
                if selected_groups
                else "  - selected_groups: *(none)*"
            )
            lines.append(f"  - final_requested_features: `{len(final_features)}` nodes")
            if stop_reason:
                lines.append(f"  - stop_reason: `{stop_reason}`")
            if stage == "C":
                # Only print full feature list for final stage to keep report short.
                if final_features:
                    lines.append(f"  - final_features: `{', '.join(final_features)}`")
            # seed-1 stats for that stage
            s1 = _seed1_stats(rpath.parent, history=history)
            if s1:
                lines.append(
                    f"  - seed1: `n_train={s1.get('n_train_samples')}`, `n_test={s1.get('n_test_samples')}`, `n_features={s1.get('n_features')}`, `sharpe={_fmt(s1.get('backtest_sharpe'))}`, `trades={s1.get('total_trades')}`"
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
    p.add_argument("--min-trades", type=int, default=10)
    p.add_argument("--shortlist-max-groups-a", type=int, default=30)
    p.add_argument("--shortlist-max-groups-b", type=int, default=20)
    p.add_argument(
        "--search-algo",
        default="pipeline",
        choices=["greedy", "halving", "beam", "sffs", "pipeline"],
        help="feature-group-search algo; best workflow defaults to pipeline (SH->Beam->SFFS).",
    )
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
    p.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip writing the markdown report (useful when running multiple strategies in parallel).",
    )
    args = p.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    tag = args.tag

    specs = _build_runs(strategies=strategies, tag=tag, search_algo=args.search_algo)
    result_json_paths: Dict[str, Dict[str, Path]] = {}

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
            # Stage A
            out_a = (
                ROOT
                / "results"
                / "feature_group_search"
                / f"{spec.strategy}_{args.search_algo}_poolb_semantic_{tag}_A"
            )
            wb_a = (
                ROOT
                / "config"
                / "strategies"
                / spec.strategy
                / f"features_suggested_{args.search_algo}_poolb_semantic_{tag}_A.yaml"
            )
            r_a = run_feature_group_search(
                spec,
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date,
                test_size=args.test_size,
                min_trades=args.min_trades,
                search_algo=args.search_algo,
                out_dir=out_a,
                writeback_yaml=wb_a,
                groups_yaml=None,
                expand_semantic_singletons=args.expand_semantic_singletons,
                rerun_search=args.rerun_search,
                preset="A",
            )
            shortlist_a = out_a / "shortlist_groups_A.yaml"
            export_shortlist_groups(
                spec=spec,
                result_json=r_a,
                out_yaml=shortlist_a,
                expand_semantic_singletons=args.expand_semantic_singletons,
                max_groups=int(args.shortlist_max_groups_a),
            )

            # Stage B (restricted by shortlist A)
            out_b = (
                ROOT
                / "results"
                / "feature_group_search"
                / f"{spec.strategy}_{args.search_algo}_poolb_semantic_{tag}_B"
            )
            wb_b = (
                ROOT
                / "config"
                / "strategies"
                / spec.strategy
                / f"features_suggested_{args.search_algo}_poolb_semantic_{tag}_B.yaml"
            )
            r_b = run_feature_group_search(
                spec,
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date,
                test_size=args.test_size,
                min_trades=args.min_trades,
                search_algo=args.search_algo,
                out_dir=out_b,
                writeback_yaml=wb_b,
                groups_yaml=shortlist_a,
                expand_semantic_singletons=args.expand_semantic_singletons,
                rerun_search=args.rerun_search,
                preset="B",
            )
            shortlist_b = out_b / "shortlist_groups_B.yaml"
            export_shortlist_groups(
                spec=spec,
                result_json=r_b,
                out_yaml=shortlist_b,
                expand_semantic_singletons=args.expand_semantic_singletons,
                max_groups=int(args.shortlist_max_groups_b),
            )

            # Stage C (final verification; restricted by shortlist B)
            out_c = (
                ROOT
                / "results"
                / "feature_group_search"
                / f"{spec.strategy}_{args.search_algo}_poolb_semantic_{tag}_C"
            )
            wb_c = (
                ROOT
                / "config"
                / "strategies"
                / spec.strategy
                / f"features_suggested_{args.search_algo}_poolb_semantic_{tag}_C.yaml"
            )
            r_c = run_feature_group_search(
                spec,
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date,
                test_size=args.test_size,
                min_trades=args.min_trades,
                search_algo=args.search_algo,
                out_dir=out_c,
                writeback_yaml=wb_c,
                groups_yaml=shortlist_b,
                expand_semantic_singletons=args.expand_semantic_singletons,
                rerun_search=args.rerun_search,
                preset="C",
            )
            result_json_paths[spec.strategy] = {"A": r_a, "B": r_b, "C": r_c}
    else:
        for spec in specs:
            base = ROOT / "results" / "feature_group_search"
            result_json_paths[spec.strategy] = {
                "A": base
                / f"{spec.strategy}_{args.search_algo}_poolb_semantic_{tag}_A"
                / "feature_group_search_result.json",
                "B": base
                / f"{spec.strategy}_{args.search_algo}_poolb_semantic_{tag}_B"
                / "feature_group_search_result.json",
                "C": base
                / f"{spec.strategy}_{args.search_algo}_poolb_semantic_{tag}_C"
                / "feature_group_search_result.json",
            }

    if not args.skip_report:
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
            out_path=report_path,
        )
    else:
        print("ℹ️  skip-report enabled: not writing summary markdown.")


if __name__ == "__main__":
    main()
