## Feature Search Playbook (Tree Models)

This document explains the **end-to-end feature search workflow** used in this repo for the 4 core tree-model strategies:
**SR reversal**, **SR breakout**, **compression breakout**, **trend following**.

It also defines the baseline-stability criteria and the next-step search algorithms (Successive Halving / Beam Search / SFFS).

---

## Key concepts (glossary)

### Feature node vs feature column

- **Feature node**: a compute function registered in the feature system, typically named `*_f` (e.g. `trade_cluster_scene_semantic_scores_f`).
- **Feature column**: an actual model input column produced by a node (e.g. `trade_cluster_absorption_scene_score`).

By default, most tools operate on **node-level** selection. Some nodes output multiple columns.

### Base features (Pool A)

Each strategy can define `config/strategies/<strategy>/features_base.yaml` as **mandatory features** for label/backtest correctness.
They are **always included** and **not optimized** (e.g. `atr_f`, `poc_hal_features_close_f`).

### Pool B (data-driven candidates)

**Pool B** is a YAML exported by `mlbot analyze factor-eval` (IC/IR-based filtering + correlation removal), used as
a data-driven candidate pool for wrapper search:

- Default output: `results/pools/<strategy>/pool_b/<tag>/features_pool_b.yaml`

### Semantic groups (human-maintained candidates)

Semantic groups live in YAML such as:

- `config/feature_groups_<strategy_dir>_semantic.yaml` (preferred if present)
- fallback: `config/feature_groups.yaml`

They represent **human-readable “path stories”** (compression/ignition/absorption/exhaustion) and curated blocks.

### `--expand-semantic-singletons` (column-level semantic ablation)

Some semantic nodes output multiple “scene columns” (often with opposite meaning across strategies).
With `--expand-semantic-singletons`, the search expands those blocks into **per-output-column singleton candidates**
so greedy can select only the helpful semantic column(s).

Trade-off: more candidates → more evaluations.

---

## Baseline stability: when we say “the baseline is stable”

We only upgrade algorithms after we have a stable, reproducible baseline across all 4 strategies.

**A baseline is considered stable when:**

- **Experimental surface is fixed**: symbol(s), timeframe, date range, `test_size`, `seeds`, `max_steps`, `min_trades`.
- **Re-run reproducibility**: repeating the same run yields broadly consistent:
  - `Sharpe_mean` (no large drift),
  - selected groups/features (core choices don’t flip frequently),
  - stop reason (e.g. “no further improvement”) behaves consistently.
- **All 4 strategies complete** and produce:
  - `feature_group_search_result.json`
  - `features_pool_b.yaml`
  - writeback `features_suggested*.yaml` (if enabled)
  - a unified summary report.

---

## Step-by-step workflow (recommended)

### Step 0: ensure feature contracts are correct

- Run contract checks and key feature tests before expensive searches (avoid silent bugs like ATR semantics issues).

### Step 1: generate Pool B (factor-eval)

Run IC/IR evaluation for a strategy to export Pool B:

```bash
mlbot analyze factor-eval \
  --strategy-config config/strategies/<strategy> \
  --symbol BTCUSDT --timeframe 240T \
  --start-date 2023-01-01 --end-date 2025-12-31 \
  --remove-correlated --filter-by-best-lag \
  --output-dir results/pools/<strategy>/pool_b/<tag> \
  --export-yaml results/pools/<strategy>/pool_b/<tag>/features_pool_b.yaml \
  --no-docker
```

### Step 2: run wrapper search (semantic groups + Pool B)

This is the main baseline wrapper loop:

```bash
mlbot diagnose feature-group-search \
  --base-strategy-config config/strategies/<strategy> \
  --symbol BTCUSDT --timeframe 240T \
  --start-date 2023-01-01 --end-date 2025-12-31 \
  --seeds 1,2,3,4,5 \
  --objective Sharpe_mean \
  --min-trades 10 --max-steps 10 \
  --pool-b-yaml results/pools/<strategy>/pool_b/<tag>/features_pool_b.yaml \
  --writeback-yaml config/strategies/<strategy>/features_suggested_<tag>.yaml \
  --output-dir results/feature_group_search/<strategy>_greedy_poolb_semantic_<tag> \
  --no-docker
```

### Step 3: singleton expansion ablation (semantic columns)

Run the same search but expand semantic blocks:

```bash
mlbot diagnose feature-group-search \
  ... \
  --expand-semantic-singletons
```

### Step 4: unify runs + compare

Compare:

- **non-singletons baseline** vs **singletons ablation**
- latest runs vs historical stable reruns (e.g. `docs/architecture/reports/feature_group_search_summary_20260102_rerun.md`)

Key things to compare per strategy:

- baseline `Sharpe_mean` vs final `Sharpe_mean`
- selected groups sequence (greedy history)
- final requested features (nodes or columns)
- reject reasons distribution (e.g. `min_trades`)

---

## Search algorithms (what they are, and why we need them)

### Current baseline: Greedy Forward Selection (GFS)

At each step:

- evaluate “current_selected + one_candidate_group” across multiple seeds
- pick the best improvement
- stop when no candidate strictly improves the objective

Pros: cheap-ish, easy to reason about.  
Cons: **local optimum**, misses **synergy** (A+B useful, but A alone not).

### Available in `feature-group-search` now

`mlbot diagnose feature-group-search` supports:

- `--search-algo greedy` (default)
- `--search-algo halving` (Successive Halving)
- `--search-algo beam`
- `--search-algo sffs`
- `--search-algo pipeline` (SH prefilter → Beam → SFFS prune)

### Recommended pipeline (SH → Beam → SFFS)

In practice, the most reliable workflow is to **combine** the algorithms:

- **Stage 1 (SH prefilter)**: rank candidates by “single-add” lift under small budgets, keep ~30 survivors.
- **Stage 2 (Beam)**: search for synergistic combinations among survivors (break greedy local optima).
- **Stage 3 (SFFS prune)**: remove redundant groups from the best combo to improve generalization.

How to run (recommended default):

```bash
mlbot diagnose feature-group-search \
  ... \
  --search-algo pipeline \
  --halving-stages 1,3,5 \
  --halving-top-fraction 0.25 \
  --halving-min-survivors 5 \
  --pipeline-survivors 30 \
  --beam-width 3 \
  --max-steps 5 \
  --sffs-max-backward-per-step 2
```

### Successive Halving (first upgrade)

Idea: allocate budget gradually.

- Evaluate many candidates with **small budget** (fewer seeds / fewer steps / shorter date range)
- Keep top fraction
- Re-evaluate survivors with **larger budget**

Why it fits this repo:

- We already have budgets (`seeds`, `max_steps`, time range) → can be reused.
- Big cost reduction when candidate count is large.

How to run (recommended for speed):

```bash
mlbot diagnose feature-group-search \
  ... \
  --search-algo halving \
  --halving-stages 1,3,5 \
  --halving-top-fraction 0.25 \
  --halving-min-survivors 5
```

### Beam Search (second upgrade)

Idea: keep the best **K partial solutions** at each depth (instead of only 1 as greedy does).

- step 1: keep top-K single-group additions
- step 2: for each of K paths, try adding candidates; keep top-K again

Why it helps:

- preserves “almost good” paths that become great after synergy groups are added.

How to run (recommended for synergy):

```bash
mlbot diagnose feature-group-search \
  ... \
  --search-algo beam \
  --beam-width 3
```

### SFFS (third upgrade)

SFFS = Sequential Floating Forward Selection.

Loop:

- forward add (like greedy)
- then repeatedly try **backward remove** any selected group that improves objective

Why it helps:

- fixes “early mistake” where greedy adds a group that later becomes harmful/redundant.

Cost: more evaluations (needs backward checks), so we do it after baselines are stable.

How to run (recommended for “add then prune”):

```bash
mlbot diagnose feature-group-search \
  ... \
  --search-algo sffs \
  --sffs-max-backward-per-step 2
```

---

## Practical notes

### Why “singleton results look sparse”

Singleton (column-level) candidates often look “too few” because:

- synergy requires multiple columns/blocks, which greedy may miss,
- singletons increase the candidate space → more local optimum traps,
- some strategies genuinely prefer a small “signal + scale columns” set.

This is exactly why we baseline with greedy first, then upgrade to Beam/SFFS/Halving.


