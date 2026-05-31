# Research Command Family — Phase 0 Capability Audit (2026-05-29)

**Purpose:** Verify actual runtime + source behavior of the new `mlbot research` family + `rd_loop` against the claims and gaps identified in the optimization plan. This audit was executed **before** any new implementation (per plan Phase 0).

**Scope:** Focused on ME/TPC-relevant paths (Gate/Entry/Prefilter) using real `features_labeled.parquet` where available (TPC validation_smoke + train_final artifacts) + source inspection + live CLI invocation. ME-specific labeled parquet not present in this workspace snapshot but patterns are identical via the shared kernels + layer_registry.

---

## 1. Verified Current-State Facts (corrections to optimistic assumptions)

These three were explicitly called out in the plan as "drive sizing" facts. All confirmed:

1. **`scripts/research/calibrate.py`**  
   - Only ever writes a **single scalar line**: `recommended_threshold: <val>` (plus two comment lines).  
   - Source: entire file ~47 LOC; `main()` extracts `recommended` / `mid` / `recommended_threshold` / `plateau_mid` from input JSON and emits exactly that.  
   - Runtime test (dummy plateau json): produced exactly the one-line draft.  
   - **Conclusion:** Cannot close a Gate loop today. No support for structured rule blocks, multi-rule, whitelist filtering, or writing `gate.yaml`-shaped output.

2. **`scripts/research/promote.py`**  
   - Pure `shutil.copy2(src, dst)` after a `--yes` human gate. No merge logic whatsoever.  
   - No handling of `locked: true`, `frozen`, `promote_never_disable`.  
   - No backup, no diff preview.  
   - Grep for "locked.*merge|merge.*locked" across `scripts/research/` → zero matches.  
   - **Conclusion:** Will clobber production `locked` rules on promote if the draft lacks them. Unsafe for real use on archetypes that contain locked rules.

3. **Gate lift / plateau / robustness kernels vs `optimize_gate_unified.py`**  
   - `src/research/stat_kernels/gate_lift.py` (`compute_lift_for_threshold`, `scan_thresholds_for_lift`) — **exist and are solid**.  
   - `src/research/stat_kernels/{plateau.py (find_stable_lift_plateau), robustness.py}` — also shared and already consumed by the old optimizer.  
   - `scripts/optimize_gate_unified.py` (~2000 LOC) already does the full thing: reads `archetypes/gate.yaml` + `features_gate.yaml` `allowed_gate_deny_features` (fnmatch whitelist), multi-rule combo pass-rate, interval writeback, robustness scoring, locked/frozen handling on write.  
   - **No CLI path** yet wires the new research verbs to this full flow for Gate. `research plateau` still only offers `--kpi {label,snotio}`.  
   - **Conclusion:** Math parity is feasible (reuse kernels); the work is integration + CLI surface + structured I/O + whitelist plumbing + parity harness (not inventing new statistics).

Additional audit findings:

4. **`research plateau --kpi`** (live via `PYTHONPATH=. python -c ...` import)  
   - Current choices: `{label, snotio}` only. No `lift`. Help text confirms.

5. **`research fit --layer gate`** (TPC + real validation_smoke parquet)  
   - `config/strategies/tpc/features_gate.yaml` contains **only** `feature_pipeline.allowed_gate_deny_features` (whitelist, fnmatch patterns). No `requested_features`.  
   - `FeaturePool.from_yaml` (src/research/subjects/feature.py) **only** reads `requested_features`.  
   - `layer_registry.FEATURES_POOL_YAML["gate"] = "features_gate.yaml"`.  
   - Result: `resolve_pool_columns` → empty → `fit.py` hits `ERROR: no feature columns from pool found in parquet`.  
   - Confirmed runtime failure.  
   - **Implication for (B) in docs:** Gate contract files are **whitelists for semantic safety**, not training candidate pools. Tree exploration on Gate (if any) must explicitly point at a full `features.yaml` via `--feature-pool`.

6. **`scripts/rd_loop.py` modes** (source + dispatch logic)  
   - Supported: `condition-set`, `feature-plateau`, `ic-decay`, `snotio-plateau`, `entry-plateau`, `pair-scan`.  
   - Explicit thin wrappers for `entry-plateau` (calls `entry_plateau_scan`).  
   - **No** `gate-plateau` or `locked-prefilter-tune` yet.  
   - `_build_research_scan_cmd` and step handlers are thin; adding the two new modes is low-risk wrapper work.

7. **Other verbs (scan, ic, robustness, segment, compare)**  
   - All registered and layer-agnostic (layer only affects mask + writeback hint).  
   - Robustness already imports the same `UnifiedOptimizationConfig` / `compute_robustness_score` used by the Gate optimizer.  
   - No blocking gaps found for non-Gate paths.

---

## 2. Capability Matrix (New Flow vs Current State)

| Area / Verb                        | Current (2026-05-29 audit)                                      | Needed for "New Explicit Flow as Default" (Gate/Entry/Prefilter + Tree) | Gap | Owner Todo |
|------------------------------------|-----------------------------------------------------------------|--------------------------------------------------------------------------|-----|------------|
| `research plateau --kpi lift`     | Absent (only label/snotio)                                     | Multi-rule Gate lift scan, read gate.yaml rules + features_gate whitelist (allowed_gate_deny_features), interval output, robustness tie-in | Large (core) | rd_loop-research-gate-core (Todo 2) |
| `research calibrate`              | One-line `recommended_threshold: X` scalar only                | Structured draft: full gate rule blocks (with locked/frozen preserved or merged), entry filter blocks, prefilter numerics; consume lift/plateau json | Medium | calibrate-promote-upgrade (Todo 3) |
| `research promote`                | Bare `shutil.copy2` + --yes prompt                             | locked-merge (never overwrite production locked rules), timestamped backup, unified diff preview, explicit human review | Medium | calibrate-promote-upgrade (Todo 3) |
| `rd_loop` `gate-plateau` mode     | Not implemented                                                | Thin wrapper: call research plateau --kpi lift, produce proposal json + draft path (like entry-plateau) | Medium | rd_loop-research-gate-core (Todo 2) |
| `rd_loop` `locked-prefilter-tune` | Not implemented                                                | Thin wrapper around existing locked_prefilter_parquet_tune logic under new verb | Small | rd_loop-research-gate-core (Todo 2) |
| `research fit --layer gate`       | Hard error (no requested_features in gate yaml)                | Documented escape hatch (`--feature-pool config/strategies/<s>/features.yaml`) + optional whitelist-as-candidates mode if tree-on-Gate becomes common | Doc + tiny | docs-foundation + tree-alignment |
| Pre-deploy `plateau_stability`    | Deferred / incomplete in contract_checks (per §9.5)            | Full multi-layer + cross-regime evidence gate (BLOCK on missing bull/bear variant-grid) | Medium | pre-deploy-harden (Todo 6) |
| Drift → rd_loop suggestion        | None                                                           | On ALERT emit runnable rd_loop yaml fragment or `mlbot research scan ...` cmd | Medium | drift-automation (Todo 5) |
| Parity harness (new vs old Gate)  | None                                                           | Side-by-side on same features_labeled.parquet: τ / interval / robustness within tolerance; report artifact | Required for deprecation | calibrate-promote-upgrade (Todo 3) |

---

## 3. Explicit "Must-Gain" List for calibrate + promote (for Todo 3)

**calibrate.py must become able to:**

- Accept `--from-plateau <lift_or_snotio_or_label.json>` (or generalize to a standard "proposal" envelope).
- When the input is Gate lift/plateau output: emit a **structured `gate.yaml` fragment** (list of rule dicts with id, feature, op, threshold or [lo,hi] interval, locked/frozen flags if present in source archetype, robustness metadata, etc.).
- Similarly for entry (filter blocks) and prefilter (numeric proposals).
- Respect `features_gate.yaml` whitelist when generating Gate drafts (only emit rules for allowed features).
- Write to `--output` with clear "DRAFT — human review + promote required" header + source traceability.
- Remain **non-destructive** (never touches live archetypes).

**promote.py must become able to:**

- Load both `--from` (draft) and `--to` (target archetype, e.g. `archetypes/gate.yaml`).
- Perform a **semantic merge**: for each rule in draft, if the corresponding production rule has `locked: true` / `frozen: true` / `promote_never_disable: true`, **preserve the production version** (or at minimum refuse to downgrade/disable it).
- On any change that would affect a locked rule, **error or require extra confirmation** (or always produce backup first).
- Always write a timestamped backup of the target before mutation (e.g. `gate.yaml.bak.20260529_1023`).
- Print a unified diff (or structured before/after) to stdout for the human to review before the write.
- The `--yes` flag still required; the diff/backup is mandatory even with --yes.
- Support an optional `--strategy` / `--layer` for context-aware merging rules.

**Parity harness (tests/research/ or scripts/audit/):**

- Given a fixed `features_labeled.parquet` (TPC or ME) + fixed gate archetype + features_gate whitelist:
  - Run `optimize_gate_unified.py` (old path) → record τ, intervals, robustness per rule.
  - Run the new `research plateau --kpi lift ... | research calibrate ...` path → record same.
  - Assert numeric tolerance (e.g. |Δτ| < 0.01 or robustness within 5% relative).
  - Produce `gate_parity_report_<ts>.md/json`.
- This artifact becomes the evidence that allows eventual hard deprecation of the old script for routine Gate work.

---

## 4. Implications for docs-foundation (Todo 1)

The audit directly feeds the three new sub-sections requested:

- **(B) features_*.yaml role re-definition + slim-down**  
  - KEEP (contracts): `features_gate.yaml:allowed_gate_deny_features`, `features_direction.yaml:candidates`, locked rule column names, tree `requested_features` (training/IC pool).  
  - CAN SLIM: the giant "throw everything at unattended meta" candidate lists that used to fuel `meta_algorithm` / quarterly optimize.  
  - Explicitly call out that `features_gate.yaml` for Gate is a **semantic safety whitelist**, not a model training feature list. Hence `research fit --layer gate` is intentionally narrow or errors unless user points elsewhere.

- **(C) Threshold-source convention**  
  - Condition-set quantiles (q50/q90 etc. pulled in rd_loop hypotheses or ME examples) = **probes / seeds for human hypothesis**, not production thresholds.  
  - Production `τ` or intervals **must** come from `research plateau` (label/snotio/lift) evidence of a flat high plateau, then `calibrate` → human review → `promote`.  
  - Document in 方法论 §3.4 and refactor plan §14 with examples from the user's open `rd_loop_me_entry_filter.yaml`.

- **(D) semantic_polarity.yaml disposition**  
  - Role: declares intended direction/sign for features (used by meta-algos and semantic guards).  
  - Under new flow: remains a **contract input** for Gate whitelist validation + direction checks + any future tree direction validation. Not "fuel for large-pool unattended discovery".  
  - Note it alongside the whitelist files.

Also update §9.5 P6 row: change the optimistic "✅" to "partial (scalar drafts + compare/robustness; structured Gate + safe promote pending Phase 9/9b)" and cross-link this audit.

---

## 5. Corrected Sizing & Risk Confirmation

- The split into **Todo 2 (rd_loop + plateau --kpi lift core)** + **Todo 3 (calibrate/promote upgrade + parity harness)** is **validated and necessary**. Treating P6 as "done" would have left an incomplete Gate story.
- No major new unknown risks surfaced; the heavy lifting (lift math, plateau detection, robustness) is already in shared kernels.
- Timeline estimate in the plan (~8.5–10.5 weeks total with parallel B + Tree streams) remains realistic.
- **Recommended first implementation order after this audit:**
  1. docs-foundation (incorporate audit + (B)(C)(D) + mermaid) — unblocks everything.
  2. rd_loop + plateau lift (Todo 2) — produces the lift json that Todo 3 consumes.
  3. calibrate/promote + parity (Todo 3) — closes the loop and gives the hard acceptance metric (ME/TPC gate refine with zero calls to optimize_gate_unified).
  4. Then tree, drift, pre-deploy, tests.

---

## 6. Artifacts & Commands Used in This Audit

- Source reads: `scripts/research/{calibrate, promote, plateau, fit, _common, rd_loop}.py`, `src/cli/main.py` (research group + _research_forward), `src/research/{layer_registry, subjects/feature, stat_kernels/gate_lift}.py`, `scripts/optimize_gate_unified.py` (header + locked handling + whitelist loader).
- Runtime probes:
  - `mlbot research --help`
  - `PYTHONPATH=. python -c 'from scripts.research.plateau import main; ...'` for --help and kpi choices.
  - `research fit --layer gate --strategy tpc --features-parquet results/validation_smoke/tpc/features_labeled.parquet` → confirmed error.
  - Dummy `calibrate` roundtrip → confirmed one-line output.
- Config inspected: `config/strategies/tpc/features_gate.yaml` (whitelist only).
- Parquet locations: `results/train_final/tpc/.../features_labeled.parquet`, `results/validation_smoke/tpc/features_labeled.parquet`, similar for BPC/SRB (ME not present in this snapshot).

---

**Next:** Mark Phase 0 complete in the running todo. Proceed to docs-foundation (or user-directed priority). This audit doc should be referenced from the main refactor plan §9.5 and the optimization plan's Phase 0 section.

**Status:** Ready for implementation of the remaining todos with accurate expectations. No "surprise" missing kernels; the gaps are integration, structured I/O, merge semantics, and wiring.