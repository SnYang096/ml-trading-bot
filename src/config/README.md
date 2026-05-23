# `src/config` — strategy path & YAML loading helpers

This package is **not** a second copy of strategy settings. All strategy **data** lives under the repo-root tree:

```text
config/strategies/<slug>/
  meta.yaml, features.yaml
  research/          # rolling / experiment profiles (often not deployed to live)
  archetypes/        # adopted knobs: prefilter, gate, execution, …
```

`src/config/` holds **Python helpers** that implement how the rest of the codebase finds and loads those trees.

## Why it exists

Introduced around **2026-05** with the research / archetypes layout work ([ADR: strategy research + archetypes](../../docs/architecture/ADR_strategy_research_archetypes_layout.md)). CLI, live engines, archetype loader, multileg, adopt/deploy, and PCM all need the same answers:

- Given `--strategy bpc`, which directory is the package root?
- Is the input a strategy dir, a `research/*.yaml` profile, or another yaml path?
- Should `bad-candidates/<slug>/` be tried (research only vs live)?
- How do `extends:` chains merge?
- For multileg, how are profile + engine + archetype layers combined?

Without a shared module, those rules would be duplicated across many entry points.

## Modules

| File | Role |
|------|------|
| `strategy_layout.py` | Path resolution, packaged profile names, `extends` merge, safe copy skip lists |
| `multileg_config.py` | Load multileg profile + engine + archetype YAML into one effective dict |
| `strategy_validation.py` | Check that a strategy package has required research profiles / multileg types |

## What “path resolution” buys you

1. **Single implementation** — one place for layout rules; callers use functions instead of hard-coded paths.
2. **Stable CLI** — e.g. `--strategy bpc` can resolve canonical `config/strategies/bpc/` or archived `bad-candidates/bpc/` without users memorizing archive paths.
3. **Input normalization** — directory vs `research/<profile>.yaml` vs other yaml → `(config_dir, profile_path, engine_path)`.
4. **Consistent `extends` handling** — shared merge order for layered research profiles.
5. **Multileg layering** — one load path for chop_grid / trend_scalp style stacks.
6. **Early validation** — fail in CI/tools when required packaged files are missing.

If the repo ever enforced a **strict** layout (fixed paths only, no `bad-candidates`, no `extends`), much of this layer could be inlined or removed. The current tree intentionally allows archives and layered YAML, so the helpers stay useful for **live / research / CLI**, not for storing extra config.

## Imports and `PYTHONPATH`

Many modules use **`from src.config...`** (repo root on `PYTHONPATH`, e.g. `pytest.ini`: `pythonpath = src .`).

The business console Docker image must include this package (see `deploy/business-console/Dockerfile`: `COPY src/config`) because `mlbot_console.services.strategy_stage_regions` reuses `time_series_model.archetype.loader`, which imports `src.config.strategy_layout`.

Trade Map only needs a **read-only** view of archetype YAML + feature bus for shading; it does not need most path-resolution features. A future simplification is to decouple CMS from `archetype.loader` so the console image stays minimal.

## Naming note

The name `src/config` is easy to confuse with root **`config/`** (strategy YAML). Mentally read it as **strategy path resolution**, not “configuration files.” A rename (e.g. `strategy_paths`) would be cosmetic but touch many imports.

## Related docs

- [ADR: strategy research + archetypes layout](../../docs/architecture/ADR_strategy_research_archetypes_layout.md)
- [Business console deploy](../../deploy/business-console/README.md)
