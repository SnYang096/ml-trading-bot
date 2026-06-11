# BPC L240 experiment tree

`bpc_soft_phase_f.lookback_breakout=240` scales breakout/pullback/recovery to ~20d @ 240T.

**Caveat:** `box_structure_f` still emits `box_pos_120` / `box_breakout_*` on a **120-bar** box (~10d).
This tree is a **soft_phase-only** lookback ablation, not a full box-scale alignment.
For retest rules use `bpc_lb120_retest_strategies` (L120 + `box_pos_120`) or
`bpc_lb20_retest_strategies` (prod L20 + retest on Phase1-calibrated parquet).
