# Box Pullback Trend

Research candidate for trading pullbacks inside box/chop regimes in the direction
of the macro trend.

## Thesis

CRF failed as a bidirectional range-fade strategy because `box_pos` did not give a
stable standalone direction. However, diagnostics showed that box/chop edge
entries improve when aligned with a macro trend filter, such as BTC `EMA1200`
position and slope.

This strategy therefore treats box/chop as a **window selector**, not as a
direction source:

- Macro up: only take lower-edge pullbacks as long entries.
- Macro down: only take upper-edge pullbacks as short entries.
- Flat macro: no trade.

## Initial Diagnostic

Script:

```text
scripts/diagnose_box_pullback_trend.py
```

Default research setup:

- `box_window: 240`
- `box_stability >= 0.90`
- `0.04 <= box_width_pct <= 0.25`
- `box_touches_hi/lo >= 8`
- `edge_frac: 0.12`
- `semantic_chop >= 0.40`
- Diagnostic macro direction: BTC close vs BTC `EMA1200` with positive/negative `EMA1200` slope.
- Pipeline v1 macro direction: each symbol's own `EMA1200` position and slope, to keep the candidate lightweight until a generic BTC EMA anchor is added.
- Pipeline v1 box window: `120`, matching `bpt_macro_box_direction_f`.

The first comparison should focus on:

- Whether macro-aligned edge signals beat counter-macro edge signals.
- Whether ATR short exits or box-opposite exits are more stable.
- Whether signal timestamps overlap heavily with TPC/BPC/ME/SRB.

## Relationship to TPC

TPC is a general trend pullback continuation strategy. `box_pullback_trend` is a
narrower candidate: it only trades pullbacks that occur inside validated
box/chop windows. If overlap with TPC is low and standalone R remains positive,
it may become an additional trend family. If overlap is high, it should be folded
into TPC as a feature/gate rather than kept as a separate strategy.
