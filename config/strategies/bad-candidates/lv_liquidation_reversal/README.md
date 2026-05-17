# LV Liquidation Reversal

Status: placeholder.

Purpose: reserve a strategy family for liquidation-exhaustion reversal logic.
This profile should study overshoot after one-sided liquidation, absorption, and
fast mean reversion once forced selling / buying is exhausted.

Planned direction:

- Signal source: liquidation spike, OI collapse, wick failure, absorption, post-event path.
- Trade direction: fade the exhausted liquidation impulse.
- Holding period: minutes to hours.
- Execution requirement: tick / event-level replay with strict slippage assumptions.
- Risk: second-leg cascade continuation, poor fills during liquidity holes.

