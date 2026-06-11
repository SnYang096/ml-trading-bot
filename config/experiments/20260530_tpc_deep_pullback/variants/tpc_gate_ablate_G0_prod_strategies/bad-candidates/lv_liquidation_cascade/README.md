# LV Liquidation Cascade

Status: placeholder.

Purpose: reserve a strategy family for liquidation-driven continuation logic.
This profile should study forced liquidation cascades, liquidity voids, and
short-lived momentum after leverage-vulnerability triggers.

Planned direction:

- Signal source: liquidation volume, OI / funding stress, price break, order-book imbalance.
- Trade direction: follow the liquidation cascade.
- Holding period: minutes to hours.
- Execution requirement: tick / event-level replay, not 2h-only OHLC.
- Risk: high slippage, failed cascade, exchange latency.

