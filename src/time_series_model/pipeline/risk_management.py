"""Risk management module for dynamic stop loss and take profit."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from time_series_model.config.settings import (
    STOP_LOSS_MULTIPLIER,
    TAKE_PROFIT_MULTIPLIER,
    MAX_CONSECUTIVE_LOSSES,
)


class RiskManager:
    """Handles risk management including regime-aware sizing, dynamic stops, and take profit."""

    def __init__(self):
        """Initialize the risk manager with sane defaults."""
        # Position history and PnL tracking
        self.position_history: List[Dict[str, float]] = []
        self.consecutive_losses: int = 0
        self.rolling_returns: List[float] = []
        self.rolling_window: int = 50  # for simple realized vol proxy
        self.equity: float = 100000.0
        self.equity_peak: float = self.equity
        self.drawdown: float = 0.0
        # Base sizing parameters
        self.base_risk_per_trade: float = 0.005  # 0.5% of equity
        self.target_annual_vol: float = 0.30
        self.vol_scale_min: float = 0.5
        self.vol_scale_max: float = 2.0
        # Regime weighting (soft gain)
        self.regime_weights: Dict[str, float] = {
            "trending": 1.5,
            "pre_breakout": 1.2,
            "range": 0.8,
            "collapse": 0.4,
            "transition": 0.7,
        }
        self.regime_gain_clip: Tuple[float, float] = (0.5, 1.8)
        # Anti-martingale (win-add) settings
        self.max_adds: int = 2
        self.add_ladder: List[float] = [1.0, 0.7]  # relative to base s2
        self.add_count: int = 0
        self.last_trade_profitable: bool = False
        self.cooldown_bars: int = 0
        self.cooldown_remaining: int = 0
        # Tiered take-profit and trailing stop (per-bar sigma multiples)
        self.tp1_mult: float = 2.0
        self.tp2_mult: float = 3.5
        self.trailing_mult: float = 2.5
        # Position state for execution-like behavior
        self.entry_price: Optional[float] = None
        self.tp1_hit: bool = False
        self.tp2_hit: bool = False
        self.prev_units: float = 0.0
        # Risk mode & throttles
        self.risk_mode: str = "Normal"  # Aggressive | Normal | Defensive
        self.dd_threshold_defensive: float = 0.12  # 12% DD -> Defensive
        self.dd_threshold_aggressive: float = 0.04  # <4% DD -> may Aggressive when trend high
        # Constraints (single-asset simplified)
        self.max_per_asset: float = 0.15  # 15% of equity notional
        self.max_total_exposure: float = 2.0  # leverage cap (not used in single-asset)

    def calculate_dynamic_levels(
        self, historical_prices: pd.Series
    ) -> Tuple[float, float]:
        """
        Calculate dynamic stop loss and take profit levels based on historical volatility.

        Args:
            historical_prices: Historical price series

        Returns:
            Tuple of (stop_loss_level, take_profit_level)
        """
        # Calculate rolling returns
        returns = historical_prices.pct_change().dropna()

        # Update rolling returns history
        self.rolling_returns.extend(returns.tolist())
        if len(self.rolling_returns) > self.rolling_window:
            self.rolling_returns = self.rolling_returns[-self.rolling_window :]

        # Calculate statistics
        if len(self.rolling_returns) > 1:
            mean_return = float(np.mean(self.rolling_returns))
            std_return = float(np.std(self.rolling_returns))
        else:
            mean_return = 0.0
            std_return = 0.01  # Default value to avoid division by zero

        # Calculate dynamic levels
        stop_loss_level = float(STOP_LOSS_MULTIPLIER * std_return)
        take_profit_level = float(TAKE_PROFIT_MULTIPLIER * std_return)

        return stop_loss_level, take_profit_level

    def check_structural_failure(self) -> bool:
        """
        Check for structural failure (consecutive losses).

        Returns:
            True if structural failure detected, False otherwise
        """
        return self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES

    def update_position_history(self, position: float, pnl: float, account_value: float):
        """
        Update position history and consecutive losses counter.

        Args:
            position: Current position size
            pnl: Profit and loss for the trade
        """
        self.position_history.append(
            {
                "position": float(position),
                "pnl": float(pnl),
                "timestamp": len(self.position_history),  # Simple timestamp
            }
        )

        # Update consecutive losses counter
        if pnl < 0:
            self.consecutive_losses += 1
            self.last_trade_profitable = False
        else:
            self.consecutive_losses = 0
            self.last_trade_profitable = True
        # Update equity and drawdown
        self.equity = float(account_value) + float(
            np.sum([h["pnl"] for h in self.position_history])
        )
        self.equity_peak = max(self.equity_peak, self.equity)
        if self.equity_peak > 0:
            self.drawdown = 1.0 - (self.equity / self.equity_peak)
        else:
            self.drawdown = 0.0
        # Cooldown progression
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    def _compute_regime_gain(self, regime_probs: Optional[Dict[str, float]]) -> float:
        if not regime_probs:
            return 1.0
        gain = 0.0
        for key, w in self.regime_weights.items():
            prob = regime_probs.get(key, 0.0)
            gain += w * prob
        low, high = self.regime_gain_clip
        return float(np.clip(gain, low, high))

    def _update_risk_mode(self, regime_probs: Optional[Dict[str, float]]) -> None:
        collapse_p = 0.0 if not regime_probs else float(regime_probs.get("collapse", 0.0))
        if self.drawdown >= self.dd_threshold_defensive or collapse_p > 0.6:
            self.risk_mode = "Defensive"
        elif self.drawdown <= self.dd_threshold_aggressive and (regime_probs and float(regime_probs.get("trending", 0.0)) > 0.6):
            self.risk_mode = "Aggressive"
        else:
            self.risk_mode = "Normal"

    def adjust_position_size(
        self,
        signal: float,  # -1..1
        expected_return: float,  # r_hat
        current_price: float,  # last price
        sigma_hat: float,  # realized/predicted vol (per bar)
        regime_probs: Optional[Dict[str, float]] = None,
        account_value: float = 100000.0,
    ) -> float:
        """
        Regime-aware, vol-targeted, anti-martingale sizing with constraints.

        Args:
            signal: Trading signal (-1 to 1)
            expected_return: Expected return from model
            current_price: Current asset price
            sigma_hat: Realized/predicted volatility (per-bar)
            regime_probs: Dict of regime probabilities {trending, range, pre_breakout, collapse, transition}
            account_value: Current account value

        Returns:
            Adjusted position size
        """
        # Update risk mode based on drawdown and regime state
        self._update_risk_mode(regime_probs)
        # 1) Base risk per trade (as fraction of equity)
        base_notional = account_value * self.base_risk_per_trade
        # 2) EV scaling
        ev_scale = max(0.0, min(3.0, abs(expected_return) / max(1e-9, np.median([abs(expected_return), 1e-4]))))
        s0 = base_notional * max(0.0, min(1.0, abs(signal))) * ev_scale
        # 3) Vol targeting (annualized -> per-bar proxy with sqrt(252*bars_per_day) omitted; use min-max clamp)
        vol_scale = 1.0
        if sigma_hat > 0:
            vol_scale = float(np.clip(self.target_annual_vol / (sigma_hat * np.sqrt(252.0)), self.vol_scale_min, self.vol_scale_max))
        s1 = s0 * vol_scale
        # 4) Regime gain
        g = self._compute_regime_gain(regime_probs)
        s2 = s1 * g
        # 5) Anti-martingale (win-add) subject to cooldown
        s3 = s2
        if self.cooldown_remaining > 0:
            s3 = s2 * 0.5  # reduced risk during cooldown
        elif self.last_trade_profitable and self.add_count < self.max_adds:
            s3 = s2 * self.add_ladder[min(self.add_count, len(self.add_ladder) - 1)]
        # 6) Risk mode throttle
        if self.risk_mode == "Defensive":
            s3 *= 0.5
        elif self.risk_mode == "Aggressive":
            s3 *= 1.2
        # 7) Per-asset cap
        cap_notional = account_value * self.max_per_asset
        notional = float(np.clip(s3, 0.0, cap_notional))
        # Convert notional to units
        units = notional / max(current_price, 1e-9)
        # Apply direction
        if signal > 0:
            final_position = units
        elif signal < 0:
            final_position = -units
        else:
            final_position = 0.0

        return float(final_position)

    def apply_risk_management(
        self,
        ensemble_df: pd.DataFrame,
        price_data: pd.DataFrame,
        regime_probs: Optional[pd.DataFrame] = None,
        account_value: float = 100000.0,
        vol_window: int = 50,
    ) -> pd.DataFrame:
        """
        Apply risk management rules to ensemble predictions.

        Args:
            ensemble_df: DataFrame with ensemble predictions
            price_data: Price data for dynamic level calculation
            regime_probs: Optional per-index regime probability DataFrame (columns: regime names)
            account_value: Starting account value
            vol_window: Window for realized volatility estimation

        Returns:
            DataFrame with risk-managed positions
        """
        result_df = ensemble_df.copy().sort_index()

        # Initialize position column
        result_df["position"] = 0.0
        result_df["stop_loss_level"] = 0.0
        result_df["take_profit_level"] = 0.0
        result_df["risk_mode"] = "Normal"

        # Align price series to signals
        px = price_data.copy().sort_index()
        if "close" not in px.columns:
            raise ValueError("price_data must contain 'close' column")
        # Estimate realized vol (per bar) via rolling std of log returns
        log_ret = np.log(px["close"]).diff().fillna(0.0)
        realized_vol = (
            log_ret.rolling(vol_window, min_periods=min(5, vol_window))
            .std()
            .reindex(result_df.index)
            .ffill()
            .fillna(0.0)
        )

        # Calculate dynamic levels
        stop_loss_level, take_profit_level = self.calculate_dynamic_levels(px["close"])

        # Apply risk management for each prediction
        prev_price = None
        for ts, row in result_df.iterrows():
            signal = float(row.get("discrete_signal", 0.0))
            expected_return = float(row.get("ensemble_return", 0.0))
            current_price = (
                float(
                    px["close"]
                    .reindex([ts])
                    .ffill()
                    .bfill()
                    .iloc[0]
                )
                if len(px) > 0
                else 100.0
            )
            sigma_hat = float(realized_vol.reindex([ts]).ffill().fillna(0.0).iloc[0])
            regime_row = None
            if regime_probs is not None and ts in regime_probs.index:
                # Normalize to expected keys
                rp = regime_probs.loc[ts].to_dict()
                regime_row = {
                    "trending": float(rp.get("trending", rp.get("TRENDING", 0.0))),
                    "pre_breakout": float(rp.get("pre_breakout", rp.get("PRE_BREAKOUT", 0.0))),
                    "range": float(rp.get("range", rp.get("RANGE", 0.0))),
                    "collapse": float(rp.get("collapse", rp.get("COLLAPSE", 0.0))),
                    "transition": float(rp.get("transition", rp.get("TRANSITION", 0.0))),
                }

            # Check for structural failure
            if self.check_structural_failure():
                # Close position if structural failure detected
                position = 0.0
            else:
                # Adjust position size based on risk management
                position = self.adjust_position_size(
                    signal=signal,
                    expected_return=expected_return,
                    current_price=current_price,
                    sigma_hat=max(sigma_hat, 1e-6),
                    regime_probs=regime_row,
                    account_value=account_value,
                )

            # Initialize/Update entry and tiered TP/trailing
            target_units = position
            # If a new position opens or flips direction, reset TP state
            if np.sign(target_units) != np.sign(self.prev_units):
                self.entry_price = current_price if target_units != 0.0 else None
                self.tp1_hit = False
                self.tp2_hit = False
            # Apply tiered take profit and trailing stop only if in position
            if self.entry_price is not None and target_units != 0.0:
                direction = np.sign(target_units)
                # Unrealized return
                ret = (current_price - self.entry_price) / self.entry_price * direction
                # TP thresholds in simple return units using sigma_hat
                tp1 = self.tp1_mult * sigma_hat
                tp2 = self.tp2_mult * sigma_hat
                # Trailing stop level relative to max favorable excursion
                # Estimate max favorable as current ret (monotonic update not stored here)
                trailing_stop = max(0.0, ret - self.trailing_mult * sigma_hat)
                # If trailing stop triggers, reduce or exit
                if trailing_stop <= 0.0 and ret > 0:
                    # give half back protection: reduce to half when giveback exceeds trailing mult
                    target_units = target_units * 0.5
                # Tiered take profit
                if not self.tp1_hit and ret >= tp1:
                    target_units = target_units * 0.5  # take partial profits
                    self.tp1_hit = True
                if not self.tp2_hit and ret >= tp2:
                    target_units = target_units * 0.25  # reduce further
                    self.tp2_hit = True
                # Structural fallback: if ret turns substantially negative, exit
                if ret < -self.tp1_mult * sigma_hat:
                    target_units = 0.0
            else:
                # Not in position or flat target; reset
                if target_units == 0.0:
                    self.entry_price = None
                    self.tp1_hit = False
                    self.tp2_hit = False

            # Store dynamic levels
            result_df.at[ts, "position"] = float(target_units)
            result_df.at[ts, "stop_loss_level"] = float(stop_loss_level)
            result_df.at[ts, "take_profit_level"] = float(take_profit_level)
            result_df.at[ts, "risk_mode"] = self.risk_mode

            # Simulate PnL update (in a real system, this would come from actual trading)
            if prev_price is not None:
                prev_position = float(result_df["position"].shift(1).reindex([ts]).fillna(0.0).iloc[0])
                if prev_position != 0.0 and prev_price > 0:
                    pnl = prev_position * (current_price - prev_price) / prev_price
                    self.update_position_history(prev_position, pnl, account_value=account_value)
                    # Anti-martingale: update add_count and cooldown on loss
                    if pnl > 0:
                        self.add_count = min(self.add_count + 1, self.max_adds)
                    else:
                        self.add_count = 0
                        self.cooldown_remaining = max(self.cooldown_remaining, 6)  # default 6 bars cooldown
            prev_price = current_price
            self.prev_units = float(result_df.at[ts, "position"])

        return result_df
