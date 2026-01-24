"""
Dynamic Position Sizer based on Breakout Quality Score and ATR
"""
from typing import Optional


class PositionSizer:

    def __init__(self,
                 risk_per_trade: float = 0.01,
                 target_r_ratio: float = 3.0,
                 leverage: float = 100.0):
        self.risk_per_trade = risk_per_trade
        self.target_r_ratio = target_r_ratio
        self.leverage = leverage

    def calculate(self, equity: float, entry_price: float, atr: float, 
                  score: float, atr_multiplier: float = 1.0) -> float:
        """
        Calculate position size based on ATR and score.
        
        Args:
            equity: Account equity
            entry_price: Entry price
            atr: Average True Range value
            score: Breakout quality score (0-10)
            atr_multiplier: ATR multiplier for stop loss distance (default: 1.0)
            
        Returns:
            Position size in units
        """
        if atr is None or atr <= 0:
            return 0.0

        # Calculate stop loss distance based on ATR
        stop_distance = atr * atr_multiplier
        
        # Adjust risk per trade based on score
        # Higher scores get more risk (up to 2% of equity)
        adjusted_risk_per_trade = self.risk_per_trade
        if score >= 8.0:  # Very high score
            adjusted_risk_per_trade = min(0.02, self.risk_per_trade * 2)
        elif score >= 6.0:  # High score
            adjusted_risk_per_trade = min(0.015, self.risk_per_trade * 1.5)
            
        # Calculate risk amount based on equity, leverage, and adjusted risk percentage
        risk_amount = equity * adjusted_risk_per_trade * self.leverage
        
        # Calculate position size
        if stop_distance <= 0:
            return 0.0
            
        base_size = risk_amount / stop_distance
        
        # Additional adjustment by score (score 0-10, normalized to 0-1)
        score_adjustment = min(1.0, score / 10.0)
        adjusted_size = base_size * score_adjustment

        # Return float size, minimum 0.001 (3 decimal places)
        return max(0.001, adjusted_size)