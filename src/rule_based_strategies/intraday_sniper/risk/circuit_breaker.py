"""
Circuit Breaker for Consecutive Losses
"""
import datetime
from typing import List, Optional
from nautilus_trader.model.events.order import OrderFilled


class CircuitBreaker:

    def __init__(self, max_losses: int = 3, cooloff_minutes: int = 60):
        self.max_losses = max_losses
        self.cooloff_minutes = cooloff_minutes
        self.loss_streak = 0
        self.cooloff_until: Optional[datetime] = None
        self.recent_losses: List[OrderFilled] = []

    def on_fill(self, fill: OrderFilled):
        # Simplified: assume we can determine if it was a loss
        # In practice, you'd need to track PnL at close
        pass

    def is_active(self) -> bool:
        from datetime import datetime, timedelta
        if self.cooloff_until and datetime.now() < self.cooloff_until:
            return True
        return False

    def trigger(self):
        from datetime import datetime, timedelta
        self.cooloff_until = datetime.now() + timedelta(
            minutes=self.cooloff_minutes)
        self.loss_streak = 0
        print("Circuit breaker activated due to consecutive losses.")
