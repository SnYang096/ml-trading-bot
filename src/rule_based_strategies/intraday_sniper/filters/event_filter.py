"""
Economic Event-based Cool-off Filter
"""
from datetime import datetime, timedelta
from typing import List


class EventFilter:

    def __init__(self, cooloff_period: int = 300):  # 300 seconds = 5 minutes
        self.cooloff_period = timedelta(seconds=cooloff_period)
        self.event_times: List[datetime] = []

    def add_event_time(self, event_time: datetime):
        self.event_times.append(event_time)
        # Keep only recent events
        now = datetime.now()
        self.event_times = [
            et for et in self.event_times if now - et < self.cooloff_period * 2
        ]

    def is_cooling_off(self, timestamp: int) -> bool:
        dt = datetime.fromtimestamp(timestamp / 1e9)
        for event_time in self.event_times:
            if abs(dt - event_time) < self.cooloff_period:
                return True
        return False
