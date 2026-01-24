"""
Time-based Trading Session Filter
"""
from datetime import datetime, time


class TimeFilter:

    def __init__(self, start_time: time, end_time: time):
        self.start_time = start_time
        self.end_time = end_time

    def is_trading_time(self, timestamp: int) -> bool:
        # Convert nanosecond timestamp to datetime
        dt = datetime.fromtimestamp(timestamp / 1e9)
        current_time = dt.time()
        return self.start_time <= current_time < self.end_time

    def is_close_time(self, timestamp: int) -> bool:
        # Convert nanosecond timestamp to datetime
        dt = datetime.fromtimestamp(timestamp / 1e9)
        current_time = dt.time()
        # Define close time as 5 minutes before end_time
        from datetime import timedelta
        close_time_start = datetime.combine(
            dt.date(), self.end_time) - timedelta(minutes=5)
        return close_time_start.time() <= current_time < self.end_time
