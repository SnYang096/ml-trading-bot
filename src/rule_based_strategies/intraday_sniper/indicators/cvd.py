import logging
from typing import Deque, Optional
from collections import deque
from nautilus_trader.core.data import Data
from nautilus_trader.indicators.base import Indicator
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide
from yin_bot.common.logger_config import setup_logger

log = setup_logger('indicator.cvd', level=logging.DEBUG)


class CVD(Indicator):
    """
    Cumulative Volume Delta (CVD) indicator following Nautilus Trader's indicator pattern.
    
    Attributes:
        cvd: total cumulative delta
        delta: latest tick delta (buy volume - sell volume)
        cvd_history: rolling window of CVD values
        delta_history: rolling window of per-tick delta values
        slope: recent CVD slope (trend)
        accel: recent CVD acceleration (momentum of slope)
    """

    def __init__(self, window_size: int = 20):
        super().__init__(params=[window_size])

        self.window_size = window_size
        self.cvd = 0.0
        self.delta = 0.0
        self.cvd_history: Deque[float] = deque(maxlen=window_size)
        self.delta_history: Deque[float] = deque(maxlen=window_size)
        self.slope = 0.0
        self.accel = 0.0

    def handle_bar(self, bar: Bar) -> None:
        """Handle bar updates - not applicable for CVD which works on ticks"""
        pass

    def update_raw(self, size: float, aggressor_side: AggressorSide) -> None:
        """
        Update the indicator with raw values.
        
        Parameters
        ----------
        size : float
            The trade size
        aggressor_side : AggressorSide
            The aggressor side (BUY/SELL)
        """
        log.debug(
            f"CVD initialized with first input: size={size}, side={aggressor_side}"
        )
        # Compute delta for this tick
        if aggressor_side == AggressorSide.BUYER:
            self.delta = size
        elif aggressor_side == AggressorSide.SELLER:
            self.delta = -size
        else:
            self.delta = 0.0

        # Update CVD
        self.cvd += self.delta

        # Append histories
        self.delta_history.append(self.delta)
        self.cvd_history.append(self.cvd)

        # Compute slope = average change over window
        if len(self.cvd_history) >= 2:
            changes = [
                self.cvd_history[i] - self.cvd_history[i - 1]
                for i in range(1, len(self.cvd_history))
            ]
            self.slope = sum(changes) / len(changes)

        # Compute accel = slope change over last few ticks
        if len(self.cvd_history) >= 3:
            recent_slopes = [
                self.cvd_history[i] - self.cvd_history[i - 1] for i in range(
                    len(self.cvd_history) - 2, len(self.cvd_history))
            ]
            self.accel = recent_slopes[-1] - recent_slopes[0]

        # Mark as initialized after first update
        if not self.initialized:
            log.debug(f"CVD has_inputs : {self.has_inputs}")
            self._set_has_inputs(True)
            if len(self.cvd_history) >= self.window_size:
                self._set_initialized(True)

    def on_trade(self, trade: TradeTick) -> None:
        """
        Process a trade tick to update CVD.
        """
        size = float(trade.size)
        log.debug(f"[CVD] Processing trade - Size: {size}, Side: {trade.aggressor_side}, Price: {trade.price}")
        self.update_raw(size, trade.aggressor_side)

    def get_cvd(self) -> float:
        return self.cvd

    def get_latest_delta(self) -> float:
        return self.delta

    def get_slope(self) -> float:
        return self.slope

    def get_accel(self) -> float:
        return self.accel

    def _reset(self) -> None:
        self.cvd = 0.0
        self.delta = 0.0
        self.cvd_history.clear()
        self.delta_history.clear()
        self.slope = 0.0
        self.accel = 0.0
        self._has_inputs = False
        self._set_initialized(False)

    @property
    def name(self) -> str:
        return f"CVD_{self.window_size}"

    @property
    def result(self) -> float:
        """Return the current CVD value"""
        return self.cvd
