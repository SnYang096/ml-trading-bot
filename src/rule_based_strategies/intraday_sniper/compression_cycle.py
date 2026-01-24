"""
Compression Cycle - Manages the lifecycle of compression-breakout events with enhanced time dependency handling
"""
from typing import Optional
from dataclasses import dataclass
from nautilus_trader.model.data import Bar


@dataclass
class CompressionCycle:
    """
    Represents a complete compression-breakout cycle with time-dependent state management
    This cycle is created when compression ends, tracking the breakout that follows
    """
    
    # Compression period information (when compression ended)
    start_idx: int                    # Bar index when compression ended (cycle tracking starts)
    end_idx: int                      # Bar index when compression ended (same as start_idx for this approach)
    high: float                      # Compression zone high
    low: float                       # Compression zone low
    duration: int                    # Duration of compression in bars (0 for this approach)
    volume_compression_ratio: float  # Volume compression strength
    atr_compression_ratio: float     # ATR compression strength
    
    # Breakout information
    breakout_idx: Optional[int]      # Bar index when breakout occurred
    breakout_direction: Optional[str]  # "up" or "down"
    confirmed: bool = False          # Whether breakout is confirmed
    
    # State management
    entry_triggered: bool = False    # Whether an entry signal has been triggered
    expiry: Optional[int] = None     # Expiration index (breakout_idx + window)
    
    def __post_init__(self):
        """Initialize expiry if not set"""
        if self.expiry is None and self.breakout_idx is not None:
            self.expiry = self.breakout_idx + 3  # Default 3-bar window after breakout
        elif self.expiry is None:
            self.expiry = self.start_idx + 10  # Default 10-bar window after compression ends
    
    def is_active(self, current_idx: int) -> bool:
        """
        Check if this compression cycle is still active for entry signals
        
        Args:
            current_idx: Current bar index
            
        Returns:
            bool: True if cycle is active and hasn't triggered entry yet
        """
        # Must have valid expiry and not have triggered entry
        if self.expiry is None or self.entry_triggered:
            return False
            
        # Must be within expiration window
        return current_idx <= self.expiry if self.expiry is not None else False
    
    def set_breakout(self, idx: int, direction: str):
        """
        Record breakout information for this cycle
        
        Args:
            idx: Bar index when breakout occurred
            direction: Breakout direction ("up" or "down")
        """
        # Only set breakout if not already set and occurs after compression ends
        if self.breakout_idx is None and idx >= self.end_idx:
            self.breakout_idx = idx
            self.breakout_direction = direction
            # Set expiry window (3 bars after breakout by default)
            self.expiry = idx + 3
    
    def bars_since_breakout(self, current_idx: int) -> Optional[int]:
        """
        Calculate number of bars since breakout occurred
        
        Args:
            current_idx: Current bar index
            
        Returns:
            int: Number of bars since breakout, or None if no breakout
        """
        if self.breakout_idx is None:
            return None
        return current_idx - self.breakout_idx
    
    def is_first_breakout_bar(self, current_idx: int) -> bool:
        """
        Check if this is the first bar after breakout
        
        Args:
            current_idx: Current bar index
            
        Returns:
            bool: True if this is the first bar after breakout
        """
        bars_since = self.bars_since_breakout(current_idx)
        return bars_since == 0 if bars_since is not None else False
    
    def is_second_breakout_bar(self, current_idx: int) -> bool:
        """
        Check if this is the second bar after breakout
        
        Args:
            current_idx: Current bar index
            
        Returns:
            bool: True if this is the second bar after breakout
        """
        bars_since = self.bars_since_breakout(current_idx)
        return bars_since == 1 if bars_since is not None else False
    
    def is_within_breakout_window(self, current_idx: int, window: int = 3) -> bool:
        """
        Check if we're within the breakout confirmation window
        
        Args:
            current_idx: Current bar index
            window: Number of bars to consider valid after breakout
            
        Returns:
            bool: True if within breakout window
        """
        bars_since = self.bars_since_breakout(current_idx)
        if bars_since is None:
            return False
        return 0 <= bars_since < window
    
    def get_signal_weight(self, current_idx: int) -> float:
        """
        Get signal weight based on time since breakout (exponential decay)
        
        Args:
            current_idx: Current bar index
            
        Returns:
            float: Signal weight (1.0 for first bar, decaying thereafter)
        """
        bars_since = self.bars_since_breakout(current_idx)
        if bars_since is None:
            return 0.0
            
        # Exponential decay - weight decreases as time passes
        decay_factor = 0.7
        return max(0.0, decay_factor ** bars_since)
    
    def can_trigger_entry(self, current_idx: int) -> bool:
        """
        Check if an entry signal can still be triggered for this cycle
        
        Args:
            current_idx: Current bar index
            
        Returns:
            bool: True if entry can still be triggered
        """
        # Must be active and not already triggered
        if not self.is_active(current_idx) or self.entry_triggered:
            return False
            
        # Must be within breakout window
        return self.is_within_breakout_window(current_idx)
    
    def reset_for_new_opportunity(self):
        """
        Reset the cycle for a new breakout opportunity
        This is used when we want to allow multiple entries in the same compression zone
        """
        self.entry_triggered = False
        # Don't reset breakout information as it's still valid
        # Update expiry to allow new entries within the window
        if self.breakout_idx is not None:
            self.expiry = self.breakout_idx + 3