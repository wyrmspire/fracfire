"""
Fractal State Manager - Hierarchical market states across timeframes

Implements nested states where larger timeframes influence smaller ones:
- Day-level state (e.g., trending day, range day, breakout day)
- Hour-level states within the day state
- Minute-level states within the hour state

This creates realistic multi-timeframe market behavior.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class DayState(Enum):
    """Day-level market states (highest timeframe)"""
    TREND_DAY = "trend_day"              # Strong directional day
    RANGE_DAY = "range_day"              # Choppy, bounded day
    BREAKOUT_DAY = "breakout_day"        # Breaks out of range
    REVERSAL_DAY = "reversal_day"        # V-shaped or inverse V
    QUIET_DAY = "quiet_day"              # Low volatility, minimal movement
    VOLATILE_DAY = "volatile_day"        # High volatility, no clear direction


class HourState(Enum):
    """Hour-level market states (medium timeframe)"""
    IMPULSE = "impulse"                  # Strong directional move
    CONSOLIDATION = "consolidation"      # Tight range
    RETRACEMENT = "retracement"          # Pullback within larger trend
    CONTINUATION = "continuation"        # Resuming previous direction
    REVERSAL = "reversal"                # Changing direction
    CHOPPY = "choppy"                    # No clear direction


class MinuteState(Enum):
    """Minute-level market states (lowest timeframe) - maps to existing MarketState"""
    RALLY = "rally"
    BREAKDOWN = "breakdown"
    RANGING = "ranging"
    FLAT = "flat"
    ZOMBIE = "zombie"
    IMPULSIVE = "impulsive"
    BREAKOUT = "breakout"


@dataclass
class FractalStateConfig:
    """Configuration for how states influence each other across timeframes"""
    
    # Required fields (no defaults)
    day_state: DayState
    hour_state: HourState
    minute_state: MinuteState
    
    # Optional fields (with defaults)
    day_directional_bias: float = 0.5    # 0=down, 0.5=neutral, 1=up
    day_volatility_mult: float = 1.0
    day_trend_strength: float = 0.5      # How strongly day state influences hours
    hour_directional_bias: float = 0.5
    hour_volatility_mult: float = 1.0
    hour_trend_strength: float = 0.5     # How strongly hour state influences minutes
    hour_transition_prob: float = 0.05   # Probability of hour state change per minute
    minute_transition_prob: float = 0.1  # Probability of minute state change per bar


# Define how day states influence hour state probabilities
DAY_TO_HOUR_TRANSITIONS: Dict[DayState, Dict[HourState, float]] = {
    DayState.TREND_DAY: {
        HourState.IMPULSE: 0.3,
        HourState.CONTINUATION: 0.3,
        HourState.RETRACEMENT: 0.2,
        HourState.CONSOLIDATION: 0.15,
        HourState.REVERSAL: 0.03,
        HourState.CHOPPY: 0.02,
    },
    DayState.RANGE_DAY: {
        HourState.CONSOLIDATION: 0.35,
        HourState.CHOPPY: 0.25,
        HourState.IMPULSE: 0.15,
        HourState.RETRACEMENT: 0.15,
        HourState.CONTINUATION: 0.05,
        HourState.REVERSAL: 0.05,
    },
    DayState.BREAKOUT_DAY: {
        HourState.IMPULSE: 0.4,
        HourState.CONTINUATION: 0.25,
        HourState.CONSOLIDATION: 0.2,
        HourState.RETRACEMENT: 0.1,
        HourState.CHOPPY: 0.03,
        HourState.REVERSAL: 0.02,
    },
    DayState.REVERSAL_DAY: {
        HourState.REVERSAL: 0.3,
        HourState.IMPULSE: 0.25,
        HourState.RETRACEMENT: 0.2,
        HourState.CONSOLIDATION: 0.15,
        HourState.CONTINUATION: 0.05,
        HourState.CHOPPY: 0.05,
    },
    DayState.QUIET_DAY: {
        HourState.CONSOLIDATION: 0.5,
        HourState.CHOPPY: 0.2,
        HourState.CONTINUATION: 0.15,
        HourState.IMPULSE: 0.1,
        HourState.RETRACEMENT: 0.03,
        HourState.REVERSAL: 0.02,
    },
    DayState.VOLATILE_DAY: {
        HourState.CHOPPY: 0.3,
        HourState.IMPULSE: 0.25,
        HourState.REVERSAL: 0.2,
        HourState.RETRACEMENT: 0.15,
        HourState.CONSOLIDATION: 0.05,
        HourState.CONTINUATION: 0.05,
    },
}


# Define how hour states influence minute state probabilities
HOUR_TO_MINUTE_TRANSITIONS: Dict[HourState, Dict[MinuteState, float]] = {
    HourState.IMPULSE: {
        MinuteState.RALLY: 0.4,
        MinuteState.BREAKOUT: 0.2,
        MinuteState.IMPULSIVE: 0.2,
        MinuteState.ZOMBIE: 0.1,
        MinuteState.RANGING: 0.05,
        MinuteState.FLAT: 0.03,
        MinuteState.BREAKDOWN: 0.02,
    },
    HourState.CONSOLIDATION: {
        MinuteState.RANGING: 0.4,
        MinuteState.FLAT: 0.3,
        MinuteState.ZOMBIE: 0.15,
        MinuteState.RALLY: 0.05,
        MinuteState.BREAKDOWN: 0.05,
        MinuteState.IMPULSIVE: 0.03,
        MinuteState.BREAKOUT: 0.02,
    },
    HourState.RETRACEMENT: {
        MinuteState.BREAKDOWN: 0.3,
        MinuteState.RANGING: 0.25,
        MinuteState.ZOMBIE: 0.2,
        MinuteState.FLAT: 0.15,
        MinuteState.RALLY: 0.05,
        MinuteState.IMPULSIVE: 0.03,
        MinuteState.BREAKOUT: 0.02,
    },
    HourState.CONTINUATION: {
        MinuteState.ZOMBIE: 0.35,
        MinuteState.RALLY: 0.25,
        MinuteState.RANGING: 0.2,
        MinuteState.FLAT: 0.1,
        MinuteState.IMPULSIVE: 0.05,
        MinuteState.BREAKOUT: 0.03,
        MinuteState.BREAKDOWN: 0.02,
    },
    HourState.REVERSAL: {
        MinuteState.IMPULSIVE: 0.3,
        MinuteState.BREAKDOWN: 0.25,
        MinuteState.RALLY: 0.2,
        MinuteState.RANGING: 0.15,
        MinuteState.BREAKOUT: 0.05,
        MinuteState.ZOMBIE: 0.03,
        MinuteState.FLAT: 0.02,
    },
    HourState.CHOPPY: {
        MinuteState.RANGING: 0.35,
        MinuteState.IMPULSIVE: 0.25,
        MinuteState.RALLY: 0.15,
        MinuteState.BREAKDOWN: 0.15,
        MinuteState.FLAT: 0.05,
        MinuteState.ZOMBIE: 0.03,
        MinuteState.BREAKOUT: 0.02,
    },
}


# Day state characteristics
DAY_STATE_PARAMS = {
    DayState.TREND_DAY: {
        'directional_bias': 0.54,  # Increased from 0.53 to balance down days
        'volatility_mult': 1.2,
        'trend_strength': 0.6,
        'mean_reversion_strength': 0.05, # Low gravity
    },
    DayState.RANGE_DAY: {
        'directional_bias': 0.5,
        'volatility_mult': 0.9,
        'trend_strength': 0.3,
        'mean_reversion_strength': 0.6, # High gravity
    },
    DayState.BREAKOUT_DAY: {
        'directional_bias': 0.7,
        'volatility_mult': 1.6,
        'trend_strength': 0.85,
        'mean_reversion_strength': 0.1,
    },
    DayState.REVERSAL_DAY: {
        'directional_bias': 0.5,  # Changes during day
        'volatility_mult': 1.4,
        'trend_strength': 0.6,
        'mean_reversion_strength': 0.4,
    },
    DayState.QUIET_DAY: {
        'directional_bias': 0.5,
        'volatility_mult': 0.5,
        'trend_strength': 0.2,
        'mean_reversion_strength': 0.3,
    },
    DayState.VOLATILE_DAY: {
        'directional_bias': 0.5,
        'volatility_mult': 2.0,
        'trend_strength': 0.4,
        'mean_reversion_strength': 0.2,
    },
}


# Hour state characteristics
HOUR_STATE_PARAMS = {
    HourState.IMPULSE: {
        'directional_bias': 0.6,   # Reduced from 0.7
        'volatility_mult': 1.5,
        'trend_strength': 0.7,
        'mean_reversion_strength': 0.0, # No gravity in impulse
    },
    HourState.CONSOLIDATION: {
        'directional_bias': 0.5,
        'volatility_mult': 0.6,
        'trend_strength': 0.3,
        'mean_reversion_strength': 0.5,
    },
    HourState.RETRACEMENT: {
        'directional_bias': 0.35,  # Against main trend
        'volatility_mult': 1.0,
        'trend_strength': 0.6,
        'mean_reversion_strength': 0.3,
    },
    HourState.CONTINUATION: {
        'directional_bias': 0.55,  # Reduced from 0.6
        'volatility_mult': 1.1,
        'trend_strength': 0.6,
        'mean_reversion_strength': 0.1,
    },
    HourState.REVERSAL: {
        'directional_bias': 0.5,  # Flips during hour
        'volatility_mult': 1.4,
        'trend_strength': 0.7,
        'mean_reversion_strength': 0.2,
    },
    HourState.CHOPPY: {
        'directional_bias': 0.5,
        'volatility_mult': 1.2,
        'trend_strength': 0.2,
        'mean_reversion_strength': 0.4,
    },
}

class FractalStateManager:
    """Manages hierarchical day/hour/minute market states.

    NOTE: This class is currently a stub of the original design.
    For now we disable its use so that the core price generator
    can import cleanly without relying on unfinished logic.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Minimal initializer to keep imports working.
        self.day_state = None
        self.hour_state = None
        self.minute_state = None
        self.current_time = None
        self.rng = np.random.default_rng()

    def initialize_day(self, *args, **kwargs):
        """Placeholder for future full implementation.

        For now, simply set a default day state so callers that
        expect a return value do not fail.
        """
        self.day_state = DayState.RANGE_DAY
        return self.day_state
    
    def _transition_hour_state(self) -> HourState:
        """Transition to a new hour state based on day state and time of day"""
        if self.day_state is None:
            return HourState.CONSOLIDATION
            
        # Base probabilities from Day State
        probs = DAY_TO_HOUR_TRANSITIONS[self.day_state].copy()
        
        # Adjust based on Time of Day (if available)
        if self.current_time:
            hour = self.current_time.hour
            
            # RTH Open (9:30 - 11:30) -> Bias towards Impulse/Volatile
            if 9 <= hour < 11:
                if HourState.IMPULSE in probs: probs[HourState.IMPULSE] *= 2.0
                if HourState.CHOPPY in probs: probs[HourState.CHOPPY] *= 1.5
                if HourState.CONSOLIDATION in probs: probs[HourState.CONSOLIDATION] *= 0.5
                
            # Midday (11:30 - 14:00) -> Bias towards Consolidation/Chop
            elif 11 <= hour < 14:
                if HourState.CONSOLIDATION in probs: probs[HourState.CONSOLIDATION] *= 2.0
                if HourState.CHOPPY in probs: probs[HourState.CHOPPY] *= 1.5
                if HourState.IMPULSE in probs: probs[HourState.IMPULSE] *= 0.3
                
            # Close (14:00 - 16:00) -> Bias towards Impulse/Reversal
            elif 14 <= hour <= 16:
                if HourState.IMPULSE in probs: probs[HourState.IMPULSE] *= 1.5
                if HourState.REVERSAL in probs: probs[HourState.REVERSAL] *= 1.5
                if HourState.CONSOLIDATION in probs: probs[HourState.CONSOLIDATION] *= 0.5
        
        # Normalize probabilities
        total = sum(probs.values())
        states = list(probs.keys())
        probabilities = [p / total for p in probs.values()]
        
        # Choose new state
        new_state = self.rng.choice(states, p=probabilities)
        return new_state
    
    def _transition_minute_state(self) -> MinuteState:
        """Transition to a new minute state based on hour state"""
        if self.hour_state is None:
            return MinuteState.RANGING
        
        # Get transition probabilities from hour state
        probs = HOUR_TO_MINUTE_TRANSITIONS[self.hour_state]
        states = list(probs.keys())
        probabilities = list(probs.values())
        
        # Choose new state
        new_state = self.rng.choice(states, p=probabilities)
        return new_state
    
    def update(self, timestamp: datetime, current_price: Optional[float] = None, force_hour_transition: bool = False) -> Tuple[DayState, HourState, MinuteState]:
        """
        Update states for a new bar.
        
        Args:
            timestamp: Current bar timestamp
            current_price: Current price level for macro gravity
            force_hour_transition: Force an hour state transition
        
        Returns:
            (day_state, hour_state, minute_state)
        """
        self.current_time = timestamp
        
        if self.day_state is None:
            self.initialize_day(current_price=current_price)
            # Initialize daily open price tracking
            self.daily_open_price = current_price if current_price else 5000.0
        
        # Track hour boundaries
        if self.current_hour_start is None:
            self.current_hour_start = timestamp.replace(minute=0, second=0, microsecond=0)
            self.bars_in_current_hour = 0
            
        # Check for Day Transition
        if self.current_hour_start.date() != timestamp.date():
             self.initialize_day(current_price=current_price)
             self.daily_open_price = current_price if current_price else 5000.0
             self.current_hour_start = timestamp.replace(minute=0, second=0, microsecond=0)
             self.bars_in_current_hour = 0
        
        # Check if we've entered a new hour
        current_hour = timestamp.replace(minute=0, second=0, microsecond=0)
        if current_hour != self.current_hour_start or force_hour_transition:
            if self.rng.random() < 0.4 or force_hour_transition:
                self.hour_state = self._transition_hour_state()
                self.hour_params = HOUR_STATE_PARAMS[self.hour_state].copy()
            self.current_hour_start = current_hour
            self.bars_in_current_hour = 0
        
        # Check for minute state transition
        if self.rng.random() < 0.1:
            self.minute_state = self._transition_minute_state()
            
        self.bars_in_current_hour += 1
        return self.day_state, self.hour_state, self.minute_state

    def get_combined_parameters(self, current_price: Optional[float] = None) -> Dict:
        """
        Get combined parameters from all timeframe states.
        
        Args:
            current_price: Current price for volatility governor checks
            
        Returns:
            Dictionary with combined directional_bias, volatility_mult, trend_strength
        """
        # Combine day and hour parameters
        day_bias = self.day_params.get('directional_bias', 0.5)
        hour_bias = self.hour_params.get('directional_bias', 0.5)
        
        # Apply Day Direction
        if self.day_direction == -1:
            day_bias = 0.5 - (day_bias - 0.5)
            hour_bias = 0.5 - (hour_bias - 0.5)
        
        # Weight by trend strength
        day_strength = self.day_params.get('trend_strength', 0.5)
        hour_strength = self.hour_params.get('trend_strength', 0.5)
        
        # Combined bias (weighted average)
        w_day = day_strength * 0.4
        w_hour = hour_strength * 0.6
        total_weight = w_day + w_hour
        
        if total_weight == 0:
            combined_bias = 0.5
        else:
            combined_bias = (day_bias * w_day + hour_bias * w_hour) / total_weight
            
        # --- DIRECTIONAL GOVERNOR (Probabilistic) ---
        # Allow price to run in the direction of the trend, but cap extreme moves.
        if current_price is not None and hasattr(self, 'daily_open_price'):
            daily_move = current_price - self.daily_open_price
            
            # Use randomized target range for this day (default to 100 if not set)
            target_range = getattr(self, 'daily_target_range', 100.0)
            governor_strength = getattr(self, 'daily_governor_strength', 0.3)
            
            # Calculate deviation relative to Day Direction
            if self.day_direction == 1: # Uptrend Day
                if daily_move > target_range:
                    # Overextended UP: Pull back gently
                    excess = daily_move - target_range
                    governor_bias = -np.tanh(excess / 50.0) * governor_strength
                elif daily_move < -30:
                    # Failed to hold trend (dropped below open): Push back UP strongly
                    deficit = abs(daily_move) - 30
                    governor_bias = np.tanh(deficit / 30.0) * 0.4
                else:
                    # In the "Good Zone": Slight tailwind
                    governor_bias = 0.05
                    
            elif self.day_direction == -1: # Downtrend Day
                if daily_move < -target_range:
                    # Overextended DOWN: Pull back gently
                    excess = abs(daily_move) - target_range
                    governor_bias = np.tanh(excess / 50.0) * governor_strength
                elif daily_move > 30:
                    # Failed to hold trend (rallied above open): Push back DOWN strongly
                    deficit = daily_move - 30
                    governor_bias = -np.tanh(deficit / 30.0) * 0.4
                else:
                    # In the "Good Zone": Slight tailwind
                    governor_bias = -0.05
            
            else: # Range Day (0)
                # Mean Revert to Open
                governor_bias = -np.tanh(daily_move / 50.0) * 0.3

            # Apply to combined_bias
            combined_bias += governor_bias
            
            # Clamp
            combined_bias = max(0.1, min(0.9, combined_bias))
        # ---------------------------
        
        # Combined volatility (multiplicative)
        combined_volatility = (
            self.day_params.get('volatility_mult', 1.0) *
            self.hour_params.get('volatility_mult', 1.0)
        )
        
        # Combined trend strength (average)
        combined_trend_strength = (
            day_strength * 0.4 +
            hour_strength * 0.6
        )
        
        # Combined mean reversion (max)
        combined_mean_reversion = max(
            self.day_params.get('mean_reversion_strength', 0.0),
            self.hour_params.get('mean_reversion_strength', 0.0)
        )
        
        return {
            'directional_bias': combined_bias,
            'volatility_mult': combined_volatility,
            'trend_strength': combined_trend_strength,
            'mean_reversion_strength': combined_mean_reversion,
            'day_state': self.day_state.value if self.day_state else None,
            'hour_state': self.hour_state.value if self.hour_state else None,
            'minute_state': self.minute_state.value if self.minute_state else None,
            'day_direction': self.day_direction
        }
