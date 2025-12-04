# Code Dump: 01_core_generator

## File: src/core/generator/__init__.py
```python
"""Generator module: synthetic price generation."""

from .engine import (
    PriceGenerator,
    PhysicsConfig,
    MarketState,
    Session,
    StateConfig,
    SessionConfig,
    DayOfWeekConfig,
    STATE_CONFIGS,
    SESSION_CONFIGS,
    DOW_CONFIGS,
)

from .states import (
    FractalStateManager,
    DayState,
    HourState,
    MinuteState,
    FractalStateConfig,
)

from .custom_states import (
    CUSTOM_STATES,
    get_custom_state,
    list_custom_states,
    EXTREME_VOLATILITY,
    DIRECTIONAL,
    SESSION_SPECIFIC,
    LOW_VOLATILITY,
)

__all__ = [
    'PriceGenerator',
    'PhysicsConfig',
    'MarketState',
    'Session',
    'StateConfig',
    'SessionConfig',
    'DayOfWeekConfig',
    'STATE_CONFIGS',
    'SESSION_CONFIGS',
    'DOW_CONFIGS',
    'FractalStateManager',
    'DayState',
    'HourState',
    'MinuteState',
    'FractalStateConfig',
    'CUSTOM_STATES',
    'get_custom_state',
    'list_custom_states',
    'EXTREME_VOLATILITY',
    'DIRECTIONAL',
    'SESSION_SPECIFIC',
    'LOW_VOLATILITY',
]

```

---

## File: src/core/generator/engine.py
```python
"""
MES Price Generator with Configurable Market States

This module generates synthetic 1-minute MES candles with realistic tick-based price action.
All parameters are exposed as "knobs" for testing different market conditions.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum


class MarketState(Enum):
    """Market behavior states that affect price generation"""
    RANGING = "ranging"          # Tight, choppy, mean-reverting
    FLAT = "flat"                # Very low volatility, minimal movement
    ZOMBIE = "zombie"            # Slow grind in one direction
    RALLY = "rally"              # Strong directional move
    IMPULSIVE = "impulsive"      # High volatility, large swings
    BREAKDOWN = "breakdown"      # Sharp downward move
    BREAKOUT = "breakout"        # Sharp upward move


class Session(Enum):
    """Trading sessions with different characteristics"""
    ASIAN = "asian"              # 18:00-03:00 CT (typically lower volume)
    LONDON = "london"            # 03:00-08:30 CT (increasing activity)
    PREMARKET = "premarket"      # 08:30-09:30 CT (building momentum)
    RTH = "rth"                  # 09:30-16:00 CT (Regular Trading Hours - highest volume)
    AFTERHOURS = "afterhours"    # 16:00-18:00 CT (declining activity)


@dataclass
class PhysicsConfig:
    """Global physics parameters for the generator (The 'Knobs')"""
    # 1. Volatility / Heat
    # Tuned defaults chosen to match real MES behavior when
    # used via `scripts/compare_real_vs_generator.py`.
    base_volatility: float = 2.0            # Overall volatility multiplier
    avg_ticks_per_bar: float = 8.0          # Base activity level
    
    # 2. Daily Structure (Golden Truth Targets)
    daily_range_mean: float = 120.0         # Target daily range (points)
    daily_range_std: float = 60.0           # Standard deviation of range
    
    # 3. Runner Days (Trend Days)
    runner_prob: float = 0.20               # Probability of Runner Day
    runner_target_mult: float = 4.0         # Multiplier for runner range (e.g. 4x mean)
    
    # 4. Macro Gravity (Drift Control)
    macro_gravity_threshold: float = 5000.0 # Points deviation before gravity kicks in
    macro_gravity_strength: float = 0.15    # Strength of pull back to center (less mean reversion)
    
    # 5. Micro Physics
    wick_probability: float = 0.20          # Probability of extended wicks
    wick_extension_avg: float = 2.0         # Average wick extension in ticks


@dataclass
class StateConfig:
    """Configuration for a specific market state"""
    name: str
    
    # Tick movement parameters
    avg_ticks_per_bar: float = 8.0          # Average number of ticks per 1m bar
    ticks_per_bar_std: float = 4.0          # Std dev of ticks per bar
    
    # Directional bias
    up_probability: float = 0.5             # Probability of upward tick (0.5 = neutral)
    trend_persistence: float = 0.5          # How likely to continue previous direction (0-1)
    
    # Tick size distribution
    avg_tick_size: float = 1.0              # Average ticks per move (1.0 = single tick)
    tick_size_std: float = 0.5              # Std dev of tick size
    max_tick_jump: int = 8                  # Maximum ticks in single move
    
    # Volatility
    volatility_multiplier: float = 1.0      # Overall volatility scaling
    
    # Wick characteristics
    wick_probability: float = 0.15          # Reduced from 0.3
    wick_extension_avg: float = 1.5         # Reduced from 2.0

    # Liquidity Physics
    mean_reversion_strength: float = 0.0    # Strength of pull towards key levels (0-1)


# Predefined state configurations
STATE_CONFIGS = {
    MarketState.RANGING: StateConfig(
        name="ranging",
        avg_ticks_per_bar=4.0,
        ticks_per_bar_std=2.0,
        up_probability=0.5,
        trend_persistence=0.4,
        avg_tick_size=1.0,
        tick_size_std=0.5,
        max_tick_jump=3,
        volatility_multiplier=1.0,
        wick_probability=0.2,
        wick_extension_avg=1.5,
    ),
    MarketState.ZOMBIE: StateConfig(
        name="zombie",
        avg_ticks_per_bar=3.0,
        ticks_per_bar_std=1.5,
        up_probability=0.55,
        trend_persistence=0.8,
        avg_tick_size=1.0,
        tick_size_std=0.2,
        max_tick_jump=2,
        volatility_multiplier=0.6,
        wick_probability=0.1,
        wick_extension_avg=1.0,
    ),
    MarketState.RALLY: StateConfig(
        name="rally",
        avg_ticks_per_bar=6.0,
        ticks_per_bar_std=3.0,
        up_probability=0.6,
        trend_persistence=0.7,
        avg_tick_size=1.2,
        tick_size_std=0.5,
        max_tick_jump=4,
        volatility_multiplier=1.2,
        wick_probability=0.15,
        wick_extension_avg=1.5,
    ),
    MarketState.IMPULSIVE: StateConfig(
        name="impulsive",
        avg_ticks_per_bar=10.0,
        ticks_per_bar_std=5.0,
        up_probability=0.5,
        trend_persistence=0.6,
        avg_tick_size=1.5,
        tick_size_std=0.8,
        max_tick_jump=6,
        volatility_multiplier=1.5,
        wick_probability=0.25,
        wick_extension_avg=2.0,
    ),
    MarketState.BREAKDOWN: StateConfig(
        name="breakdown",
        avg_ticks_per_bar=12.0,
        ticks_per_bar_std=6.0,
        up_probability=0.25,
        trend_persistence=0.85,
        avg_tick_size=1.5,
        tick_size_std=0.8,
        max_tick_jump=6,
        volatility_multiplier=1.8,
        wick_probability=0.25,
        wick_extension_avg=2.5,
    ),
    MarketState.BREAKOUT: StateConfig(
        name="breakout",
        avg_ticks_per_bar=12.0,
        ticks_per_bar_std=6.0,
        up_probability=0.75,
        trend_persistence=0.85,
        avg_tick_size=1.5,
        tick_size_std=0.8,
        max_tick_jump=6,
        volatility_multiplier=1.8,
        wick_probability=0.25,
        wick_extension_avg=2.5,
    ),
    # Fallback config for FLAT state (very low volatility / movement)
    MarketState.FLAT: StateConfig(
        name="flat",
        avg_ticks_per_bar=2.0,
        ticks_per_bar_std=1.0,
        up_probability=0.5,
        trend_persistence=0.2,
        avg_tick_size=0.5,
        tick_size_std=0.2,
        max_tick_jump=2,
        volatility_multiplier=0.3,
        wick_probability=0.05,
        wick_extension_avg=1.0,
    ),
}


@dataclass
class SessionConfig:
    """Configuration for session-based effects"""
    name: str
    volume_multiplier: float = 1.0          # Relative volume level
    volatility_multiplier: float = 1.0      # Relative volatility
    state_transition_prob: float = 0.05     # Probability of state change per bar


SESSION_CONFIGS = {
    Session.ASIAN: SessionConfig(
        name="asian",
        volume_multiplier=0.4,
        volatility_multiplier=0.6,
        state_transition_prob=0.02,
    ),
    Session.LONDON: SessionConfig(
        name="london",
        volume_multiplier=0.8,
        volatility_multiplier=1.1,
        state_transition_prob=0.05,
    ),
    Session.PREMARKET: SessionConfig(
        name="premarket",
        volume_multiplier=0.6,
        volatility_multiplier=0.9,
        state_transition_prob=0.08,
    ),
    Session.RTH: SessionConfig(
        name="rth",
        volume_multiplier=1.5,
        volatility_multiplier=1.3,
        state_transition_prob=0.06,
    ),
    Session.AFTERHOURS: SessionConfig(
        name="afterhours",
        volume_multiplier=0.5,
        volatility_multiplier=0.7,
        state_transition_prob=0.03,
    ),
}


@dataclass
class DayOfWeekConfig:
    """Day of week effects"""
    name: str
    volume_multiplier: float = 1.0
    volatility_multiplier: float = 1.0


DOW_CONFIGS = {
    0: DayOfWeekConfig("monday", volume_multiplier=1.1, volatility_multiplier=1.2),
    1: DayOfWeekConfig("tuesday", volume_multiplier=1.0, volatility_multiplier=1.0),
    2: DayOfWeekConfig("wednesday", volume_multiplier=1.0, volatility_multiplier=1.0),
    3: DayOfWeekConfig("thursday", volume_multiplier=1.0, volatility_multiplier=1.0),
    4: DayOfWeekConfig("friday", volume_multiplier=1.2, volatility_multiplier=1.1),
    5: DayOfWeekConfig("saturday", volume_multiplier=0.3, volatility_multiplier=0.5),
    6: DayOfWeekConfig("sunday", volume_multiplier=0.4, volatility_multiplier=0.6),
}


class PriceGenerator:
    """
    Generate synthetic MES price bars with configurable market dynamics.
    
    All ticks are 0.25 (MES tick size).
    """
    
    TICK_SIZE = 0.25
    
    def __init__(
        self,
        initial_price: float = 5000.0,
        seed: Optional[int] = None,
        physics_config: Optional[PhysicsConfig] = None,
    ):
        """
        Initialize the price generator.
        
        Args:
            initial_price: Starting price level
            seed: Random seed for reproducibility
            physics_config: Physics configuration (knobs)
        """
        self.initial_price = initial_price
        self.current_price = initial_price
        self.last_direction = 0  # -1, 0, or 1
        self.prev_close_ticks = int(initial_price / self.TICK_SIZE)  # Track for delta_ticks
        
        # Physics Config
        self.physics = physics_config if physics_config else PhysicsConfig()
        
        # Liquidity Levels
        self.prev_day_close = initial_price
        self.prev_week_close = initial_price
        
        if seed is not None:
            np.random.seed(seed)
        
        self.rng = np.random.default_rng(seed)
    
    def get_session(self, dt: datetime) -> Session:
        """Determine trading session based on time (Chicago time)"""
        hour = dt.hour
        
        if 18 <= hour or hour < 3:
            return Session.ASIAN
        elif 3 <= hour < 8 or (hour == 8 and dt.minute < 30):
            return Session.LONDON
        elif (hour == 8 and dt.minute >= 30) or hour == 9 and dt.minute < 30:
            return Session.PREMARKET
        elif (hour == 9 and dt.minute >= 30) or (10 <= hour < 15) or (hour == 15 and dt.minute <= 15):
            return Session.RTH
        else:
            return Session.AFTERHOURS
    
    def generate_tick_movement(
        self,
        state_config: StateConfig,
        session_config: SessionConfig,
        dow_config: DayOfWeekConfig,
    ) -> Tuple[int, int]:
        """
        Generate a single tick movement.
        
        Returns:
            (direction, num_ticks) where direction is -1 or 1, num_ticks is the size
        """
        # Determine direction
        if self.rng.random() < state_config.trend_persistence and self.last_direction != 0:
            # Continue previous direction
            direction = self.last_direction
        else:
            # Calculate Gravity Bias
            # Pull towards prev_day_close
            # dist_to_pdc = self.prev_day_close - self.current_price
            
            # Sigmoid-like gravity: stronger pull when further away, but capped
            # Scale distance: 100 points (400 ticks) = strong pull
            # norm_dist = dist_to_pdc / 100.0 
            # gravity_factor = np.tanh(norm_dist) * 0.5 # -0.5 to 0.5
            
            # Apply strength
            # gravity_bias = gravity_factor * state_config.mean_reversion_strength
            
            # Adjust up_probability
            # adjusted_prob = state_config.up_probability + gravity_bias
            # adjusted_prob = max(0.05, min(0.95, adjusted_prob)) # Clamp
            
            # New direction based on adjusted bias
            direction = 1 if self.rng.random() < state_config.up_probability else -1
        
        # Determine tick size
        volatility = (
            state_config.volatility_multiplier *
            session_config.volatility_multiplier *
            dow_config.volatility_multiplier
        )
        
        tick_size = max(
            1,
            int(self.rng.normal(
                state_config.avg_tick_size * volatility,
                state_config.tick_size_std * volatility
            ))
        )
        tick_size = min(tick_size, state_config.max_tick_jump)
        
        self.last_direction = direction
        return direction, tick_size
    
    def generate_bar(
        self,
        timestamp: datetime,
        state: MarketState = MarketState.RANGING,
        custom_state_config: Optional[StateConfig] = None,
    ) -> dict:
        """
        Generate a single 1-minute OHLCV bar.
        
        Args:
            timestamp: Bar timestamp
            state: Market state to use
            custom_state_config: Optional custom state configuration (overrides state)
        
        Returns:
            Dictionary with keys: time, open, high, low, close, volume
        """
        # Get configurations
        state_config = custom_state_config or STATE_CONFIGS[state]
        session = self.get_session(timestamp)
        session_config = SESSION_CONFIGS[session]
        dow_config = DOW_CONFIGS[timestamp.weekday()]
        
        # Determine number of ticks for this bar
        num_ticks = max(
            1,
            int(self.rng.normal(
                state_config.avg_ticks_per_bar,
                state_config.ticks_per_bar_std
            ))
        )
        
        # Generate tick-by-tick price action
        open_price = self.current_price
        high_price = open_price
        low_price = open_price
        current = open_price
        
        for _ in range(num_ticks):
            direction, tick_size = self.generate_tick_movement(
                state_config, session_config, dow_config
            )
            
            # Move price
            price_change = direction * tick_size * self.TICK_SIZE
            current += price_change
            
            # Update high/low
            high_price = max(high_price, current)
            low_price = min(low_price, current)
        
        close_price = current
        
        # Add wicks (extended highs/lows that don't close there)
        if self.rng.random() < state_config.wick_probability:
            # Upper wick
            wick_ticks = max(1, int(self.rng.normal(
                state_config.wick_extension_avg,
                state_config.wick_extension_avg * 0.5
            )))
            high_price += wick_ticks * self.TICK_SIZE
        
        if self.rng.random() < state_config.wick_probability:
            # Lower wick
            wick_ticks = max(1, int(self.rng.normal(
                state_config.wick_extension_avg,
                state_config.wick_extension_avg * 0.5
            )))
            low_price -= wick_ticks * self.TICK_SIZE
        
        # Generate volume (scaled by session and day)
        base_volume = max(
            10,
            int(self.rng.normal(100, 50))
        )
        volume = int(
            base_volume *
            session_config.volume_multiplier *
            dow_config.volume_multiplier *
            state_config.volatility_multiplier
        )
        
        # Update current price for next bar
        self.current_price = close_price
        
        # Round prices to tick size
        open_price = round(open_price / self.TICK_SIZE) * self.TICK_SIZE
        high_price = round(high_price / self.TICK_SIZE) * self.TICK_SIZE
        low_price = round(low_price / self.TICK_SIZE) * self.TICK_SIZE
        close_price = round(close_price / self.TICK_SIZE) * self.TICK_SIZE
        
        # Convert to tick units (integer ticks from zero)
        open_ticks = int(open_price / self.TICK_SIZE)
        high_ticks = int(high_price / self.TICK_SIZE)
        low_ticks = int(low_price / self.TICK_SIZE)
        close_ticks = int(close_price / self.TICK_SIZE)
        
        # Compute tick-based deltas and features
        delta_ticks = close_ticks - self.prev_close_ticks
        range_ticks = high_ticks - low_ticks
        body_ticks = abs(close_ticks - open_ticks)
        
        # Wick calculations in ticks
        upper_body = max(open_ticks, close_ticks)
        lower_body = min(open_ticks, close_ticks)
        upper_wick_ticks = high_ticks - upper_body
        lower_wick_ticks = lower_body - low_ticks
        
        # Update prev_close for next bar
        self.prev_close_ticks = close_ticks
        
        return {
            'time': timestamp,
            # Price columns (floats)
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            # Tick columns (integers) - ML-friendly
            'open_ticks': open_ticks,
            'high_ticks': high_ticks,
            'low_ticks': low_ticks,
            'close_ticks': close_ticks,
            'delta_ticks': delta_ticks,
            'range_ticks': range_ticks,
            'body_ticks': body_ticks,
            'upper_wick_ticks': upper_wick_ticks,
            'lower_wick_ticks': lower_wick_ticks,
            # State labels
            'state': state.value,
            'session': session.value,
        }
    
    def generate_day(
        self,
        start_date: datetime,
        state_sequence: Optional[List[Tuple[int, MarketState]]] = None,
        auto_transition: bool = True,
        segment_length: Optional[int] = None,
        macro_regime: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate a full day of 1-minute bars (1440 bars).
        
        Args:
            start_date: Starting datetime (should be midnight)
            state_sequence: Optional list of (bar_index, state) tuples to control states
            auto_transition: If True, randomly transition between states based on session
            segment_length: If set, only allow state transitions at segment boundaries (e.g., 15 for 15-min segments)
            macro_regime: Optional day-level regime label (e.g., 'UP_DAY', 'DOWN_DAY', 'CHOP_DAY')
        
        Returns:
            DataFrame with OHLCV, tick columns, state labels, and optional macro_regime
        """
        bars = []
        current_state = MarketState.RANGING
        current_segment_id = 0
        
        # Update Liquidity Levels (assuming start of new day)
        # In a continuous simulation, this should be managed externally or carefully here.
        # For generate_day, we assume it's a new day, so update PDC if we have history?
        # Actually, the caller should set prev_day_close on the generator instance before calling generate_day.
        # But we should update it at the END of the day.

        
        # Build state map if provided
        state_map = {}
        if state_sequence:
            for bar_idx, state in state_sequence:
                state_map[bar_idx] = state
        
        for minute in range(1440):  # 24 hours * 60 minutes
            timestamp = start_date + timedelta(minutes=minute)
            
            # Update segment ID if using segments
            if segment_length:
                current_segment_id = minute // segment_length
            
            # Check for manual state transition
            if minute in state_map:
                current_state = state_map[minute]
            elif auto_transition:
                # Only transition at segment boundaries if segment_length is set
                can_transition = True
                if segment_length:
                    can_transition = (minute % segment_length == 0)
                
                if can_transition:
                    # Random state transition based on session
                    session = self.get_session(timestamp)
                    session_config = SESSION_CONFIGS[session]
                    
                    if self.rng.random() < session_config.state_transition_prob:
                        # Transition to a new state
                        current_state = self.rng.choice(list(MarketState))
            
            bar = self.generate_bar(timestamp, current_state)
            
            # Add segment ID if using segments
            if segment_length:
                bar['segment_id'] = current_segment_id
            
            bars.append(bar)
        
        df = pd.DataFrame(bars)
        
        # Add macro regime label if provided
        if macro_regime:
            df['macro_regime'] = macro_regime
        else:
            # Infer simple macro regime from net movement
            net_move = df['close'].iloc[-1] - df['open'].iloc[0]
            total_range = df['high'].max() - df['low'].min()
            
            if abs(net_move) > total_range * 0.3:
                df['macro_regime'] = 'UP_DAY' if net_move > 0 else 'DOWN_DAY'
            elif total_range < df['close'].iloc[0] * 0.01:  # Less than 1% range
                df['macro_regime'] = 'QUIET_DAY'
            else:
                df['macro_regime'] = 'CHOP_DAY'
        
        # Update Previous Day Close at end of day
        if not df.empty:
            self.prev_day_close = df['close'].iloc[-1]
            
        return df

```

---

## File: src/core/generator/states.py
```python
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

```

---

## File: src/core/generator/custom_states.py
```python
"""
Custom Market State Configurations

Pre-defined extreme and specialized market states for testing and simulation.
These complement the standard STATE_CONFIGS in engine.py
"""

from .engine import StateConfig


# Extreme volatility states
CUSTOM_STATES = {
    'mega_volatile': StateConfig(
        name="mega_volatile",
        avg_ticks_per_bar=15.0, # Reduced from 40.0
        ticks_per_bar_std=8.0,
        up_probability=0.5,
        trend_persistence=0.4,
        avg_tick_size=1.5,
        tick_size_std=1.0,
        max_tick_jump=6,
        volatility_multiplier=2.0,
        wick_probability=0.3,
        wick_extension_avg=3.0,
    ),
    
    'flash_crash': StateConfig(
        name="flash_crash",
        avg_ticks_per_bar=50.0,
        ticks_per_bar_std=15.0,
        up_probability=0.15,  # Strong downward bias
        trend_persistence=0.9,  # Very persistent down moves
        avg_tick_size=4.0,
        tick_size_std=2.5,
        max_tick_jump=20,
        volatility_multiplier=3.5,
        wick_probability=0.4,  # Some wicks but mostly directional
        wick_extension_avg=6.0,
    ),
    
    'melt_up': StateConfig(
        name="melt_up",
        avg_ticks_per_bar=35.0,
        ticks_per_bar_std=12.0,
        up_probability=0.85,  # Strong upward bias
        trend_persistence=0.85,  # Very persistent up moves
        avg_tick_size=2.5,
        tick_size_std=1.5,
        max_tick_jump=12,
        volatility_multiplier=2.5,
        wick_probability=0.35,
        wick_extension_avg=4.0,
    ),
    
    'whipsaw': StateConfig(
        name="whipsaw",
        avg_ticks_per_bar=12.0, # Reduced from 30.0
        ticks_per_bar_std=6.0,
        up_probability=0.5,
        trend_persistence=0.1,
        avg_tick_size=1.2,
        tick_size_std=0.8,
        max_tick_jump=5,
        volatility_multiplier=1.5,
        wick_probability=0.4,
        wick_extension_avg=2.5,
    ),
    
    'death_spiral': StateConfig(
        name="death_spiral",
        avg_ticks_per_bar=45.0,
        ticks_per_bar_std=18.0,
        up_probability=0.2,  # Strong down bias
        trend_persistence=0.75,  # Persistent but with some bounces
        avg_tick_size=3.5,
        tick_size_std=2.0,
        max_tick_jump=18,
        volatility_multiplier=3.2,
        wick_probability=0.5,
        wick_extension_avg=7.0,
    ),
    
    'moonshot': StateConfig(
        name="moonshot",
        avg_ticks_per_bar=38.0,
        ticks_per_bar_std=14.0,
        up_probability=0.8,  # Strong up bias
        trend_persistence=0.8,  # Very persistent
        avg_tick_size=3.0,
        tick_size_std=1.8,
        max_tick_jump=16,
        volatility_multiplier=2.8,
        wick_probability=0.4,
        wick_extension_avg=5.0,
    ),
    
    'slow_bleed': StateConfig(
        name="slow_bleed",
        avg_ticks_per_bar=8.0,
        ticks_per_bar_std=4.0,
        up_probability=0.35,  # Moderate down bias
        trend_persistence=0.7,  # Persistent grind
        avg_tick_size=1.0,
        tick_size_std=0.5,
        max_tick_jump=3,
        volatility_multiplier=0.8,
        wick_probability=0.25,
        wick_extension_avg=2.0,
    ),
    
    'slow_grind_up': StateConfig(
        name="slow_grind_up",
        avg_ticks_per_bar=8.0,
        ticks_per_bar_std=4.0,
        up_probability=0.65,  # Moderate up bias
        trend_persistence=0.7,  # Persistent grind
        avg_tick_size=1.0,
        tick_size_std=0.5,
        max_tick_jump=3,
        volatility_multiplier=0.8,
        wick_probability=0.25,
        wick_extension_avg=2.0,
    ),
    
    'opening_bell': StateConfig(
        name="opening_bell",
        avg_ticks_per_bar=14.0, # Reduced from 35.0
        ticks_per_bar_std=7.0,
        up_probability=0.5,
        trend_persistence=0.3,
        avg_tick_size=1.2,
        tick_size_std=0.8,
        max_tick_jump=5,
        volatility_multiplier=1.8,
        wick_probability=0.3,
        wick_extension_avg=3.0,
    ),
    
    'closing_squeeze': StateConfig(
        name="closing_squeeze",
        avg_ticks_per_bar=25.0,
        ticks_per_bar_std=12.0,
        up_probability=0.55,  # Slight up bias (short covering)
        trend_persistence=0.6,
        avg_tick_size=2.0,
        tick_size_std=1.5,
        max_tick_jump=10,
        volatility_multiplier=2.0,
        wick_probability=0.5,
        wick_extension_avg=4.0,
    ),
    
    'news_spike': StateConfig(
        name="news_spike",
        avg_ticks_per_bar=60.0,
        ticks_per_bar_std=25.0,
        up_probability=0.7,  # Usually news is positive initially
        trend_persistence=0.5,  # Mixed reactions
        avg_tick_size=4.0,
        tick_size_std=3.0,
        max_tick_jump=25,
        volatility_multiplier=4.0,
        wick_probability=0.7,
        wick_extension_avg=8.0,
    ),
    
    'dead_zone': StateConfig(
        name="dead_zone",
        avg_ticks_per_bar=2.0,
        ticks_per_bar_std=1.0,
        up_probability=0.5,
        trend_persistence=0.5,
        avg_tick_size=1.0,
        tick_size_std=0.2,
        max_tick_jump=1,
        volatility_multiplier=0.2,
        wick_probability=0.1,
        wick_extension_avg=1.0,
    ),
}


# Categorized for easy access
EXTREME_VOLATILITY = ['mega_volatile', 'flash_crash', 'melt_up', 'whipsaw', 'news_spike']
DIRECTIONAL = ['flash_crash', 'melt_up', 'death_spiral', 'moonshot', 'slow_bleed', 'slow_grind_up']
SESSION_SPECIFIC = ['opening_bell', 'closing_squeeze']
LOW_VOLATILITY = ['dead_zone', 'slow_bleed', 'slow_grind_up']


def get_custom_state(name: str) -> StateConfig:
    """
    Get a custom state configuration by name.
    
    Args:
        name: Name of the custom state
    
    Returns:
        StateConfig instance
    
    Raises:
        KeyError: If state name not found
    """
    if name not in CUSTOM_STATES:
        available = ', '.join(CUSTOM_STATES.keys())
        raise KeyError(f"Custom state '{name}' not found. Available: {available}")
    
    return CUSTOM_STATES[name]


def list_custom_states() -> dict:
    """
    Get a summary of all custom states.
    
    Returns:
        Dictionary with state names and descriptions
    """
    summaries = {}
    
    for name, config in CUSTOM_STATES.items():
        summaries[name] = {
            'avg_ticks_per_bar': config.avg_ticks_per_bar,
            'up_probability': config.up_probability,
            'trend_persistence': config.trend_persistence,
            'volatility_multiplier': config.volatility_multiplier,
            'max_tick_jump': config.max_tick_jump,
        }
    
    return summaries


def print_custom_states():
    """Print a formatted table of all custom states"""
    print("\n" + "=" * 80)
    print("CUSTOM MARKET STATES")
    print("=" * 80)
    print(f"\n{'State':<20} {'AvgTicks':>8} {'UpProb':>7} {'Persist':>7} {'VolMult':>7} {'MaxJump':>8}")
    print("-" * 80)
    
    for name, config in CUSTOM_STATES.items():
        print(f"{name:<20} {config.avg_ticks_per_bar:>8.1f} {config.up_probability:>7.2f} "
              f"{config.trend_persistence:>7.2f} {config.volatility_multiplier:>7.1f} "
              f"{config.max_tick_jump:>8}")
    
    print("\n" + "=" * 80)
    print("\nCategories:")
    print(f"  Extreme Volatility: {', '.join(EXTREME_VOLATILITY)}")
    print(f"  Directional: {', '.join(DIRECTIONAL)}")
    print(f"  Session Specific: {', '.join(SESSION_SPECIFIC)}")
    print(f"  Low Volatility: {', '.join(LOW_VOLATILITY)}")
    print("=" * 80)


if __name__ == "__main__":
    print_custom_states()

```

---

## File: src/core/generator/utils.py
```python
"""
Utilities for analyzing synthetic price data

Helper functions to sanity-check generator output and compute statistics.
"""

import pandas as pd
from typing import Dict, Any


def summarize_day(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a day of generated data.
    
    Args:
        df: DataFrame from PriceGenerator.generate_day()
    
    Returns:
        Dictionary with overall stats, per-state stats, and per-session stats
    """
    summary = {}
    
    # Overall day stats
    summary['overall'] = {
        'num_bars': len(df),
        'start_price': float(df['open'].iloc[0]),
        'end_price': float(df['close'].iloc[-1]),
        'net_move': float(df['close'].iloc[-1] - df['open'].iloc[0]),
        'net_move_ticks': int(df['close_ticks'].iloc[-1] - df['open_ticks'].iloc[0]),
        'high': float(df['high'].max()),
        'low': float(df['low'].min()),
        'total_range': float(df['high'].max() - df['low'].min()),
        'total_range_ticks': int((df['high'].max() - df['low'].min()) / 0.25),
        'avg_volume': float(df['volume'].mean()),
        'total_volume': int(df['volume'].sum()),
        'max_range_ticks': int(df['range_ticks'].max()),
        'avg_range_ticks': float(df['range_ticks'].mean()),
        'avg_body_ticks': float(df['body_ticks'].mean()),
        'avg_delta_ticks': float(df['delta_ticks'].mean()),
        'std_delta_ticks': float(df['delta_ticks'].std()),
    }
    
    # Macro regime if present
    if 'macro_regime' in df.columns:
        summary['overall']['macro_regime'] = df['macro_regime'].iloc[0]
    
    # Per-state statistics
    if 'state' in df.columns:
        summary['by_state'] = {}
        for state in df['state'].unique():
            state_df = df[df['state'] == state]
            summary['by_state'][state] = {
                'count': len(state_df),
                'pct_of_day': float(len(state_df) / len(df) * 100),
                'avg_delta_ticks': float(state_df['delta_ticks'].mean()),
                'std_delta_ticks': float(state_df['delta_ticks'].std()),
                'avg_range_ticks': float(state_df['range_ticks'].mean()),
                'avg_body_ticks': float(state_df['body_ticks'].mean()),
                'avg_volume': float(state_df['volume'].mean()),
                'net_move_ticks': int(state_df['delta_ticks'].sum()),
                'up_bars': int((state_df['delta_ticks'] > 0).sum()),
                'down_bars': int((state_df['delta_ticks'] < 0).sum()),
                'flat_bars': int((state_df['delta_ticks'] == 0).sum()),
            }
    
    # Per-session statistics
    if 'session' in df.columns:
        summary['by_session'] = {}
        for session in df['session'].unique():
            session_df = df[df['session'] == session]
            summary['by_session'][session] = {
                'count': len(session_df),
                'pct_of_day': float(len(session_df) / len(df) * 100),
                'avg_delta_ticks': float(session_df['delta_ticks'].mean()),
                'avg_range_ticks': float(session_df['range_ticks'].mean()),
                'avg_volume': float(session_df['volume'].mean()),
                'net_move_ticks': int(session_df['delta_ticks'].sum()),
            }
    
    # Per-segment statistics if segments exist
    if 'segment_id' in df.columns:
        summary['by_segment'] = {}
        for seg_id in df['segment_id'].unique():
            seg_df = df[df['segment_id'] == seg_id]
            summary['by_segment'][int(seg_id)] = {
                'count': len(seg_df),
                'state': seg_df['state'].iloc[0] if len(seg_df) > 0 else None,
                'net_move_ticks': int(seg_df['delta_ticks'].sum()),
                'range_ticks': int(seg_df['range_ticks'].sum()),
                'avg_volume': float(seg_df['volume'].mean()),
            }
    
    return summary


def print_summary(summary: Dict[str, Any], verbose: bool = True) -> None:
    """
    Pretty-print a summary dictionary.
    
    Args:
        summary: Output from summarize_day()
        verbose: If True, print detailed per-state and per-session stats
    """
    print("\n" + "=" * 60)
    print("DAY SUMMARY")
    print("=" * 60)
    
    # Overall stats
    overall = summary['overall']
    print(f"\nOverall:")
    print(f"  Bars: {overall['num_bars']}")
    print(f"  Price: {overall['start_price']:.2f} → {overall['end_price']:.2f}")
    print(f"  Net Move: {overall['net_move']:.2f} ({overall['net_move_ticks']:+d} ticks)")
    print(f"  Range: {overall['low']:.2f} - {overall['high']:.2f} ({overall['total_range_ticks']} ticks)")
    print(f"  Avg Bar Range: {overall['avg_range_ticks']:.1f} ticks")
    print(f"  Avg Bar Body: {overall['avg_body_ticks']:.1f} ticks")
    print(f"  Avg Delta: {overall['avg_delta_ticks']:.2f} ± {overall['std_delta_ticks']:.2f} ticks")
    print(f"  Total Volume: {overall['total_volume']:,}")
    
    if 'macro_regime' in overall:
        print(f"  Macro Regime: {overall['macro_regime']}")
    
    if verbose and 'by_state' in summary:
        print(f"\nBy State:")
        print(f"  {'State':<15} {'Count':>6} {'%':>6} {'AvgΔ':>8} {'AvgRng':>8} {'NetΔ':>8} {'Up/Dn':>10}")
        print("  " + "-" * 70)
        for state, stats in summary['by_state'].items():
            print(f"  {state:<15} {stats['count']:>6} {stats['pct_of_day']:>5.1f}% "
                  f"{stats['avg_delta_ticks']:>7.2f} {stats['avg_range_ticks']:>7.1f} "
                  f"{stats['net_move_ticks']:>7d} "
                  f"{stats['up_bars']:>4}/{stats['down_bars']:<4}")
    
    if verbose and 'by_session' in summary:
        print(f"\nBy Session:")
        print(f"  {'Session':<15} {'Count':>6} {'%':>6} {'AvgΔ':>8} {'AvgRng':>8} {'NetΔ':>8}")
        print("  " + "-" * 60)
        for session, stats in summary['by_session'].items():
            print(f"  {session:<15} {stats['count']:>6} {stats['pct_of_day']:>5.1f}% "
                  f"{stats['avg_delta_ticks']:>7.2f} {stats['avg_range_ticks']:>7.1f} "
                  f"{stats['net_move_ticks']:>7d}")
    
    print("=" * 60)


def compare_states(df: pd.DataFrame, states_to_compare: list = None) -> pd.DataFrame:
    """
    Create a comparison table of different states.
    
    Args:
        df: DataFrame from generator
        states_to_compare: List of states to compare, or None for all
    
    Returns:
        DataFrame with comparison metrics
    """
    if states_to_compare is None:
        states_to_compare = df['state'].unique()
    
    comparison = []
    
    for state in states_to_compare:
        state_df = df[df['state'] == state]
        if len(state_df) == 0:
            continue
        
        comparison.append({
            'state': state,
            'count': len(state_df),
            'avg_delta_ticks': state_df['delta_ticks'].mean(),
            'std_delta_ticks': state_df['delta_ticks'].std(),
            'avg_range_ticks': state_df['range_ticks'].mean(),
            'avg_body_ticks': state_df['body_ticks'].mean(),
            'avg_upper_wick': state_df['upper_wick_ticks'].mean(),
            'avg_lower_wick': state_df['lower_wick_ticks'].mean(),
            'avg_volume': state_df['volume'].mean(),
            'up_pct': (state_df['delta_ticks'] > 0).sum() / len(state_df) * 100,
        })
    
    return pd.DataFrame(comparison)

```

---

