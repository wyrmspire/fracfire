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
from .sampler import PhysicsSampler, PhysicsProfile
from .fractal_planner import FractalPlanner, TrajectoryPlan


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
    
    # 6. Advanced Distribution Options (OPTIONAL - defaults to Gaussian for backward compatibility)
    use_fat_tails: bool = False             # Enable Student-t distribution for fat tails (realistic spikes)
    fat_tail_df: float = 3.0                # Degrees of freedom for Student-t (lower = fatter tails)
    
    # 7. Volatility Clustering Options (OPTIONAL - defaults to no clustering)
    use_volatility_clustering: bool = False # Enable GARCH-like volatility clustering
    volatility_persistence: float = 0.3     # How much recent volatility affects current (0-1)
    volatility_smoothing: float = 0.3       # EMA smoothing factor for volatility tracking (0-1)


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
# Base state configuration (Neutral / "Good Physics")
BASE_STATE = StateConfig(
    name="base",
    avg_ticks_per_bar=8.0,
    ticks_per_bar_std=4.0,
    up_probability=0.5,
    trend_persistence=0.4, # Reduced from 0.5 to reduce clumping
    avg_tick_size=1.0,
    tick_size_std=0.5,
    max_tick_jump=5,
    volatility_multiplier=1.0,
    wick_probability=0.2,
    wick_extension_avg=1.5,
    mean_reversion_strength=0.0,
)

# Predefined state configurations (Relative offsets/biases from BASE)
STATE_CONFIGS = {
    MarketState.RANGING: StateConfig(
        name="ranging",
        avg_ticks_per_bar=6.0,          # Slightly lower activity
        ticks_per_bar_std=3.0,
        up_probability=0.5,             # Neutral
        trend_persistence=0.4,          # Lower persistence (mean reverting)
        avg_tick_size=1.0,
        tick_size_std=0.5,
        max_tick_jump=4,
        volatility_multiplier=0.9,      # Slightly lower vol
        wick_probability=0.2,
        wick_extension_avg=1.5,
    ),
    MarketState.ZOMBIE: StateConfig(
        name="zombie",
        avg_ticks_per_bar=4.0,          # Low activity
        ticks_per_bar_std=2.0,
        up_probability=0.52,            # Slight drift
        trend_persistence=0.6,          # Sticky
        avg_tick_size=1.0,
        tick_size_std=0.2,
        max_tick_jump=2,
        volatility_multiplier=0.7,      # Low vol
        wick_probability=0.1,
        wick_extension_avg=1.0,
    ),
    MarketState.RALLY: StateConfig(
        name="rally",
        avg_ticks_per_bar=10.0,         # Higher activity
        ticks_per_bar_std=5.0,
        up_probability=0.60,            # Bias up (was 0.75+)
        trend_persistence=0.65,         # Trending (was 0.85)
        avg_tick_size=1.1,
        tick_size_std=0.5,
        max_tick_jump=5,
        volatility_multiplier=1.3,      # Higher vol (was 1.8)
        wick_probability=0.15,
        wick_extension_avg=1.5,
    ),
    MarketState.IMPULSIVE: StateConfig(
        name="impulsive",
        avg_ticks_per_bar=12.0,
        ticks_per_bar_std=6.0,
        up_probability=0.5,             # Neutral direction, high variance
        trend_persistence=0.55,
        avg_tick_size=1.3,
        tick_size_std=0.6,
        max_tick_jump=6,
        volatility_multiplier=1.5,      # High vol
        wick_probability=0.25,
        wick_extension_avg=2.0,
    ),
    MarketState.BREAKDOWN: StateConfig(
        name="breakdown",
        avg_ticks_per_bar=12.0,
        ticks_per_bar_std=6.0,
        up_probability=0.40,            # Bias down (was 0.25)
        trend_persistence=0.70,         # Strong trend (was 0.85)
        avg_tick_size=1.2,
        tick_size_std=0.6,
        max_tick_jump=6,
        volatility_multiplier=1.6,      # High vol (was 1.8)
        wick_probability=0.2,
        wick_extension_avg=2.0,
    ),
    MarketState.BREAKOUT: StateConfig(
        name="breakout",
        avg_ticks_per_bar=12.0,
        ticks_per_bar_std=6.0,
        up_probability=0.60,            # Bias up (was 0.75)
        trend_persistence=0.70,         # Strong trend (was 0.85)
        avg_tick_size=1.2,
        tick_size_std=0.6,
        max_tick_jump=6,
        volatility_multiplier=1.6,      # High vol (was 1.8)
        wick_probability=0.2,
        wick_extension_avg=2.0,
    ),
    MarketState.FLAT: StateConfig(
        name="flat",
        avg_ticks_per_bar=3.0,
        ticks_per_bar_std=1.5,
        up_probability=0.5,
        trend_persistence=0.3,
        avg_tick_size=0.8,
        tick_size_std=0.3,
        max_tick_jump=2,
        volatility_multiplier=0.5,
        wick_probability=0.1,
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
        physics_config: Optional[PhysicsConfig] = None,
        seed: Optional[int] = None,
        warmup_data: Optional[pd.DataFrame] = None
    ):
        """
        Initialize the price generator.
        
        Args:
            initial_price: Starting price level
            physics_config: Physics configuration (knobs)
            seed: Random seed for reproducibility
            warmup_data: Optional DataFrame with historical data to warm up the planner
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
        
        # Physics Sampler (Generative Model)
        self.sampler = PhysicsSampler()
        self.current_physics_profile: Optional[PhysicsProfile] = None
        self.last_sampler_update = None
        
        # Fractal Planner (Mission Control)
        self.planner = FractalPlanner()
        self.current_plan: Optional[TrajectoryPlan] = None
        self.last_plan_update = None
        
        if warmup_data is not None:
            self.planner.warmup_history(warmup_data)
            # Also set initial price to last close of warmup
            if not warmup_data.empty:
                self.current_price = warmup_data['close'].iloc[-1]
                self.prev_day_close = self.current_price # Approximation
        
        if seed is not None:
            np.random.seed(seed)
        
        self.rng = np.random.default_rng(seed)
        
        # Volatility clustering tracking
        self.recent_volatility = 1.0  # Normalized volatility level (1.0 = baseline)
    
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
    
    def _sample_distribution(self, mean: float, std: float) -> float:
        """
        Sample from either Gaussian (default) or Student-t (optional) distribution.
        
        This is configurable via PhysicsConfig.use_fat_tails flag.
        - If False (default): Uses normal distribution (smooth, no extreme moves)
        - If True: Uses Student-t distribution (fat tails, occasional spikes)
        
        Uses NumPy's rng.standard_t() and rng.normal() implementations.
        
        Args:
            mean: Mean/center of distribution
            std: Standard deviation/scale
            
        Returns:
            Sampled value
        """
        if self.physics.use_fat_tails:
            # Student-t distribution with fat tails (NumPy implementation)
            # Lower df = fatter tails = more extreme moves
            t_sample = self.rng.standard_t(df=self.physics.fat_tail_df)
            return mean + t_sample * std
        else:
            # Standard Gaussian (backward compatible default)
            return self.rng.normal(mean, std)
    
    def _apply_volatility_clustering(self, base_volatility: float) -> float:
        """
        Apply optional GARCH-like volatility clustering.
        
        If enabled, recent high volatility increases current volatility.
        This creates realistic "clumping" of volatile periods.
        
        Args:
            base_volatility: Base volatility multiplier
            
        Returns:
            Adjusted volatility with clustering effect applied
        """
        if not self.physics.use_volatility_clustering:
            return base_volatility
        
        # Apply clustering: current vol influenced by recent vol
        clustered_vol = base_volatility * (
            1.0 + self.physics.volatility_persistence * (self.recent_volatility - 1.0)
        )
        return max(0.1, clustered_vol)  # Floor to prevent collapse
    
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
        
        # Apply optional volatility clustering
        volatility = self._apply_volatility_clustering(volatility)
        
        tick_size = max(
            1,
            int(self._sample_distribution(
                state_config.avg_tick_size * volatility,
                state_config.tick_size_std * volatility
            ))
        )
        tick_size = min(tick_size, state_config.max_tick_jump)
        
        self.last_direction = direction
        return direction, tick_size
    
    def blend_state_configs(self, base: StateConfig, target: StateConfig, alpha: float) -> StateConfig:
        """Blend two state configs with a mixing factor alpha (0.0 to 1.0)."""
        def lerp(a, b):
            return a + alpha * (b - a)
            
        return StateConfig(
            name=target.name,
            avg_ticks_per_bar=lerp(base.avg_ticks_per_bar, target.avg_ticks_per_bar),
            ticks_per_bar_std=lerp(base.ticks_per_bar_std, target.ticks_per_bar_std),
            up_probability=lerp(base.up_probability, target.up_probability),
            trend_persistence=lerp(base.trend_persistence, target.trend_persistence),
            avg_tick_size=lerp(base.avg_tick_size, target.avg_tick_size),
            tick_size_std=lerp(base.tick_size_std, target.tick_size_std),
            max_tick_jump=int(lerp(base.max_tick_jump, target.max_tick_jump)),
            volatility_multiplier=lerp(base.volatility_multiplier, target.volatility_multiplier),
            wick_probability=lerp(base.wick_probability, target.wick_probability),
            wick_extension_avg=lerp(base.wick_extension_avg, target.wick_extension_avg),
            mean_reversion_strength=lerp(base.mean_reversion_strength, target.mean_reversion_strength),
        )

    def _update_physics_from_sampler(self, timestamp: datetime):
        """
        Query the PhysicsSampler and update the current physics profile.
        Run this every 30 minutes.
        """
        if self.last_sampler_update is None or (timestamp - self.last_sampler_update).total_seconds() >= 1800:
            # We need recent metrics to query the sampler.
            # For now, let's use placeholders or track them in the generator.
            # Ideally, we'd track rolling volatility and trend.
            
            # Placeholder: Random walk context if we don't have history
            recent_vol = 1.0
            recent_trend = 0.0
            recent_dir = 0.0
            
            profile = self.sampler.sample_physics(
                current_time=timestamp,
                recent_volatility=recent_vol,
                recent_trend=recent_trend,
                recent_direction=recent_dir
            )
            
            if profile:
                self.current_physics_profile = profile
                self.last_sampler_update = timestamp
                # print(f"[{timestamp}] Updated Physics: Vol={profile.volatility:.2f}, Trend={profile.trend_efficiency:.2f}")

    def _update_plan(self, timestamp: datetime):
        """
        Query the FractalPlanner for a new mission.
        Run this every 60 minutes.
        """
        if self.last_plan_update is None or (timestamp - self.last_plan_update).total_seconds() >= 3600:
            plan = self.planner.get_plan()
            if plan:
                self.current_plan = plan
                self.last_plan_update = timestamp
                # print(f"[{timestamp}] New Plan: Move={plan.displacement:.2f}, Dist={plan.total_distance:.2f}")

    def generate_bar(
        self, 
        timestamp: datetime, 
        state: MarketState = MarketState.RANGING,
        custom_state_config: Optional[StateConfig] = None
    ) -> dict:
        """
        Generate a single 1-minute OHLCV bar.
        
        Args:
            timestamp: Bar timestamp
            state: Market state (affects dynamics)
            custom_state_config: Override for state config
            
        Returns:
            Dict with open, high, low, close, volume
        """
        # 0. Update Drivers
        self._update_physics_from_sampler(timestamp)
        self._update_plan(timestamp)
        
        # 1. Determine Configs
        # Blend BASE_STATE with the target state (alpha=0.3 for stronger bias)
        target_cfg = custom_state_config or STATE_CONFIGS[state]
        
        # Apply Planner Overrides (Mission Control)
        # If we have a plan, it takes precedence over the Sampler and State
        if self.current_plan:
            # Map Plan to Physics
            # Displacement -> Direction Bias
            # Total Distance -> Volatility
            
            # Volatility
            # Base distance for 1h is ~100-200 points? 
            # Let's normalize. 
            # If total_distance is high, high vol.
            vol_mult = max(0.5, self.current_plan.total_distance / 100.0) # Scaling factor
            
            # Direction
            # Displacement is net move.
            # If displacement is +50, we want up_prob > 0.5
            # Scale: 50 points = strong trend
            disp_norm = self.current_plan.displacement / 50.0
            up_prob = 0.5 + (np.tanh(disp_norm) * 0.2) # +/- 0.2 bias
            
            # Persistence
            # If High/Low excursion is large but displacement is small -> High Vol, Low Trend (Chop)
            # If Displacement ~= Total Distance -> High Trend
            efficiency = abs(self.current_plan.displacement) / (self.current_plan.total_distance + 1e-9)
            persistence = 0.3 + (efficiency * 0.5) # 0.3 to 0.8
            
            planned_cfg = StateConfig(
                name="planned",
                volatility_multiplier=vol_mult,
                up_probability=up_prob,
                trend_persistence=persistence,
                wick_probability=0.2, # Default
                avg_ticks_per_bar=8.0,
                ticks_per_bar_std=4.0,
                avg_tick_size=1.0,
                tick_size_std=0.5,
                max_tick_jump=5,
                wick_extension_avg=1.5,
                mean_reversion_strength=0.0
            )
            
            # Strong override
            state_config = self.blend_state_configs(BASE_STATE, planned_cfg, alpha=0.8)
            
        # Apply Sampler Overrides if active (and no plan yet)
        elif self.current_physics_profile:
            # ... (Existing Sampler Logic) ...
            # Create a dynamic state config based on the sampled profile
            # We map the profile metrics to StateConfig parameters
            
            # Volatility -> Vol Multiplier
            # Base vol in library is ~1.0-2.0.
            vol_mult = self.current_physics_profile.volatility / 1.5 
            
            # Trend -> Trend Persistence & Up Prob
            # Direction is -1 to 1.
            direction = self.current_physics_profile.direction
            trend_strength = self.current_physics_profile.trend_efficiency
            
            # Bias up_prob based on direction
            # 0.5 +/- 0.2
            up_prob = 0.5 + (direction * 20.0) # Scale direction (small number) to prob
            up_prob = max(0.4, min(0.6, up_prob)) # Clamp
            
            # Persistence based on efficiency
            persistence = 0.4 + (trend_strength * 2.0)
            persistence = min(0.8, persistence)
            
            # Wick ratio
            wick_prob = 0.1 + (self.current_physics_profile.wick_ratio * 0.5)
            
            sampled_cfg = StateConfig(
                name="sampled",
                volatility_multiplier=vol_mult,
                up_probability=up_prob,
                trend_persistence=persistence,
                wick_probability=wick_prob,
                # Keep others default
                avg_ticks_per_bar=8.0,
                ticks_per_bar_std=4.0,
                avg_tick_size=1.0,
                tick_size_std=0.5,
                max_tick_jump=5,
                wick_extension_avg=1.5,
                mean_reversion_strength=0.0
            )
            
            # Blend the Sampled Config with the Base State
            # We give the Sampled Config high weight (e.g. 0.7) because it represents "Reality"
            state_config = self.blend_state_configs(BASE_STATE, sampled_cfg, alpha=0.7)
            
        else:
            # Fallback to standard state logic
            state_config = self.blend_state_configs(BASE_STATE, target_cfg, alpha=0.3)
        
        session = self.get_session(timestamp)
        session_config = SESSION_CONFIGS[session]
        dow_config = DOW_CONFIGS[timestamp.weekday()]
        
        # Intra-day Volume Profile (RTH only)
        time_mult = 1.0
        if session == Session.RTH:
            hour = timestamp.hour
            minute = timestamp.minute
            
            # Open (9:30 - 10:30): High volume
            if hour == 9 or (hour == 10 and minute < 30):
                time_mult = 1.5
            # Lunch (11:30 - 13:00): Low volume
            elif (hour == 11 and minute >= 30) or hour == 12:
                time_mult = 0.6
            # Close (15:00 - 16:00): High volume
            elif hour == 15:
                time_mult = 1.4
        
        # 2. Determine Activity Level (Tick Count)
        # Base ticks * Session Mult * DOW Mult * Time Mult * Random Noise
        avg_ticks = state_config.avg_ticks_per_bar * self.physics.avg_ticks_per_bar / 8.0
        avg_ticks *= session_config.volume_multiplier * dow_config.volume_multiplier * time_mult
        
        num_ticks = int(self._sample_distribution(avg_ticks, state_config.ticks_per_bar_std))
        num_ticks = max(1, num_ticks) # At least 1 tick
        
        # 3. Generate Ticks
        open_price = self.current_price
        high_price = open_price
        low_price = open_price
        
        # Reset direction for new bar? Or keep momentum?
        # Let's keep momentum but decay it slightly
        
        for _ in range(num_ticks):
            direction, tick_size = self.generate_tick_movement(state_config, session_config, dow_config)
            
            # Apply movement
            move = direction * tick_size * self.TICK_SIZE
            self.current_price += move
            
            # Update High/Low
            if self.current_price > high_price:
                high_price = self.current_price
            if self.current_price < low_price:
                low_price = self.current_price
                
            # Update last direction
            if direction != 0:
                self.last_direction = direction
                
        # 4. Apply Wicks (Post-processing physics)
        # Sometimes price extends further than the random walk suggests (stop runs)
        if self.rng.random() < state_config.wick_probability * self.physics.wick_probability / 0.2:
            extension = self.rng.exponential(state_config.wick_extension_avg) * self.TICK_SIZE
            if self.rng.random() < 0.5:
                high_price += extension
                # If we extended high, maybe close pulls back?
                # For now, just extend the range
            else:
                low_price -= extension
        
        close_price = self.current_price
        
        # Generate volume (scaled by session and day)
        base_volume = max(
            10,
            int(self._sample_distribution(100, 50))
        )
        volume = int(
            base_volume *
            session_config.volume_multiplier *
            dow_config.volume_multiplier *
            state_config.volatility_multiplier *
            time_mult
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
        
        bar = {
            'time': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'open_ticks': open_ticks,
            'high_ticks': high_ticks,
            'low_ticks': low_ticks,
            'close_ticks': close_ticks,
            'delta_ticks': delta_ticks,
            'range_ticks': range_ticks,
            'body_ticks': body_ticks,
            'upper_wick_ticks': upper_wick_ticks,
            'lower_wick_ticks': lower_wick_ticks,
            'state': state.value,
            'session': session.value,
        }
        
        # Update Planner History
        self.planner.update_history(bar)
        
        # Update volatility tracking for clustering (if enabled)
        if self.physics.use_volatility_clustering:
            # Measure this bar's volatility relative to baseline
            bar_volatility = range_ticks / max(1, state_config.avg_ticks_per_bar)
            # Exponential moving average of recent volatility
            alpha = self.physics.volatility_smoothing
            self.recent_volatility = (1 - alpha) * self.recent_volatility + alpha * bar_volatility
        
        return bar
    
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
