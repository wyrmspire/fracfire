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
