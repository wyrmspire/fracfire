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
