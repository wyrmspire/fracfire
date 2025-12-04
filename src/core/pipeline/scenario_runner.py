"""
Scenario Runner Module

Orchestrates the generation/loading of data, application of indicators,
and execution of setup detection for a given scenario specification.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.core.generator.engine import PriceGenerator, PhysicsConfig, MarketState, StateConfig
from src.core.detector.indicators import IndicatorConfig, add_1m_indicators, add_5m_indicators
from src.core.detector.library import (
    ORBConfig, LevelScalpConfig, EMA200ContinuationConfig,
    BreakoutConfig, ReversalConfig, OpeningPushConfig, MOCConfig,
    run_orb_family, run_ema200_continuation_family,
    run_breakout_family, run_reversal_family,
    run_opening_push_family, run_moc_family,
    SetupOutcome
)

@dataclass
class SetupConfig:
    """Container for all setup family configurations."""
    active_families: List[str] = field(default_factory=lambda: ["orb"])
    orb: ORBConfig = field(default_factory=ORBConfig)
    level_scalp: LevelScalpConfig = field(default_factory=LevelScalpConfig)
    ema200: EMA200ContinuationConfig = field(default_factory=EMA200ContinuationConfig)
    breakout: BreakoutConfig = field(default_factory=BreakoutConfig)
    reversal: ReversalConfig = field(default_factory=ReversalConfig)
    opening_push: OpeningPushConfig = field(default_factory=OpeningPushConfig)
    moc: MOCConfig = field(default_factory=MOCConfig)

@dataclass
class ScenarioSpec:
    """Specification for a market scenario."""
    source: Literal["synthetic", "real"] = "synthetic"

    # Synthetic-only options
    day_archetype: Optional[str] = None        # e.g. "trend_day_up", "range_day"
    macro_regime: Optional[str] = None         # map to generator macro regime
    state_sequence: Optional[List[Tuple[int, str]]] = None
    # e.g. [(0, "RANGING"), (60, "BREAKOUT"), (120, "RALLY")]
    
    # Force a single state for the entire day (overrides state_sequence)
    force_state: Optional[str] = None

    # Common options
    session: str = "RTH"
    start_time: Optional[datetime] = None
    duration_minutes: int = 390
    physics_overrides: Dict[str, Any] = field(default_factory=dict)

    indicator_cfg: IndicatorConfig = field(default_factory=IndicatorConfig)
    setup_cfg: SetupConfig = field(default_factory=SetupConfig)

@dataclass
class ScenarioResult:
    """Results of a scenario run."""
    df_1m: pd.DataFrame
    df_5m: pd.DataFrame
    outcomes: List[SetupOutcome]
    metadata: Dict[str, Any] = field(default_factory=dict)

def run_scenario(spec: ScenarioSpec, seed: int = 42) -> ScenarioResult:
    """
    Execute a scenario based on the specification.
    """
    
    # 1. Generate or Load Data
    if spec.source == "synthetic":
        # Apply physics overrides
        physics = PhysicsConfig(**spec.physics_overrides)
        
        # Initialize Generator
        start_date = spec.start_time or datetime(2024, 1, 1, 9, 30)
        day_start = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        generator = PriceGenerator(seed=seed, physics_config=physics)
        
        # Handle state sequence
        seq = None
        if spec.force_state:
            # If forced state, set sequence to start with that state
            try:
                state_enum = MarketState(spec.force_state.lower())
                seq = [(0, state_enum)]
            except ValueError:
                print(f"Warning: Unknown forced state {spec.force_state}, using default.")
        elif spec.state_sequence:
            seq = []
            for min_idx, state_name in spec.state_sequence:
                try:
                    state_enum = MarketState(state_name.lower())
                    seq.append((min_idx, state_enum))
                except ValueError:
                    print(f"Warning: Unknown state {state_name}, skipping.")
        
        # Generate Data
        # If force_state is set, we might want to disable auto_transition in generate_day
        # But generate_day doesn't expose auto_transition directly in arguments, 
        # it infers from state_sequence. If state_sequence is provided, it uses it.
        # If state_sequence covers the whole day (starts at 0), it should stick to it 
        # unless generate_day has logic to override.
        # Looking at engine.py, if state_sequence is provided, it uses it.
        
        df_1m = generator.generate_day(
            start_date=day_start,
            state_sequence=seq,
            macro_regime=spec.macro_regime
        )
        
        # Slice to requested duration/session
        if spec.start_time:
            end_time = spec.start_time + timedelta(minutes=spec.duration_minutes)
            df_1m = df_1m[(df_1m['time'] >= spec.start_time) & (df_1m['time'] < end_time)].copy()
        
        df_1m.set_index('time', inplace=True)
        
    else:
        raise NotImplementedError("Real data loading not yet implemented")

    if df_1m.empty:
        return ScenarioResult(pd.DataFrame(), pd.DataFrame(), [], {"error": "No data generated"})

    # 2. Resample to 5m
    df_5m = df_1m.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # 3. Add Indicators
    df_1m = add_1m_indicators(df_1m, spec.indicator_cfg)
    df_5m = add_5m_indicators(df_5m, spec.indicator_cfg)

    # 4. Run Setups
    outcomes: List[SetupOutcome] = []
    
    # Helper to run family if active
    def run_family(name, runner, cfg_obj):
        if name in spec.setup_cfg.active_families:
            kwargs = cfg_obj.__dict__
            return runner(df_5m, **kwargs)
        return []

    outcomes.extend(run_family("orb", run_orb_family, spec.setup_cfg.orb))
    outcomes.extend(run_family("ema200", run_ema200_continuation_family, spec.setup_cfg.ema200))
    outcomes.extend(run_family("breakout", run_breakout_family, spec.setup_cfg.breakout))
    outcomes.extend(run_family("reversal", run_reversal_family, spec.setup_cfg.reversal))
    outcomes.extend(run_family("opening_push", run_opening_push_family, spec.setup_cfg.opening_push))
    outcomes.extend(run_family("moc", run_moc_family, spec.setup_cfg.moc))

    # 5. Return Result
    return ScenarioResult(
        df_1m=df_1m,
        df_5m=df_5m,
        outcomes=outcomes,
        metadata={
            "seed": seed,
            "source": spec.source,
            "setup_count": len(outcomes),
            "forced_state": spec.force_state
        }
    )
