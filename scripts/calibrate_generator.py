"""
Calibrate Generator

Runs the generator for single days with different configurations to measure
daily range and drift. Used to tune parameters to realistic levels.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.core.generator import PriceGenerator, StateConfig
from src.core.generator.states import FractalStateManager, DayState, DAY_STATE_PARAMS

def test_day_config(day_state: DayState, direction: int = 1, n_days: int = 5):
    print(f"\nTesting {day_state.name} (Direction: {direction})")
    
    gen = PriceGenerator(initial_price=5000.0)
    fractal = FractalStateManager()
    
    # Force state
    fractal.day_state = day_state
    fractal.day_direction = direction
    fractal.day_params = DAY_STATE_PARAMS[day_state].copy()
    
    results = []
    
    for i in range(n_days):
        current_price = 5000.0
        gen.current_price = current_price
        start_time = datetime(2025, 1, 1)
        
        # Run 1440 minutes
        for _ in range(1440):
            # Update fractal (but keep day state fixed)
            fractal.update(start_time, force_hour_transition=False)
            # Reset day state if update changed it (hacky but needed for isolation)
            fractal.day_state = day_state
            fractal.day_direction = direction
            
            params = fractal.get_combined_parameters()
            
            config = StateConfig(
                name="dynamic",
                avg_ticks_per_bar=12.0 * params['volatility_mult'],
                ticks_per_bar_std=6.0 * params['volatility_mult'],
                up_probability=params['directional_bias'],
                trend_persistence=params['trend_strength'],
                volatility_multiplier=params['volatility_mult'],
            )
            
            bar = gen.generate_bar(start_time, custom_state_config=config)
            start_time += timedelta(minutes=1)
            
        net_move = gen.current_price - 5000.0
        results.append(net_move)
        print(f"  Day {i+1}: {net_move:.2f}")
        
    avg_move = np.mean(results)
    print(f"  Average Move: {avg_move:.2f}")
    return avg_move

def test_gap_fill():
    print("\nTesting Gap Fill (Range Day)")
    gen = PriceGenerator(initial_price=5050.0) # Gap up 50 points
    gen.prev_day_close = 5000.0
    
    fractal = FractalStateManager()
    fractal.day_state = DayState.RANGE_DAY
    fractal.day_params = DAY_STATE_PARAMS[DayState.RANGE_DAY].copy()
    
    success_count = 0
    n_trials = 20
    
    for i in range(n_trials):
        gen.current_price = 5050.0
        start_time = datetime(2025, 1, 1)
        filled = False
        
        for _ in range(1440):
            fractal.update(start_time, force_hour_transition=False)
            params = fractal.get_combined_parameters()
            
            config = StateConfig(
                name="dynamic",
                avg_ticks_per_bar=12.0 * params['volatility_mult'],
                ticks_per_bar_std=6.0 * params['volatility_mult'],
                up_probability=params['directional_bias'],
                trend_persistence=params['trend_strength'],
                volatility_multiplier=params['volatility_mult'],
                mean_reversion_strength=params['mean_reversion_strength']
            )
            
            bar = gen.generate_bar(start_time, custom_state_config=config)
            start_time += timedelta(minutes=1)
            
            if bar['low'] <= 5000.0:
                filled = True
                break
        
        if filled:
            success_count += 1
            
    print(f"  Gap Fill Rate: {success_count}/{n_trials} ({success_count/n_trials*100:.1f}%)")

def main():
    print("=" * 60)
    print("GENERATOR CALIBRATION")
    print("=" * 60)
    
    # Test Gap Fill
    test_gap_fill()
    
    # Test Trend Day Up
    test_day_config(DayState.TREND_DAY, direction=1)
    
    # Test Trend Day Down
    test_day_config(DayState.TREND_DAY, direction=-1)
    
    # Test Range Day
    test_day_config(DayState.RANGE_DAY, direction=1)

if __name__ == "__main__":
    main()
