"""
Generate Validation Data

Generates 3 months of continuous synthetic data using the updated FractalStateManager.
This data is used to validate the realism of the generator before pre-training.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from lab.generators.price_generator import PriceGenerator, StateConfig, PhysicsConfig
from lab.generators.fractal_states import FractalStateManager
from lab.generators.custom_states import CUSTOM_STATES

def main():
    print("=" * 60)
    print("GENERATING 3-MONTH VALIDATION DATASET")
    print("=" * 60)
    
    # Configuration
    start_date = datetime(2025, 1, 1, 0, 0, 0)
    days_to_generate = 90
    output_dir = root / "out" / "data" / "synthetic" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    current_time = start_date
    
    # --- PHYSICS CALIBRATION (Golden Truth) ---
    # Metrics from Real Data (measure_market_physics.py):
    # Daily Range Mean: 82.05
    # Daily Range Std: 81.85
    # Runner Prob: 12.1%
    # Drift 3M: 646.60
    
    physics = PhysicsConfig(
        # 1. Volatility / Heat
        # Reduced from 4.0 to "turn down the heat" as requested
        avg_ticks_per_bar=3.5,
        
        # 2. Daily Structure
        daily_range_mean=82.0,
        daily_range_std=40.0,  # Slightly lower than real (81) to avoid explosions
        
        # 3. Runner Days
        runner_prob=0.12,      # Match real 12%
        runner_target_mult=3.0, # 3 * 82 = ~246 pts for runners
        
        # 4. Macro Gravity
        # Tightened to match real drift (646)
        macro_gravity_threshold=1000.0, 
        macro_gravity_strength=0.4,
        
        # 5. Micro Physics
        wick_probability=0.15,
        wick_extension_avg=1.5
    )
    
    gen = PriceGenerator(
        initial_price=5000.0, 
        seed=42,
        physics_config=physics
    )
    
    fractal = FractalStateManager(seed=42, physics_config=physics)
    
    # Progress bar for days
    for day in tqdm(range(days_to_generate)):
        day_bars = []
        
        # Generate 1440 minutes (24 hours)
        for _ in range(1440):
            # Update fractal state
            fractal.update(current_time, current_price=gen.current_price)
            params = fractal.get_combined_parameters(current_price=gen.current_price)
            
            # Select Base Config
            base_config = None
            
            # 1. Time-based overrides
            hour = current_time.hour
            minute = current_time.minute
            
            if hour == 9 and 30 <= minute < 45:
                base_config = CUSTOM_STATES['opening_bell']
            elif hour == 15 and 45 <= minute <= 59:
                base_config = CUSTOM_STATES['closing_squeeze']
                
            # 2. State-based overrides
            elif params['hour_state'] == 'volatile': # Assuming lowercase string match
                base_config = CUSTOM_STATES['mega_volatile']
            elif params['hour_state'] == 'choppy':
                base_config = CUSTOM_STATES['whipsaw']
                
            # Construct Final Config
            if base_config:
                # Use custom state as base, but override directional logic with fractal governor
                config = StateConfig(
                    name=f"dynamic_{base_config.name}",
                    avg_ticks_per_bar=base_config.avg_ticks_per_bar * params['volatility_mult'],
                    ticks_per_bar_std=base_config.ticks_per_bar_std,
                    
                    # Direction comes from Fractal Manager (which has the Governor)
                    up_probability=params['directional_bias'],
                    trend_persistence=base_config.trend_persistence, # Keep custom persistence (e.g. choppy)
                    
                    # Volatility
                    volatility_multiplier=base_config.volatility_multiplier * params['volatility_mult'],
                    
                    # Micro-structure from custom state
                    avg_tick_size=base_config.avg_tick_size,
                    tick_size_std=base_config.tick_size_std,
                    max_tick_jump=base_config.max_tick_jump,
                    wick_probability=base_config.wick_probability,
                    wick_extension_avg=base_config.wick_extension_avg,
                    
                    mean_reversion_strength=params['mean_reversion_strength']
                )
            else:
                # Standard Dynamic Config
                config = StateConfig(
                    name="dynamic",
                    avg_ticks_per_bar=12.0 * params['volatility_mult'],
                    ticks_per_bar_std=6.0 * params['volatility_mult'],
                    up_probability=params['directional_bias'],
                    trend_persistence=params['trend_strength'],
                    volatility_multiplier=params['volatility_mult'],
                    
                    # Standard defaults
                    avg_tick_size=1.2,
                    tick_size_std=0.8,
                    max_tick_jump=6,
                    wick_probability=0.3,
                    wick_extension_avg=2.0,
                    
                    mean_reversion_strength=params['mean_reversion_strength']
                )
            # Generate bar
            bar = gen.generate_bar(
                timestamp=current_time,
                custom_state_config=config
            )
            
            # Add metadata
            bar['day_state'] = params['day_state']
            bar['hour_state'] = params['hour_state']
            bar['minute_state'] = params['minute_state']
            bar['day_direction'] = params['day_direction']
            
            day_bars.append(bar)
            current_time += timedelta(minutes=1)
            
        # Save daily file to avoid memory issues
        df = pd.DataFrame(day_bars)
        filename = f"validation_day_{day:03d}.parquet"
        df.to_parquet(output_dir / filename)
        
        # Update Generator's Previous Day Close
        if not df.empty:
            gen.prev_day_close = df['close'].iloc[-1]
            print(f"Day {day} Close: {gen.prev_day_close:.2f}")
        
        # Keep light in memory
        if day % 10 == 0:
            all_bars = [] 
            
    print(f"\nData saved to {output_dir}")
    
    # Now verify/visualize
    # We need to stitch them back together for the chart? 
    # Or just load them in the viz script.
    # Let's create a separate viz script as requested.

if __name__ == "__main__":
    main()
