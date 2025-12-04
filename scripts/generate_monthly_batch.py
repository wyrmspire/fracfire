
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.generator.engine import PriceGenerator, MarketState
from src.core.pipeline.visualizer import plot_scenario
from src.core.pipeline.scenario_runner import ScenarioResult

def load_real_data():
    """Load real data for warmup."""
    path = Path("src/data/continuous_contract.json")
    if not path.exists():
        print(f"Error: {path} not found.")
        return None
        
    print(f"Loading real data from {path}...")
    with open(path, 'r') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    return df

def generate_month(month_idx, real_df, out_dir):
    """Generate a single month of data."""
    print(f"Generating Month {month_idx+1}/12...")
    
    # 1. Select Random Warmup
    # Need at least 5 days of history
    min_idx = 24 * 60 * 5
    max_idx = len(real_df) - min_idx - 1
    
    start_idx = np.random.randint(min_idx, max_idx)
    warmup_end_time = real_df.index[start_idx]
    warmup_start_time = warmup_end_time - timedelta(days=7) # 7 days warmup
    
    warmup_df = real_df[(real_df.index >= warmup_start_time) & (real_df.index < warmup_end_time)].copy()
    
    if len(warmup_df) < 240: # Check if we have enough data
        print("  Warning: Warmup data too short, retrying...")
        return generate_month(month_idx, real_df, out_dir)
        
    # 2. Initialize Generator
    seed = np.random.randint(0, 100000)
    print(f"  Seed: {seed}, Warmup End: {warmup_end_time}")
    
    # Start generation from the end of warmup
    start_time = warmup_end_time
    
    generator = PriceGenerator(
        initial_price=warmup_df['close'].iloc[-1],
        seed=seed,
        warmup_data=warmup_df
    )
    
    # 3. Generate 20 Days (approx 1 month)
    # 20 days * 1440 mins = 28800 bars
    # Using generate_day loop
    days_to_generate = 20
    all_bars = []
    
    current_time = start_time
    for day in range(days_to_generate):
        # Generate full day
        df_day = generator.generate_day(
            start_date=current_time,
            auto_transition=True # Let physics drive states
        )
        all_bars.append(df_day)
        current_time += timedelta(days=1)
        
    df_month = pd.concat(all_bars)
    
    # 4. Process for Plotting
    # Resample to 30m
    df_30m = df_month.resample('30min', on='time').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Create Result object for visualizer
    # Visualizer expects ScenarioResult
    result = ScenarioResult(
        df_1m=df_month,
        df_5m=df_30m, # Hack: passing 30m as 5m for plotting
        outcomes=[],
        metadata={
            "seed": seed,
            "warmup_end": str(warmup_end_time),
            "month_idx": month_idx
        }
    )
    
    # Plot
    filename = f"month_{month_idx+1:02d}_seed_{seed}"
    plot_scenario(result, save_dir=str(out_dir), filename_prefix=filename)
    print(f"  Saved {filename}.png")

def main():
    out_dir = Path("out/monthly_batch")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    real_df = load_real_data()
    if real_df is None:
        return

    for i in range(12):
        generate_month(i, real_df, out_dir)
        
    print("Done.")

if __name__ == "__main__":
    main()
