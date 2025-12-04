
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.generator.engine import PriceGenerator

def generate_synthetic_data(days=65, output_path="data/processed/synthetic_3m.parquet"):
    """
    Generate synthetic data, label it for 3R setups, and save to Parquet.
    """
    print(f"Generating {days} days of synthetic data...")
    
    gen = PriceGenerator(initial_price=5000.0, seed=123)
    start_date = datetime(2024, 1, 1)
    all_bars = []
    
    current_date = start_date
    days_generated = 0
    
    while days_generated < days:
        if current_date.weekday() >= 5: # Skip weekends
            current_date += timedelta(days=1)
            continue
            
        # Generate full day
        df_day = gen.generate_day(
            start_date=current_date,
            auto_transition=True
        )
        all_bars.append(df_day)
        
        current_date += timedelta(days=1)
        days_generated += 1
        
    print("Concatenating and processing...")
    df = pd.concat(all_bars)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    
    # Labeling for 3R Setups
    # 1. Calculate Stop Loss (Min/Max of last 60 mins)
    # We need a rolling window. 
    # For Longs: Stop = Lowest Low of last 60m
    # For Shorts: Stop = Highest High of last 60m
    
    # Use 1m bars for precision
    window_size = 60
    
    df['rolling_low'] = df['low'].rolling(window=window_size).min()
    df['rolling_high'] = df['high'].rolling(window=window_size).max()
    
    # Shift rolling values so we don't look ahead (stop is based on PAST 60m)
    # Actually, for the current bar, the "last 60m" includes the current bar? 
    # User said "compare it to the 1 hour before". So strictly past 60m.
    # Let's shift by 1 to exclude current bar from the "past" window if we want strictness,
    # but usually "last 60 candles" includes current.
    # Let's use shift(1) to be safe - stop is determined by price action BEFORE entry.
    df['stop_long'] = df['rolling_low'].shift(1)
    df['stop_short'] = df['rolling_high'].shift(1)
    
    # 2. Calculate Targets and Labels (Future looking)
    # This is computationally expensive to do iteratively. Vectorized approach?
    # For each bar, we have Entry (Close), Stop, Target.
    # We need to check if Price hits Target before Stop in the FUTURE.
    
    # We can limit the "hold time" to avoid infinite lookahead. Say max 4 hours (240 bars).
    max_hold = 240
    
    # Prepare arrays for fast access
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    stop_longs = df['stop_long'].values
    stop_shorts = df['stop_short'].values
    
    labels_long = np.zeros(len(df), dtype=int)
    labels_short = np.zeros(len(df), dtype=int)
    
    print("Labeling data (this may take a minute)...")
    
    # Iterate (could be optimized with Numba but Python loop is okay for 65 days ~ 25k bars)
    # 65 days * 1440 bars = 93,600 bars. Loop might be slow.
    # Let's try a slightly optimized loop.
    
    n = len(df)
    
    for i in range(window_size, n - 1):
        if i % 10000 == 0:
            print(f"Processed {i}/{n} bars")
            
        entry = closes[i]
        
        # Long Setup
        stop = stop_longs[i]
        if not np.isnan(stop) and entry > stop:
            risk = entry - stop
            target = entry + (3 * risk)
            
            # Check future
            end_idx = min(i + max_hold, n)
            future_highs = highs[i+1:end_idx]
            future_lows = lows[i+1:end_idx]
            
            # Did we hit target?
            hit_target_indices = np.where(future_highs >= target)[0]
            hit_stop_indices = np.where(future_lows <= stop)[0]
            
            if len(hit_target_indices) > 0:
                first_target = hit_target_indices[0]
                # Did we hit stop first?
                if len(hit_stop_indices) > 0:
                    first_stop = hit_stop_indices[0]
                    if first_target < first_stop:
                        labels_long[i] = 1
                else:
                    labels_long[i] = 1
                    
        # Short Setup
        stop = stop_shorts[i]
        if not np.isnan(stop) and entry < stop:
            risk = stop - entry
            target = entry - (3 * risk)
            
            # Check future
            end_idx = min(i + max_hold, n)
            future_highs = highs[i+1:end_idx]
            future_lows = lows[i+1:end_idx]
            
            # Did we hit target?
            hit_target_indices = np.where(future_lows <= target)[0]
            hit_stop_indices = np.where(future_highs >= stop)[0]
            
            if len(hit_target_indices) > 0:
                first_target = hit_target_indices[0]
                # Did we hit stop first?
                if len(hit_stop_indices) > 0:
                    first_stop = hit_stop_indices[0]
                    if first_target < first_stop:
                        labels_short[i] = 1
                else:
                    labels_short[i] = 1

    df['label_long_3r'] = labels_long
    df['label_short_3r'] = labels_short
    
    # Drop NaN rows (start of data)
    df.dropna(inplace=True)
    
    # Save
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path)
    print(f"Saved synthetic training data to {output_path}")
    print(f"Total Bars: {len(df)}")
    print(f"Long 3R Wins: {df['label_long_3r'].sum()} ({df['label_long_3r'].mean():.2%})")
    print(f"Short 3R Wins: {df['label_short_3r'].sum()} ({df['label_short_3r'].mean():.2%})")

if __name__ == "__main__":
    generate_synthetic_data()
