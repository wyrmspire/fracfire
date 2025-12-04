
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import RealDataLoader

def process_real_data():
    input_path = "src/data/continuous_contract.json"
    output_path = "data/processed/real_3m.parquet"
    
    print(f"Loading real data from {input_path}...")
    loader = RealDataLoader()
    df = loader.load_json(input_path)
    
    # Labeling for 3R Setups (Same logic as synthetic)
    print("Labeling real data...")
    
    window_size = 60
    
    df['rolling_low'] = df['low'].rolling(window=window_size).min()
    df['rolling_high'] = df['high'].rolling(window=window_size).max()
    
    df['stop_long'] = df['rolling_low'].shift(1)
    df['stop_short'] = df['rolling_high'].shift(1)
    
    max_hold = 240
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    stop_longs = df['stop_long'].values
    stop_shorts = df['stop_short'].values
    
    labels_long = np.zeros(len(df), dtype=int)
    labels_short = np.zeros(len(df), dtype=int)
    
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
            
            end_idx = min(i + max_hold, n)
            future_highs = highs[i+1:end_idx]
            future_lows = lows[i+1:end_idx]
            
            hit_target_indices = np.where(future_highs >= target)[0]
            hit_stop_indices = np.where(future_lows <= stop)[0]
            
            if len(hit_target_indices) > 0:
                first_target = hit_target_indices[0]
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
            
            end_idx = min(i + max_hold, n)
            future_highs = highs[i+1:end_idx]
            future_lows = lows[i+1:end_idx]
            
            hit_target_indices = np.where(future_lows <= target)[0]
            hit_stop_indices = np.where(future_highs >= stop)[0]
            
            if len(hit_target_indices) > 0:
                first_target = hit_target_indices[0]
                if len(hit_stop_indices) > 0:
                    first_stop = hit_stop_indices[0]
                    if first_target < first_stop:
                        labels_short[i] = 1
                else:
                    labels_short[i] = 1

    df['label_long_3r'] = labels_long
    df['label_short_3r'] = labels_short
    
    df.dropna(inplace=True)
    
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path)
    print(f"Saved real training data to {output_path}")
    print(f"Total Bars: {len(df)}")
    print(f"Long 3R Wins: {df['label_long_3r'].sum()} ({df['label_long_3r'].mean():.2%})")
    print(f"Short 3R Wins: {df['label_short_3r'].sum()} ({df['label_short_3r'].mean():.2%})")

if __name__ == "__main__":
    process_real_data()
