
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import RealDataLoader

def calculate_segment_physics(df_segment):
    """
    Calculate physics metrics for a given price segment.
    """
    if len(df_segment) < 10:
        return None
        
    # 1. Volatility (Std Dev of 1m returns)
    returns = df_segment['close'].pct_change().dropna()
    volatility = returns.std() * 10000 # Scale up for readability
    
    # 2. Trend (Net Change / Total Distance)
    start_price = df_segment['open'].iloc[0]
    end_price = df_segment['close'].iloc[-1]
    net_change = abs(end_price - start_price)
    
    # Total distance traveled (sum of absolute 1m moves)
    total_distance = df_segment['close'].diff().abs().sum()
    
    if total_distance == 0:
        trend_efficiency = 0
    else:
        trend_efficiency = net_change / total_distance
        
    # 3. Direction (Signed Return)
    direction = (end_price - start_price) / start_price
    
    # 4. Wicks (Average Wick Size relative to Body)
    # This is a bit rough, but gives an idea of "messiness"
    bodies = (df_segment['close'] - df_segment['open']).abs()
    ranges = (df_segment['high'] - df_segment['low'])
    wicks = ranges - bodies
    avg_wick_ratio = (wicks / ranges).mean() if ranges.sum() > 0 else 0
    
    return {
        'volatility': volatility,
        'trend_efficiency': trend_efficiency,
        'direction': direction,
        'wick_ratio': avg_wick_ratio,
        'start_time': df_segment.index[0],
        'end_time': df_segment.index[-1]
    }

def build_library():
    input_path = "src/data/continuous_contract.json"
    output_path = "data/processed/physics_library.parquet"
    
    print(f"Loading real data from {input_path}...")
    loader = RealDataLoader()
    df = loader.load_json(input_path)
    
    print("Slicing into 2-hour segments...")
    
    segments = []
    
    # Iterate through the data in 2-hour chunks
    # We want rolling windows? Or non-overlapping?
    # User said "look at all the stories". Overlapping gives more stories.
    # Let's do rolling windows with a step of 30 minutes.
    
    window_size = timedelta(hours=2)
    step_size = timedelta(minutes=30)
    
    start_time = df.index[0]
    end_time = df.index[-1]
    
    current_time = start_time
    
    count = 0
    
    while current_time + window_size <= end_time:
        if count % 1000 == 0:
            print(f"Processed {count} segments...")
            
        segment_end = current_time + window_size
        
        # Slice
        mask = (df.index >= current_time) & (df.index < segment_end)
        df_segment = df[mask]
        
        if not df_segment.empty:
            physics = calculate_segment_physics(df_segment)
            
            if physics:
                # Add Context
                physics['hour'] = current_time.hour
                physics['minute'] = current_time.minute
                
                # Previous Context (Last 2 hours before this segment)
                # This is what we use to query.
                # "If previous 2 hours were X, what happened next?"
                prev_start = current_time - window_size
                mask_prev = (df.index >= prev_start) & (df.index < current_time)
                df_prev = df[mask_prev]
                
                if not df_prev.empty:
                    prev_physics = calculate_segment_physics(df_prev)
                    if prev_physics:
                        physics['prev_volatility'] = prev_physics['volatility']
                        physics['prev_trend'] = prev_physics['trend_efficiency']
                        physics['prev_direction'] = prev_physics['direction']
                        
                        segments.append(physics)
        
        current_time += step_size
        count += 1
        
    print(f"Created {len(segments)} segments.")
    
    df_library = pd.DataFrame(segments)
    
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_library.to_parquet(output_path)
    print(f"Saved physics library to {output_path}")
    print(df_library.head())

if __name__ == "__main__":
    build_library()
