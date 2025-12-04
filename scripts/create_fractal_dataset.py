
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

def normalize_sequence(df_seq, base_price, base_vol):
    """Normalize price/volume sequence relative to base."""
    # Prices: % diff from base
    # Volume: ratio to base (log?)
    
    # We use simple % diff for prices
    # For volume, we normalize by base_vol (e.g. avg volume of window)
    
    seq = np.zeros((len(df_seq), 5)) # OHLCV
    
    seq[:, 0] = (df_seq['open'] / base_price) - 1.0
    seq[:, 1] = (df_seq['high'] / base_price) - 1.0
    seq[:, 2] = (df_seq['low'] / base_price) - 1.0
    seq[:, 3] = (df_seq['close'] / base_price) - 1.0
    
    # Volume: Add epsilon
    seq[:, 4] = (df_seq['volume'] / (base_vol + 1e-9)) - 1.0
    
    return seq

def create_fractal_dataset():
    input_path = "src/data/continuous_contract.json"
    output_path = "data/processed/fractal_dataset.npz"
    
    print(f"Loading real data from {input_path}...")
    loader = RealDataLoader()
    df_1m = loader.load_json(input_path)
    
    print("Resampling data (Meso/Macro)...")
    
    # Meso: 15m
    ohlc = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    df_15m = df_1m.resample('15min').agg(ohlc).dropna()
    
    # Macro: 1h
    df_1h = df_1m.resample('1h').agg(ohlc).dropna()
    
    print(f"1m bars: {len(df_1m)}")
    print(f"15m bars: {len(df_15m)}")
    print(f"1h bars: {len(df_1h)}")
    
    # Configuration
    micro_len = 240 # 4 hours of 1m
    meso_len = 96   # 24 hours of 15m
    macro_len = 120 # 5 days of 1h
    
    target_len = 60 # Next 1 hour of 1m bars
    
    # We iterate by 1 hour steps
    step_size = timedelta(hours=1)
    
    start_time = df_1m.index[0] + timedelta(days=7) # Warmup for macro
    end_time = df_1m.index[-1] - timedelta(hours=2) # Buffer for target
    
    current_time = start_time
    
    X_micro = []
    X_meso = []
    X_macro = []
    Y_targets = []
    
    count = 0
    
    print("Building dataset...")
    
    # Pre-compute indices for speed? 
    # Pandas slicing by time is okay but slow in loop.
    # Let's use searchsorted on index if possible, or just standard slicing.
    
    while current_time < end_time:
        if count % 1000 == 0:
            print(f"Processed {count} samples...")
            
        # 1. Define Windows
        t_now = current_time
        t_micro_start = t_now - timedelta(minutes=micro_len)
        t_meso_start = t_now - timedelta(minutes=meso_len * 15)
        t_macro_start = t_now - timedelta(hours=macro_len)
        t_target_end = t_now + timedelta(minutes=target_len)
        
        # 2. Extract Dataframes
        # Micro
        mask_micro = (df_1m.index > t_micro_start) & (df_1m.index <= t_now)
        df_micro = df_1m[mask_micro]
        
        # Meso
        mask_meso = (df_15m.index > t_meso_start) & (df_15m.index <= t_now)
        df_meso = df_15m[mask_meso]
        
        # Macro
        mask_macro = (df_1h.index > t_macro_start) & (df_1h.index <= t_now)
        df_macro = df_1h[mask_macro]
        
        # Target (Next 1h)
        mask_target = (df_1m.index > t_now) & (df_1m.index <= t_target_end)
        df_target = df_1m[mask_target]
        
        # Check lengths (allow small missing data but not too much)
        if (len(df_micro) >= micro_len * 0.9 and 
            len(df_meso) >= meso_len * 0.9 and 
            len(df_macro) >= macro_len * 0.9 and 
            len(df_target) >= target_len * 0.9):
            
            # 3. Normalize Inputs
            # Base is current close (t_now)
            base_price = df_micro['close'].iloc[-1]
            base_vol = df_micro['volume'].mean() # Use micro avg vol as base
            
            # Pad or Trim to exact length
            # We take the *last* N elements
            
            # Helper to process array
            def process_array(df, length):
                arr = normalize_sequence(df, base_price, base_vol)
                if len(arr) < length:
                    # Pad with zeros at start
                    pad = np.zeros((length - len(arr), 5))
                    arr = np.vstack([pad, arr])
                elif len(arr) > length:
                    # Trim
                    arr = arr[-length:]
                return arr
                
            x_mic = process_array(df_micro, micro_len)
            x_mes = process_array(df_meso, meso_len)
            x_mac = process_array(df_macro, macro_len)
            
            # 4. Calculate Targets (The "20 Questions")
            # We want to predict the physics of the next hour.
            
            # Target 1: Net Displacement (Points)
            target_close = df_target['close'].iloc[-1]
            displacement = target_close - base_price
            
            # Target 2: High Excursion (Points)
            max_high = df_target['high'].max()
            high_exc = max_high - base_price
            
            # Target 3: Low Excursion (Points)
            min_low = df_target['low'].min()
            low_exc = min_low - base_price
            
            # Target 4: Path Length (Total Distance / Net Move) -> Efficiency
            # If Net Move is small, this explodes.
            # Let's just predict Total Distance (Volatility proxy)
            total_dist = df_target['close'].diff().abs().sum()
            
            # Target 5: Close Location (0.0 = Low, 1.0 = High)
            rng = max_high - min_low
            if rng == 0:
                close_loc = 0.5
            else:
                close_loc = (target_close - min_low) / rng
                
            # Vector: [Displacement, HighExc, LowExc, TotalDist, CloseLoc]
            y = np.array([displacement, high_exc, low_exc, total_dist, close_loc])
            
            X_micro.append(x_mic)
            X_meso.append(x_mes)
            X_macro.append(x_mac)
            Y_targets.append(y)
            
        current_time += step_size
        count += 1
        
    # Convert to numpy arrays
    X_micro = np.array(X_micro)
    X_meso = np.array(X_meso)
    X_macro = np.array(X_macro)
    Y_targets = np.array(Y_targets)
    
    print(f"Dataset Shapes:")
    print(f"Micro: {X_micro.shape}")
    print(f"Meso: {X_meso.shape}")
    print(f"Macro: {X_macro.shape}")
    print(f"Targets: {Y_targets.shape}")
    
    # Save
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        X_micro=X_micro,
        X_meso=X_meso,
        X_macro=X_macro,
        Y_targets=Y_targets
    )
    print(f"Saved fractal dataset to {output_path}")

if __name__ == "__main__":
    create_fractal_dataset()
