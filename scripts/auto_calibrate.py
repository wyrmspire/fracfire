"""
Auto-Calibrate Generator

This script performs a 3-step calibration of the FracFire generator against real market data:
1.  **Session Profiling**: Analyzes real volume/volatility by hour to recommend Session Configs.
2.  **Volatility Sweep**: Sweeps `base_volatility` to match daily range and 15m volatility.
3.  **Wick Sweep**: Sweeps `wick_probability` to match candle wick ratios.

Usage:
    python scripts/auto_calibrate.py --days 30
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.core.generator import PriceGenerator, PhysicsConfig
from src.data.loader import RealDataLoader

def load_real_data(days: int):
    """Load the most recent `days` of real 1m data."""
    loader = RealDataLoader()
    real_path = root / "src" / "data" / "continuous_contract.json"
    df = loader.load_json(real_path)
    
    # Filter to last N days
    end_date = df.index[-1]
    start_date = end_date - pd.Timedelta(days=days)
    df = df[df.index >= start_date]
    return df

def analyze_session_profile(df: pd.DataFrame):
    """Analyze volume and range by hour of day."""
    print("\n--- Step 1: Session Profile Analysis ---")
    
    # Calculate hourly metrics
    df['hour'] = df.index.hour
    df['range'] = df['high'] - df['low']
    
    hourly_stats = df.groupby('hour').agg({
        'volume': 'mean',
        'range': 'mean'
    })
    
    # Normalize to get multipliers (baseline = mean of all hours)
    vol_baseline = hourly_stats['volume'].mean()
    range_baseline = hourly_stats['range'].mean()
    
    hourly_stats['vol_mult'] = hourly_stats['volume'] / vol_baseline
    hourly_stats['range_mult'] = hourly_stats['range'] / range_baseline
    
    # Ensure all hours exist
    hourly_stats = hourly_stats.reindex(range(24))
    
    print(f"{'Hour':<6} {'Vol Mult':<10} {'Range Mult':<10}")
    print("-" * 30)
    for hour, row in hourly_stats.iterrows():
        print(f"{hour:<6} {row['vol_mult']:.2f}       {row['range_mult']:.2f}")
        
    # Define sessions roughly
    # Asian: 18-3
    # London: 3-8
    # RTH: 9-16
    
    print("\nRecommended Session Multipliers (Approximate):")
    asian = hourly_stats.loc[[18,19,20,21,22,23,0,1,2]].mean()
    london = hourly_stats.loc[[3,4,5,6,7,8]].mean()
    rth = hourly_stats.loc[[9,10,11,12,13,14,15,16]].mean()
    
    print(f"ASIAN:  Vol={asian['vol_mult']:.2f}, Volatility={asian['range_mult']:.2f}")
    print(f"LONDON: Vol={london['vol_mult']:.2f}, Volatility={london['range_mult']:.2f}")
    print(f"RTH:    Vol={rth['vol_mult']:.2f}, Volatility={rth['range_mult']:.2f}")
    
    return hourly_stats

def generate_synthetic(real_index: pd.DatetimeIndex, physics: PhysicsConfig, seed: int = 42):
    """Generate synthetic data on the exact timestamps of real data."""
    gen = PriceGenerator(
        initial_price=5000.0,
        seed=seed,
        physics_config=physics
    )
    
    bars = []
    # Batch generation for speed? No, engine is tick-by-tick.
    # We can use generate_bar in a loop.
    for ts in real_index:
        bar = gen.generate_bar(ts)
        bars.append(bar)
        
    df = pd.DataFrame(bars)
    df.set_index('time', inplace=True)
    return df

def sweep_volatility(real_df: pd.DataFrame):
    """Sweep base_volatility to match daily range."""
    print("\n--- Step 2: Volatility Sweep ---")
    
    # Real Metrics
    real_daily = real_df.resample('1D').agg({'high':'max', 'low':'min'}).dropna()
    real_daily_range = (real_daily['high'] - real_daily['low']).mean()
    print(f"Target Daily Range: {real_daily_range:.2f}")
    
    scales = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    best_score = float('inf')
    best_scale = 1.0
    
    results = []
    
    # Use a smaller subset for sweep to be faster (e.g. 1 week)
    subset_df = real_df.iloc[:5*24*60] # 5 days
    
    for scale in tqdm(scales, desc="Sweeping Volatility"):
        physics = PhysicsConfig(base_volatility=scale)
        synth_df = generate_synthetic(subset_df.index, physics)
        
        synth_daily = synth_df.resample('1D').agg({'high':'max', 'low':'min'}).dropna()
        synth_daily_range = (synth_daily['high'] - synth_daily['low']).mean()
        
        score = abs(synth_daily_range - real_daily_range)
        results.append((scale, synth_daily_range, score))
        
        if score < best_score:
            best_score = score
            best_scale = scale
            
    print(f"\n{'Scale':<6} {'Range':<10} {'Error':<10}")
    print("-" * 30)
    for r in results:
        print(f"{r[0]:<6.1f} {r[1]:<10.2f} {r[2]:<10.2f}")
        
    print(f"\nBest Base Volatility: {best_scale}")
    return best_scale

def sweep_wicks(real_df: pd.DataFrame, best_vol: float):
    """Sweep wick_probability to match wick ratio."""
    print("\n--- Step 3: Wick Sweep ---")
    
    # Real Metrics (using 15m bars)
    real_15m = real_df.resample('15min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()
    real_body = (real_15m['close'] - real_15m['open']).abs()
    real_range = real_15m['high'] - real_15m['low']
    real_wick_ratio = 1.0 - (real_body / real_range.replace(0, 1))
    target_wick = real_wick_ratio.mean()
    print(f"Target Wick Ratio (15m): {target_wick:.3f}")
    
    probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    best_score = float('inf')
    best_prob = 0.2
    
    results = []
    subset_df = real_df.iloc[:5*24*60]
    
    for prob in tqdm(probs, desc="Sweeping Wicks"):
        physics = PhysicsConfig(base_volatility=best_vol, wick_probability=prob)
        synth_df = generate_synthetic(subset_df.index, physics)
        
        synth_15m = synth_df.resample('15min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()
        synth_body = (synth_15m['close'] - synth_15m['open']).abs()
        synth_range = synth_15m['high'] - synth_15m['low']
        synth_wick_ratio = 1.0 - (synth_body / synth_range.replace(0, 1))
        avg_wick = synth_wick_ratio.mean()
        
        score = abs(avg_wick - target_wick)
        results.append((prob, avg_wick, score))
        
        if score < best_score:
            best_score = score
            best_prob = prob
            
    print(f"\n{'Prob':<6} {'WickRatio':<10} {'Error':<10}")
    print("-" * 30)
    for r in results:
        print(f"{r[0]:<6.1f} {r[1]:<10.3f} {r[2]:<10.3f}")
        
    print(f"\nBest Wick Probability: {best_prob}")
    return best_prob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=30, help="Days of real data to analyze")
    args = parser.parse_args()
    
    print(f"Loading {args.days} days of real data...")
    real_df = load_real_data(args.days)
    print(f"Loaded {len(real_df)} minutes.")
    
    # 1. Session Analysis
    analyze_session_profile(real_df)
    
    # 2. Volatility Sweep
    best_vol = sweep_volatility(real_df)
    
    # 3. Wick Sweep
    best_wick = sweep_wicks(real_df, best_vol)
    
    print("\n" + "="*40)
    print("CALIBRATION COMPLETE")
    print("="*40)
    print(f"Recommended PhysicsConfig:")
    print(f"  base_volatility = {best_vol}")
    print(f"  wick_probability = {best_wick}")
    print("="*40)

if __name__ == "__main__":
    main()
