"""
Measure Market Physics

Calculates the "Golden Truth" metrics from real market data to calibrate the generator.
Metrics include:
- Daily Range (Mean, Std, Max)
- 3-Month Drift (Mean, Max)
- Wick Ratio (Mean)
- Gap Size (Mean)
- Runner Day Probability (Days > 2x ADR)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.data.loader import RealDataLoader

def calculate_physics_metrics(df: pd.DataFrame) -> dict:
    """Calculate comprehensive physics metrics"""
    # 1. Daily Metrics
    daily = df.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    daily['range'] = daily['high'] - daily['low']
    daily['body'] = (daily['close'] - daily['open']).abs()
    daily['wick'] = daily['range'] - daily['body']
    daily['wick_ratio'] = daily['wick'] / daily['range']
    
    # Gaps (Open - Prev Close)
    daily['prev_close'] = daily['close'].shift(1)
    daily['gap'] = (daily['open'] - daily['prev_close']).abs()
    
    # 2. Drift Analysis (Rolling 3-Month / 63 Days)
    # Calculate net move over 63 trading days
    daily['drift_3m'] = (daily['close'] - daily['close'].shift(63)).abs()
    
    # 3. Runner Days
    # Define Runner as Range > 2 * Median Range
    median_range = daily['range'].median()
    runner_threshold = 2.0 * median_range
    runner_days = daily[daily['range'] > runner_threshold]
    runner_prob = len(runner_days) / len(daily)
    
    metrics = {
        'daily_range_mean': daily['range'].mean(),
        'daily_range_std': daily['range'].std(),
        'daily_range_median': median_range,
        'daily_range_max': daily['range'].max(),
        
        'wick_ratio_mean': daily['wick_ratio'].mean(),
        
        'gap_mean': daily['gap'].mean(),
        'gap_std': daily['gap'].std(),
        
        'drift_3m_mean': daily['drift_3m'].mean(),
        'drift_3m_max': daily['drift_3m'].max(),
        
        'runner_prob': runner_prob,
        'runner_threshold': runner_threshold
    }
    
    return metrics

def main():
    print("=" * 60)
    print("MARKET PHYSICS MEASUREMENT (Golden Truth)")
    print("=" * 60)
    
    # Load Real Data
    print("Loading Real Data...")
    loader = RealDataLoader()
    real_1m = loader.load_json(root / "src" / "data" / "continuous_contract.json")
    
    # Calculate Metrics
    metrics = calculate_physics_metrics(real_1m)
    
    print("\nDaily Dynamics:")
    print(f"  Range Mean:      {metrics['daily_range_mean']:.2f}")
    print(f"  Range Median:    {metrics['daily_range_median']:.2f}")
    print(f"  Range Std:       {metrics['daily_range_std']:.2f}")
    print(f"  Range Max:       {metrics['daily_range_max']:.2f}")
    print(f"  Wick Ratio:      {metrics['wick_ratio_mean']:.1%}")
    print(f"  Gap Mean:        {metrics['gap_mean']:.2f}")
    
    print("\nLong-Term Drift (3-Month / 63 Days):")
    print(f"  Drift Mean:      {metrics['drift_3m_mean']:.2f}")
    print(f"  Drift Max:       {metrics['drift_3m_max']:.2f}")
    
    print("\nRunner Days:")
    print(f"  Threshold:       > {metrics['runner_threshold']:.2f} pts")
    print(f"  Probability:     {metrics['runner_prob']:.1%}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
