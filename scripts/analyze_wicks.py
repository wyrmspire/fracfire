"""
Analyze Wicks

Calculates the Wick-to-Body ratio and other wick metrics for Real vs Synthetic data.
Used to tune the generator to produce realistic candle shapes.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.data.loader import RealDataLoader

def load_synthetic_data(data_dir: Path) -> pd.DataFrame:
    """Load all synthetic validation parquet files"""
    files = sorted(list(data_dir.glob("validation_day_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No validation files found in {data_dir}")
        
    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df['time'] = pd.to_datetime(full_df['time'])
    full_df.set_index('time', inplace=True)
    return full_df

def calculate_wick_metrics(df: pd.DataFrame, timeframe: str) -> dict:
    """Calculate wick metrics for a given timeframe"""
    # Resample
    ohlc = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }
    resampled = df.resample(timeframe).agg(ohlc).dropna()
    
    # Calculate dimensions
    resampled['range'] = resampled['high'] - resampled['low']
    resampled['body'] = (resampled['close'] - resampled['open']).abs()
    resampled['upper_wick'] = resampled['high'] - resampled[['open', 'close']].max(axis=1)
    resampled['lower_wick'] = resampled[['open', 'close']].min(axis=1) - resampled['low']
    resampled['total_wick'] = resampled['upper_wick'] + resampled['lower_wick']
    
    # Avoid division by zero
    resampled = resampled[resampled['range'] > 0]
    
    # Metrics
    metrics = {
        'avg_range': resampled['range'].mean(),
        'avg_body': resampled['body'].mean(),
        'avg_wick': resampled['total_wick'].mean(),
        'wick_ratio': resampled['total_wick'].sum() / resampled['range'].sum()
    }
    return metrics

def main():
    print("=" * 60)
    print("WICK ANALYSIS (Real vs Synthetic)")
    print("=" * 60)
    
    # 1. Load Data
    print("Loading Real Data...")
    loader = RealDataLoader()
    real_1m = loader.load_json(root / "src" / "data" / "continuous_contract.json")
    # Loader sets index to 'time'
    
    print("Loading Synthetic Data...")
    synth_path = root / "out" / "data" / "synthetic" / "validation"
    synth_1m = load_synthetic_data(synth_path)
    
    # 2. Analyze Timeframes
    timeframes = ['1D', '1h', '15min', '5min']
    
    print(f"\n{'Timeframe':<10} {'Metric':<15} {'Real':<15} {'Synthetic':<15} {'Diff %':<10}")
    print("-" * 70)
    
    for tf in timeframes:
        real_metrics = calculate_wick_metrics(real_1m, tf)
        synth_metrics = calculate_wick_metrics(synth_1m, tf)
        
        # Print Wick Ratio
        r_ratio = real_metrics['wick_ratio'] * 100
        s_ratio = synth_metrics['wick_ratio'] * 100
        diff = ((s_ratio - r_ratio) / r_ratio) * 100
        
        print(f"{tf:<10} {'Wick Ratio %':<15} {r_ratio:>6.1f}%        {s_ratio:>6.1f}%        {diff:>+6.1f}%")
        
        # Print Avg Range
        r_range = real_metrics['avg_range']
        s_range = synth_metrics['avg_range']
        diff_range = ((s_range - r_range) / r_range) * 100
        print(f"{tf:<10} {'Avg Range':<15} {r_range:>6.2f}          {s_range:>6.2f}          {diff_range:>+6.1f}%")
        print("-" * 70)

if __name__ == "__main__":
    main()
