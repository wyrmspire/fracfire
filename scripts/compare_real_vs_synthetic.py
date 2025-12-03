"""
Compare Real vs Synthetic Data

Generates side-by-side comparison charts for Real Data vs Synthetic Data
across multiple timeframes (Daily, Hourly, 15m, 1m).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.data.loader import RealDataLoader

def load_synthetic_data(data_dir: Path) -> pd.DataFrame:
    """Load all synthetic validation parquet files and combine them"""
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

def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 1m data to target timeframe"""
    ohlc = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    return df.resample(timeframe).agg(ohlc).dropna()

def plot_candlesticks(ax, df, title):
    """Plot candlesticks on the given axes"""
    # Define colors
    up_color = '#26a69a'   # Green
    down_color = '#ef5350' # Red
    wick_color = '#666666' # Gray
    
    # Calculate colors for each bar
    colors = [up_color if c >= o else down_color for c, o in zip(df['close'], df['open'])]
    
    # Plot wicks
    ax.vlines(df.index, df['low'], df['high'], color=wick_color, linewidth=1, alpha=0.6)
    
    # Plot bodies
    # Use bar width proportional to timeframe
    # We need to estimate width based on index frequency or diff
    if len(df) > 1:
        min_diff = (df.index[1] - df.index[0]).total_seconds()
        width = min_diff / (24*3600) * 0.8 # Convert seconds to days for matplotlib date axis? 
        # Actually matplotlib bar width on datetime index is in days.
        # 1 minute = 1/(24*60) days
        width = min_diff / 86400 * 0.8
    else:
        width = 0.0005 # Default small width
        
    # Matplotlib bar chart for bodies
    # Bottom is min(open, close), height is abs(close-open)
    bottoms = df[['open', 'close']].min(axis=1)
    heights = (df['close'] - df['open']).abs()
    
    # Ensure minimum height for dojis so they are visible
    heights = heights.replace(0, 0.01) # Min height 0.01 point
    
    ax.bar(df.index, heights, bottom=bottoms, width=width, color=colors, align='center')
    
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.set_ylabel("Price")

def plot_comparison(real_df: pd.DataFrame, synth_df: pd.DataFrame, timeframe: str, title: str, filename: str):
    """Plot Real vs Synthetic side-by-side using Candlesticks"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Calculate ranges for title stats
    real_range = real_df['high'].max() - real_df['low'].min()
    synth_range = synth_df['high'].max() - synth_df['low'].min()
    
    # Plot Real
    plot_candlesticks(ax1, real_df, f"REAL DATA - {title}\nRange: {real_range:.2f}")
    
    # Plot Synthetic
    plot_candlesticks(ax2, synth_df, f"SYNTHETIC DATA - {title}\nRange: {synth_range:.2f}")
    
    plt.tight_layout()
    output_path = root / "out" / "charts" / filename
    plt.savefig(output_path)
    print(f"Saved {output_path}")
    plt.close()

def main():
    print("=" * 60)
    print("COMPARING REAL VS SYNTHETIC DATA")
    print("=" * 60)
    
    # 1. Load Data
    print("Loading Real Data...")
    loader = RealDataLoader()
    real_1m = loader.load_json(root / "src" / "data" / "continuous_contract.json")
    # real_1m.set_index('time', inplace=True) # Loader already sets index
    
    print("Loading Synthetic Data...")
    synth_path = root / "out" / "data" / "synthetic" / "validation"
    synth_1m = load_synthetic_data(synth_path)
    
    # 2. Macro Comparison (Daily)
    print("\nGenerating Macro Comparison (Daily)...")
    real_daily = resample_data(real_1m, 'D')
    synth_daily = resample_data(synth_1m, 'D')
    
    # Slice Real to match Synthetic duration (90 days)
    start_idx = np.random.randint(0, len(real_daily) - 90)
    real_daily_slice = real_daily.iloc[start_idx : start_idx + 90]
    
    plot_comparison(real_daily_slice, synth_daily, 'D', "3-Month Daily View", "compare_macro_daily.png")
    
    # 3. Swing Comparison (Hourly)
    print("Generating Swing Comparison (Hourly)...")
    real_1h = resample_data(real_1m, '1h')
    synth_1h = resample_data(synth_1m, '1h')
    
    # Pick random 1-week slice (5 trading days * 24 hours = 120 bars)
    # Actually continuous contract has gaps, so let's just take 120 bars
    start_real = np.random.randint(0, len(real_1h) - 120)
    real_1h_slice = real_1h.iloc[start_real : start_real + 120]
    
    start_synth = np.random.randint(0, len(synth_1h) - 120)
    synth_1h_slice = synth_1h.iloc[start_synth : start_synth + 120]
    
    plot_comparison(real_1h_slice, synth_1h_slice, '1h', "1-Week Hourly View", "compare_swing_hourly.png")
    
    # 4. Intraday Comparison (5-Minute)
    print("Generating Intraday Comparison (5-Minute)...")
    real_5m = resample_data(real_1m, '5min')
    synth_5m = resample_data(synth_1m, '5min')
    
    # Pick random 1-day slice (24 hours * 12 bars = 288 bars)
    start_real = np.random.randint(0, len(real_5m) - 288)
    real_5m_slice = real_5m.iloc[start_real : start_real + 288]
    
    start_synth = np.random.randint(0, len(synth_5m) - 288)
    synth_5m_slice = synth_5m.iloc[start_synth : start_synth + 288]
    
    plot_comparison(real_5m_slice, synth_5m_slice, '5min', "1-Day 5-Minute View", "compare_intraday_5m.png")
    
    # 5. Micro Comparison (1-Minute)
    print("Generating Micro Comparison (1-Minute)...")
    
    # Pick random 2-hour slice (120 bars)
    start_real = np.random.randint(0, len(real_1m) - 120)
    real_1m_slice = real_1m.iloc[start_real : start_real + 120]
    
    start_synth = np.random.randint(0, len(synth_1m) - 120)
    synth_1m_slice = synth_1m.iloc[start_synth : start_synth + 120]
    
    plot_comparison(real_1m_slice, synth_1m_slice, '1min', "2-Hour 1-Minute View", "compare_micro_1m.png")

if __name__ == "__main__":
    main()
