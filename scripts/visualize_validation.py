"""
Visualize Validation Data

Resamples the generated 1-minute validation data to 15-minute candles
and plots the 3-month chart for user verification.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

def main():
    print("=" * 60)
    print("VISUALIZING VALIDATION DATA")
    print("=" * 60)
    
    data_dir = root / "out" / "data" / "synthetic" / "validation"
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
        
    # 1. Load Data
    print("Loading data...")
    files = sorted(list(data_dir.glob("*.parquet")))
    if not files:
        print("No files found.")
        return
        
    dfs = []
    for f in tqdm(files):
        dfs.append(pd.read_parquet(f))
        
    df = pd.concat(dfs, ignore_index=True)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Loaded {len(df)} 1-minute bars.")
    
    # 2. Resample to 15m
    print("Resampling to 15m...")
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    df_15m = df.resample('15min').agg(ohlc_dict).dropna()
    print(f"Resampled to {len(df_15m)} 15-minute bars.")
    
    # 3. Plot
    print("Plotting...")
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot full 3 months
    plt.figure(figsize=(20, 10))
    
    # Simple line chart for speed/clarity on long timeframe
    plt.plot(df_15m.index, df_15m['close'], linewidth=1, color='black', alpha=0.8)
    
    plt.title("3-Month Synthetic Validation Chart (15m)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    output_path = output_dir / "validation_3month_15m.png"
    plt.savefig(output_path, dpi=300)
    print(f"Full chart saved to: {output_path}")
    
    # Plot zoomed in weeks (first 4 weeks)
    for i in range(4):
        start_idx = i * 4 * 5 * 24 * 4 # Approx 1 week of 15m bars (assuming 24h)
        # 1 week = 7 days * 24 hours * 4 bars/hr = 672 bars
        start_idx = i * 672
        end_idx = start_idx + 672
        
        if start_idx >= len(df_15m):
            break
            
        subset = df_15m.iloc[start_idx:end_idx]
        
        plt.figure(figsize=(15, 8))
        plt.plot(subset.index, subset['close'], linewidth=1.5, color='blue')
        plt.title(f"Week {i+1} Zoom (15m)")
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.gcf().autofmt_xdate()
        
        plt.savefig(output_dir / f"validation_week_{i+1}.png")
        print(f"Week {i+1} chart saved.")

if __name__ == "__main__":
    main()
