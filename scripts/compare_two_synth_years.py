import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.core.generator import PriceGenerator, PhysicsConfig

def generate_year(seed, start_price=5000.0, vol=2.5):
    """Generate 1 year of 1-minute data (approx 375,000 minutes)."""
    # 260 trading days * 1440 minutes = 374,400
    # Let's generate a fixed number of minutes for a "year"
    n_minutes = 260 * 1440
    
    # Create a dummy datetime index for 1 year
    # We don't care about exact dates, just the sequence
    timestamps = pd.date_range(start="2024-01-01", periods=n_minutes, freq="1min")
    
    physics = PhysicsConfig(base_volatility=vol, wick_probability=0.5)
    gen = PriceGenerator(initial_price=start_price, seed=seed, physics_config=physics)
    
    print(f"Generating year with seed {seed}...")
    df = gen.generate_batch(timestamps)
    return df

def resample_ohlcv(df, timeframe):
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df.resample(timeframe, on='time').agg(ohlc).dropna()

def plot_candlesticks(ax, df, title):
    up_color = '#26a69a'
    down_color = '#ef5350'
    wick_color = '#666666'
    
    colors = [up_color if c >= o else down_color for c, o in zip(df['close'], df['open'])]
    
    x = np.arange(len(df))
    ax.vlines(x, df['low'], df['high'], color=wick_color, linewidth=1, alpha=0.6)
    
    width = 0.6
    bottoms = df[['open', 'close']].min(axis=1)
    heights = (df['close'] - df['open']).abs().replace(0, 0.01)
    
    ax.bar(x, heights, bottom=bottoms, width=width, color=colors, align='center')
    
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.set_ylabel("Price")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vol', type=float, default=2.5)
    parser.add_argument('--seed1', type=int, default=42)
    parser.add_argument('--seed2', type=int, default=999)
    args = parser.parse_args()
    
    # Generate two independent years
    df1 = generate_year(args.seed1, vol=args.vol)
    df2 = generate_year(args.seed2, vol=args.vol)
    
    # Resample to Daily
    # But our data is continuous minutes.
    # Let's resample to '1D' using the datetime index we created
    df1_w = resample_ohlcv(df1, '1D')
    df2_w = resample_ohlcv(df2, '1D')
    
    print(f"Year 1 (Seed {args.seed1}): {len(df1_w)} days")
    print(f"Year 2 (Seed {args.seed2}): {len(df2_w)} days")
    
    # Plot side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    plot_candlesticks(ax1, df1_w, f"Synthetic Year 1 (Seed {args.seed1}, Vol {args.vol})")
    plot_candlesticks(ax2, df2_w, f"Synthetic Year 2 (Seed {args.seed2}, Vol {args.vol})")
    
    plt.tight_layout()
    out_path = root / "out" / "charts" / "compare_two_synth_years.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
