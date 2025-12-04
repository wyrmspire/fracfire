
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.generator.engine import PriceGenerator, MarketState
from src.core.pipeline.visualizer import plot_candles

def main():
    print("Generating Month-Long Chart (20 Days)...")
    
    out_dir = Path("out/month_chart")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Generator
    gen = PriceGenerator(initial_price=5000.0, seed=42)
    
    start_date = datetime(2024, 1, 1)
    all_bars = []
    
    # Generate 20 days
    days_generated = 0
    current_date = start_date
    
    while days_generated < 20:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
            
        print(f"Generating Day {days_generated + 1}: {current_date.date()}")
        
        # Generate full day (auto-transition)
        df_day = gen.generate_day(
            start_date=current_date,
            auto_transition=True,
            segment_length=None # Allow random transitions
        )
        
        # Filter to RTH + Extended for better chart (e.g. 08:00 to 16:30)
        # Or just keep full 24h? User asked for "appropriate transitions".
        # Let's keep full 24h to show overnight drift, but maybe highlight RTH?
        # For a 15m chart, full 24h is fine.
        
        all_bars.append(df_day)
        
        current_date += timedelta(days=1)
        days_generated += 1
        
    # Combine all days
    print("Concatenating data...")
    full_df = pd.concat(all_bars)
    full_df.set_index('time', inplace=True)
    
    # Resample to 15m
    print("Resampling to 15m...")
    ohlc = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    df_15m = full_df.resample('15min').agg(ohlc).dropna()
    
    # Plotting
    print("Plotting Giant Chart...")
    # Width: 20 days * 96 bars/day (15m) = 1920 bars
    # To see clearly, maybe 5 pixels per bar? ~10,000 px wide
    width_px = 10000
    height_px = 1200
    dpi = 100
    figsize = (width_px / dpi, height_px / dpi)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    plot_candles(ax, df_15m)
    
    ax.set_title("Month-Long Simulation (15m Candles)", fontsize=24)
    ax.set_xlabel("Time", fontsize=16)
    ax.set_ylabel("Price", fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"month_15m_giant_{timestamp}.png"
    filepath = out_dir / filename
    
    print(f"Saving to {filepath} (this may take a moment)...")
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print("Done.")

if __name__ == "__main__":
    main()
