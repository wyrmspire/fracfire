"""
Visualize Real Data Predictions

Plots the real price data colored by the predicted market state.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

def main():
    print("=" * 60)
    print("VISUALIZING REAL DATA PREDICTIONS")
    print("=" * 60)
    
    # 1. Load Predictions
    data_path = root / "out" / "data" / "real" / "processed" / "predicted_states.parquet"
    if not data_path.exists():
        print(f"Predictions not found at {data_path}")
        print("Run apply_to_real.py first.")
        return
        
    print(f"Loading predictions from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # 2. Select a subset to visualize (e.g., first 5 days or 2000 bars)
    # Visualizing the whole year is too dense
    subset_size = 2000
    print(f"Visualizing first {subset_size} bars...")
    df_subset = df.iloc[:subset_size].copy()
    
    # 3. Plot
    plt.figure(figsize=(15, 8))
    
    # Create a color map for states
    states = df['predicted_state'].unique()
    palette = sns.color_palette("husl", len(states))
    color_map = dict(zip(states, palette))
    
    # Plot price line
    # We can't easily color a single line with multiple colors in matplotlib without segments
    # So we'll plot scatter points on top of a thin gray line
    
    plt.plot(df_subset.index, df_subset['close'], color='gray', alpha=0.5, linewidth=1, label='Price')
    
    for state in states:
        mask = df_subset['predicted_state'] == state
        plt.scatter(
            df_subset.index[mask], 
            df_subset['close'][mask], 
            c=[color_map[state]], 
            label=state,
            s=10,
            alpha=0.8
        )
        
    plt.title("Real Market Data - Predicted States (Baseline Model)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    # 4. Save
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "real_data_states.png"
    
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to: {output_path}")

if __name__ == "__main__":
    main()
