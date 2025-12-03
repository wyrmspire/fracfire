"""
Analyze Volatility

Calculates real-world volatility metrics (ATR, ADR) from the continuous contract data.
These metrics are used to bound the synthetic price generator.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.data.loader import RealDataLoader

def main():
    print("=" * 60)
    print("ANALYZING REAL VOLATILITY (ATR/ADR)")
    print("=" * 60)
    
    # 1. Load Real Data
    loader = RealDataLoader()
    data_path = root / "src" / "data" / "continuous_contract.json"
    df = loader.load_json(data_path)
    
    print(f"Loaded {len(df)} 1-minute bars.")
    
    # 2. Resample to Daily
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    daily_df = df.resample('D').agg(ohlc_dict).dropna()
    print(f"Resampled to {len(daily_df)} days.")
    
    # 3. Calculate Metrics
    
    # Daily Range (High - Low)
    daily_df['range'] = daily_df['high'] - daily_df['low']
    
    # True Range
    # TR = max(H-L, |H-PDC|, |L-PDC|)
    daily_df['prev_close'] = daily_df['close'].shift(1)
    daily_df['h_l'] = daily_df['high'] - daily_df['low']
    daily_df['h_pc'] = abs(daily_df['high'] - daily_df['prev_close'])
    daily_df['l_pc'] = abs(daily_df['low'] - daily_df['prev_close'])
    daily_df['tr'] = daily_df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    
    # ATR (14-day)
    daily_df['atr_14'] = daily_df['tr'].rolling(window=14).mean()
    
    # ADR (Average Daily Range)
    adr_mean = daily_df['range'].mean()
    adr_std = daily_df['range'].std()
    
    # Weekly Range
    weekly_df = df.resample('W').agg(ohlc_dict).dropna()
    weekly_df['range'] = weekly_df['high'] - weekly_df['low']
    awr_mean = weekly_df['range'].mean()
    
    # 4. Output Stats
    print("\n" + "-" * 40)
    print("REAL WORLD METRICS")
    print("-" * 40)
    print(f"Average Daily Range (ADR): {adr_mean:.2f} points")
    print(f"Daily Range Std Dev:     {adr_std:.2f} points")
    print(f"Max Daily Range:         {daily_df['range'].max():.2f} points")
    print(f"Average Weekly Range:    {awr_mean:.2f} points")
    print(f"Average ATR (14):        {daily_df['atr_14'].mean():.2f} points")
    print("-" * 40)
    
    # 5. Drift Analysis (Net Move per Week)
    weekly_df['net_move'] = weekly_df['close'] - weekly_df['open']
    avg_weekly_drift = abs(weekly_df['net_move']).mean()
    print(f"Avg Weekly Net Move:     {avg_weekly_drift:.2f} points")
    print("-" * 40)

if __name__ == "__main__":
    main()
