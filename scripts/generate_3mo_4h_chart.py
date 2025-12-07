"""
Generate 3 months of synthetic 1-minute data on GPU,
aggregate to 4-hour candles, and visualize with Plotly.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(ROOT, os.pardir))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from src.core.generator.engine import PriceGenerator, PhysicsConfig, MarketState
from tqdm import tqdm

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed. Install with: pip install plotly")


def generate_3months_1m(
    start_date: datetime = None,
    num_days: int = 90,
    physics_config: PhysicsConfig = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate 3 months (~90 days) of 1-minute OHLCV data.
    Uses GPU-optimized PriceGenerator.
    
    Args:
        start_date: Starting date (defaults to Jan 1, 2024)
        num_days: Number of trading days to simulate (realistic: ~21-22 days/month)
        physics_config: PhysicsConfig for generator
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with 1-minute bars
    """
    if start_date is None:
        start_date = datetime(2024, 1, 1)
    
    if physics_config is None:
        physics_config = PhysicsConfig()
    
    gen = PriceGenerator(physics_config=physics_config, seed=seed)
    
    # Log device info
    if gen.device:
        print(f"[OK] Using device: {gen.device}")
    
    all_bars = []
    current_date = start_date
    
    print(f"Generating {num_days} days of 1-minute data (est. {num_days * 1440:,} bars)...")
    for day_idx in tqdm(range(num_days), desc="Days"):
        # Generate one day of bars (1440 bars = 24h * 60min)
        # For simplicity, generate full 24h (real markets would filter to RTH only)
        
        df_day = gen.generate_day(
            start_date=current_date.replace(hour=0, minute=0, second=0, microsecond=0),
            auto_transition=True
        )
        
        all_bars.append(df_day)
        current_date += timedelta(days=1)
    
    df_1m = pd.concat(all_bars, ignore_index=True)
    df_1m['time'] = pd.to_datetime(df_1m['time'])
    df_1m.set_index('time', inplace=True)
    df_1m.sort_index(inplace=True)
    
    print(f"Generated {len(df_1m):,} bars from {df_1m.index[0]} to {df_1m.index[-1]}")
    
    return df_1m


def aggregate_to_4h(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-minute OHLCV data to 4-hour candles.
    
    Args:
        df_1m: DataFrame with 1-minute bars (indexed by time)
        
    Returns:
        DataFrame with 4-hour bars
    """
    print("Aggregating to 4-hour timeframe...")
    
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'state': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],  # Most common state
    }
    
    df_4h = df_1m.resample('4h').agg(agg_dict).dropna()
    
    # Ensure valid OHLC (high >= low, etc.)
    df_4h['high'] = df_4h[['open', 'high', 'close']].max(axis=1)
    df_4h['low'] = df_4h[['open', 'low', 'close']].min(axis=1)
    
    print(f"Aggregated to {len(df_4h)} 4-hour candles")
    
    return df_4h


def plot_candlestick(df_4h: pd.DataFrame, title: str = "MES 4H") -> None:
    """
    Create and display an interactive Plotly candlestick chart.
    
    Args:
        df_4h: DataFrame with 4-hour OHLCV
        title: Chart title
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping visualization.")
        return
    
    print("Creating candlestick chart...")
    
    # Reset index to have 'time' as a column
    df_plot = df_4h.reset_index()
    
    # Determine colors: green for up, red for down
    colors = ['green' if close >= open_ else 'red' 
              for close, open_ in zip(df_plot['close'], df_plot['open'])]
    
    fig = go.Figure(data=[go.Candlestick(
        x=df_plot['time'],
        open=df_plot['open'],
        high=df_plot['high'],
        low=df_plot['low'],
        close=df_plot['close'],
        name='Price',
    )])
    
    # Add volume as a secondary trace (light bars)
    fig.add_trace(go.Bar(
        x=df_plot['time'],
        y=df_plot['volume'],
        name='Volume',
        yaxis='y2',
        marker=dict(color=['rgba(0,255,0,0.3)' if c == 'green' else 'rgba(255,0,0,0.3)' 
                           for c in colors]),
        showlegend=True,
    ))
    
    # Update layout with volume axis
    fig.update_layout(
        title=title,
        yaxis_title='Price (Points)',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
        ),
        xaxis_title='Date',
        height=700,
        hovermode='x unified',
        template='plotly_dark',
    )
    
    # Save to HTML
    output_file = 'out/synthetic_4h_chart.html'
    os.makedirs('out', exist_ok=True)
    fig.write_html(output_file)
    print(f"[OK] Chart saved to: {output_file}")
    
    # Also show in browser if in interactive mode
    fig.show()


def main():
    print("=" * 60)
    print("Synthetic MES 3-Month 4H Chart Generator")
    print("=" * 60)
    
    # Generate 3 months of 1-minute data
    df_1m = generate_3months_1m(
        start_date=datetime(2024, 1, 1),
        num_days=90,
        seed=42
    )
    
    # Aggregate to 4-hour
    df_4h = aggregate_to_4h(df_1m)
    
    # Display summary stats
    print("\n4H Candle Statistics:")
    print(f"  Open:   {df_4h['open'].iloc[0]:.2f} -> {df_4h['open'].iloc[-1]:.2f}")
    print(f"  Range:  {df_4h['low'].min():.2f} to {df_4h['high'].max():.2f}")
    print(f"  Avg Vol: {df_4h['volume'].mean():.0f}")
    
    # Plot
    plot_candlestick(df_4h)
    
    # Save to CSV for reference
    csv_file = 'out/synthetic_4h_data.csv'
    df_4h.to_csv(csv_file)
    print(f"[OK] Data saved to: {csv_file}")


if __name__ == "__main__":
    main()
