"""
Generate synthetic data at different scales:
- Quick test: 10 days -> ~2 four-hour candles per day
- Full run: 90 days -> ~540 four-hour candles
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


def generate_synthetic_1m(
    start_date: datetime = None,
    num_days: int = 10,
    physics_config: PhysicsConfig = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate N days of 1-minute OHLCV data on GPU.
    """
    if start_date is None:
        start_date = datetime(2024, 1, 1)
    
    if physics_config is None:
        physics_config = PhysicsConfig()
    
    gen = PriceGenerator(physics_config=physics_config, seed=seed)
    
    if gen.device:
        print(f"[Device] {gen.device}")
    
    all_bars = []
    current_date = start_date
    
    print(f"Generating {num_days} days (~{num_days * 1440:,} bars)...")
    for day_idx in tqdm(range(num_days), desc="Days"):
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
    
    print(f"[OK] Generated {len(df_1m):,} bars ({df_1m.index[0]} to {df_1m.index[-1]})")
    
    return df_1m


def aggregate_to_4h(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute to 4-hour OHLCV."""
    print("Aggregating to 4H...")
    
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'state': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
    }
    
    df_4h = df_1m.resample('4h').agg(agg_dict).dropna()
    
    # Ensure valid OHLC
    df_4h['high'] = df_4h[['open', 'high', 'close']].max(axis=1)
    df_4h['low'] = df_4h[['open', 'low', 'close']].min(axis=1)
    
    print(f"[OK] Aggregated to {len(df_4h)} 4H candles")
    
    return df_4h


def plot_candlestick(df_4h: pd.DataFrame, title: str = "MES 4H Synthetic") -> None:
    """Create and save interactive Plotly candlestick chart."""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping visualization.")
        return
    
    print("Creating Plotly chart...")
    
    df_plot = df_4h.reset_index()
    
    # Create candlestick
    fig = go.Figure(data=[go.Candlestick(
        x=df_plot['time'],
        open=df_plot['open'],
        high=df_plot['high'],
        low=df_plot['low'],
        close=df_plot['close'],
        name='Price',
    )])
    
    # Add volume bars
    colors = ['rgba(0,200,0,0.4)' if close >= open_ else 'rgba(200,0,0,0.4)' 
              for close, open_ in zip(df_plot['close'], df_plot['open'])]
    
    fig.add_trace(go.Bar(
        x=df_plot['time'],
        y=df_plot['volume'],
        name='Volume',
        yaxis='y2',
        marker=dict(color=colors),
    ))
    
    # Layout
    fig.update_layout(
        title=title,
        yaxis_title='Price (Points)',
        yaxis2=dict(title='Volume', overlaying='y', side='right'),
        xaxis_title='Date',
        height=700,
        hovermode='x unified',
        template='plotly_dark',
    )
    
    os.makedirs('out', exist_ok=True)
    output_file = 'out/synthetic_4h_chart.html'
    fig.write_html(output_file)
    print(f"[OK] Chart saved: {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic price data and plot 4H chart')
    parser.add_argument('--days', type=int, default=10, help='Number of days to generate (default: 10 for quick test)')
    parser.add_argument('--start', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Synthetic Price Data Generator (GPU-Accelerated)")
    print("=" * 70)
    
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    
    # Generate 1-minute data
    df_1m = generate_synthetic_1m(
        start_date=start_date,
        num_days=args.days,
        seed=args.seed
    )
    
    # Aggregate to 4-hour
    df_4h = aggregate_to_4h(df_1m)
    
    # Print stats
    print("\n4H Statistics:")
    print(f"  Date range:  {df_4h.index[0]} to {df_4h.index[-1]}")
    print(f"  Price range: {df_4h['low'].min():.2f} to {df_4h['high'].max():.2f}")
    print(f"  Candles:     {len(df_4h)}")
    print(f"  Avg volume:  {df_4h['volume'].mean():.0f}")
    
    # Create visualization
    plot_candlestick(df_4h)
    
    # Save CSV
    csv_file = 'out/synthetic_4h_data.csv'
    df_4h.to_csv(csv_file)
    print(f"[OK] CSV saved: {csv_file}")
    
    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
