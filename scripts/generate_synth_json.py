"""
Generate a synthetic minute-level JSON matching the real data structure and time window.
Uses the project's PriceGenerator with session-aware volume.
Saves to data/synth_continuous.json.
"""

import os
import sys
import pandas as pd
import torch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.generator.engine import PriceGenerator, PhysicsConfig, MarketState

def load_real_data(path):
    """Load and inspect real data."""
    df = pd.read_json(path)
    df['time'] = pd.to_datetime(df['time'])
    return df.sort_values('time').reset_index(drop=True)

def generate_synth_json(real_df, out_path, vol_scale=0.8, volume_scale=10.0, seed=42):
    """
    Generate synthetic data using PriceGenerator (session-aware volume).
    
    Args:
        real_df: Real data DataFrame
        out_path: Output path for JSON
        vol_scale: Price volatility scale (default 0.8)
        volume_scale: Volume magnitude scale to match real data (default 10.0 to match ES ~800 vol)
        seed: Random seed
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating synthetic on {device}")
    
    # Extract timestamps and starting price from real
    timestamps = pd.to_datetime(real_df['time'].values)
    start_price = float(real_df['close'].iloc[0])
    
    # Create physics config with adjusted volatility
    physics = PhysicsConfig()
    physics.base_volatility *= vol_scale  # Scale the price volatility
    
    # Initialize generator
    gen = PriceGenerator(
        initial_price=start_price,
        physics_config=physics,
        seed=seed
    )
    
    # Generate using batch method (faster, on GPU)
    print(f"Generating {len(timestamps)} bars...")
    synth_df = gen.generate_batch(
        timestamps=timestamps,
        state=MarketState.RANGING
    )
    
    # **SCALE VOLUME** to match real data magnitude
    # The generator produces base volume ~100, we need to scale to match real data
    synth_df['volume'] = (synth_df['volume'] * volume_scale).astype(int)
    
    # Ensure time column is correct
    synth_df['time'] = pd.to_datetime(timestamps)
    synth_df['original_symbol'] = 'MESM5'  # Match real symbol
    
    # Reorder columns to match real structure
    synth_df = synth_df[['time', 'open', 'high', 'low', 'close', 'volume', 'original_symbol']]
    
    # Convert time to ISO format string (matching the real data format)
    synth_df['time'] = synth_df['time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Write JSON
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    synth_df.to_json(out_path, orient='records', indent=2)
    print(f"Saved {len(synth_df)} synthetic minutes to: {out_path}")
    
    # Print summary
    print(f"\nTime range: {synth_df['time'].min()} to {synth_df['time'].max()}")
    print(f"Price range: {synth_df['close'].min():.2f} to {synth_df['close'].max():.2f}")
    print(f"Volume stats: min={synth_df['volume'].min():.0f}, max={synth_df['volume'].max():.0f}, mean={synth_df['volume'].mean():.0f}")

if __name__ == '__main__':
    real = load_real_data('data/continuous_contract.json')
    print(f"Real data: {len(real)} rows, time range {real['time'].min()} to {real['time'].max()}")
    print(f"Real volume stats: min={real['volume'].min():.0f}, max={real['volume'].max():.0f}, mean={real['volume'].mean():.0f}")
    print()
    # Use volume_scale to match real data magnitude (~10x the base generator volume)
    generate_synth_json(real, 'data/synth_continuous.json', vol_scale=0.8, volume_scale=10.0)
