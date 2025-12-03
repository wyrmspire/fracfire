"""
Real Data Loader

Loads and preprocesses real market data (JSON format) for use with the ML pipeline.
Handles conversion to tick-based metrics.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union

class RealDataLoader:
    """
    Loads real market data and aligns it with synthetic data schema.
    """
    
    def __init__(self, tick_size: float = 0.25):
        self.tick_size = tick_size
        
    def load_json(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load JSON data and preprocess it.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            DataFrame with required columns for FeatureBuilder
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        print(f"Loading real data from {file_path}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        df = pd.DataFrame(data)
        
        # Parse time
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        
        # Ensure numeric columns
        cols = ['open', 'high', 'low', 'close', 'volume']
        for col in cols:
            df[col] = pd.to_numeric(df[col])
            
        # Calculate tick-based metrics
        # Note: We round to nearest int to avoid floating point issues
        df['delta_ticks'] = ((df['close'] - df['open']) / self.tick_size).round().astype(int)
        df['range_ticks'] = ((df['high'] - df['low']) / self.tick_size).round().astype(int)
        df['body_ticks'] = (abs(df['close'] - df['open']) / self.tick_size).round().astype(int)
        
        # Calculate wick ticks (useful for debugging/validation even if not directly used by FeatureBuilder currently)
        df['upper_wick_ticks'] = ((df['high'] - df[['open', 'close']].max(axis=1)) / self.tick_size).round().astype(int)
        df['lower_wick_ticks'] = ((df[['open', 'close']].min(axis=1) - df['low']) / self.tick_size).round().astype(int)
        
        # Add synthetic-compatible columns if missing
        # Synthetic data has 'state' but real data doesn't (that's what we want to predict)
        
        print(f"Loaded {len(df)} bars of real data.")
        return df

if __name__ == "__main__":
    # Test run
    root = Path(__file__).resolve().parents[2]
    path = root / "src" / "data" / "continuous_contract.json"
    loader = RealDataLoader()
    df = loader.load_json(path)
    print(df.head())
    print(df.describe())
