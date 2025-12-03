"""
Feature Builder Module

Responsible for transforming raw generator output (DataFrames) into 
ML-ready feature matrices (X) and targets (y).
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

class FeatureBuilder:
    """
    Builds features from tick-based OHLCV data.
    """
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract technical and statistical features.
        
        Args:
            df: Raw DataFrame from PriceGenerator
            
        Returns:
            DataFrame with feature columns
        """
        features = pd.DataFrame(index=df.index)
        
        # 1. Log Returns
        # Use close prices. Fillna(0) for the first bar.
        features['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        
        # 2. Volatility (Rolling Std Dev of Log Returns)
        features['volatility'] = features['log_return'].rolling(self.window_size).std().fillna(0)
        
        # 3. Tick Momentum (Rolling mean of delta ticks)
        # Captures short-term directional pressure
        features['tick_momentum'] = df['delta_ticks'].rolling(10).mean().fillna(0)
        
        # 4. Relative Range
        # Current range vs average range (volatility expansion/contraction)
        avg_range = df['range_ticks'].rolling(self.window_size).mean()
        features['relative_range'] = (df['range_ticks'] / avg_range.replace(0, 1)).fillna(1.0)
        
        # 5. Volume Intensity
        # Current volume vs average volume
        avg_vol = df['volume'].rolling(self.window_size).mean()
        features['volume_intensity'] = (df['volume'] / avg_vol.replace(0, 1)).fillna(1.0)
        
        # 6. Body Dominance
        # How much of the range is the body? (Trend vs Indecision)
        features['body_dominance'] = (df['body_ticks'] / df['range_ticks'].replace(0, 1)).fillna(0)
        
        # 7. Wick Percentages
        # Upper wick % (Selling pressure)
        features['upper_wick_pct'] = ((df['high'] - df[['open', 'close']].max(axis=1)) / df['range_ticks'].replace(0, 1) * 4).fillna(0) # *4 because ticks are 0.25? No, range_ticks is integer.
        # Wait, range_ticks is int. (high-max)/range is correct ratio.
        features['upper_wick_pct'] = ((df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low']).replace(0, 1)).fillna(0)
        
        # Lower wick % (Buying pressure)
        features['lower_wick_pct'] = ((df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low']).replace(0, 1)).fillna(0)
        
        # 8. Cumulative Delta (Intraday Trend)
        # Reset per day usually, but here we just take cumsum of the DF provided
        features['cum_delta'] = df['delta_ticks'].cumsum()
        
        # 9. Distance from VWAP (Approximate)
        # VWAP = Sum(Price * Vol) / Sum(Vol)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cum_vol_price = (typical_price * df['volume']).cumsum()
        cum_vol = df['volume'].cumsum()
        vwap = cum_vol_price / cum_vol.replace(0, 1)
        features['dist_vwap'] = (df['close'] - vwap) / vwap
        
        # 10. RSI-like Tick Strength (14 period)
        # Simple implementation using delta ticks
        delta = df['delta_ticks']
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        features['tick_rsi'] = 100 - (100 / (1 + rs)).fillna(50)
        
        # Clean up NaNs/Infs
        features = features.replace([np.inf, -np.inf], 0).fillna(0)
        
        return features
    
    def create_dataset(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'state',
        lookback: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create (X, y) dataset for training.
        
        Args:
            df: Raw DataFrame
            target_col: Column to use as target
            lookback: Sequence length for RNN/Transformer (NOT IMPLEMENTED YET)
            
        Returns:
            X: Feature matrix
            y: Target vector
        """
        # Extract features
        features = self.extract_features(df)
        
        # Get target
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in DataFrame")
            
        y = df[target_col].values
        X = features.values
        
        if lookback > 0:
            raise NotImplementedError("Sequence lookback not yet implemented")
            
        return X, y
