# Code Dump: 04_ml_features

## File: src/ml/features/builder.py
```python
"""
Feature Builder Module

Responsible for transforming raw generator output (DataFrames) into 
ML-ready feature matrices (X) and targets (y).
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

from src.core.detector.indicators import add_1m_indicators, IndicatorConfig

class FeatureBuilder:
    """
    Builds features from tick-based OHLCV data.
    """
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.indicator_config = IndicatorConfig()
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract technical and statistical features.
        
        Args:
            df: Raw DataFrame from PriceGenerator
            
        Returns:
            DataFrame with feature columns
        """
        # Enrich with indicators first
        df = add_1m_indicators(df, self.indicator_config)
        
        features = pd.DataFrame(index=df.index)
        
        # 1. Log Returns
        features['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        
        # 2. Volatility (Rolling Std Dev of Log Returns)
        features['volatility'] = features['log_return'].rolling(self.window_size).std().fillna(0)
        
        # 3. Tick Momentum (Rolling mean of delta ticks)
        features['tick_momentum'] = df['delta_ticks'].rolling(10).mean().fillna(0)
        
        # 4. Relative Range
        avg_range = df['range_ticks'].rolling(self.window_size).mean()
        features['relative_range'] = (df['range_ticks'] / avg_range.replace(0, 1)).fillna(1.0)
        
        # 5. Volume Intensity
        avg_vol = df['volume'].rolling(self.window_size).mean()
        features['volume_intensity'] = (df['volume'] / avg_vol.replace(0, 1)).fillna(1.0)
        
        # 6. Body Dominance
        features['body_dominance'] = (df['body_ticks'] / df['range_ticks'].replace(0, 1)).fillna(0)
        
        # 7. Wick Percentages
        features['upper_wick_pct'] = ((df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low']).replace(0, 1)).fillna(0)
        features['lower_wick_pct'] = ((df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low']).replace(0, 1)).fillna(0)
        
        # 8. Cumulative Delta
        features['cum_delta'] = df['delta_ticks'].cumsum()
        
        # 9. Distance from VWAP (using pre-calculated vwap from indicators)
        if 'vwap' in df.columns:
            features['dist_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        else:
            features['dist_vwap'] = 0.0
            
        # 10. Distance from EMAs
        if 'ema_fast' in df.columns:
            features['dist_ema_fast'] = (df['close'] - df['ema_fast']) / df['ema_fast']
        if 'ema_slow' in df.columns:
            features['dist_ema_slow'] = (df['close'] - df['ema_slow']) / df['ema_slow']
        
        # 11. RSI-like Tick Strength (14 period)
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

```

---

## File: src/ml/features/images.py
```python
"""
Image Feature Builder

Converts time-series data into image representations for CNN consumption.
"""

import pandas as pd
import numpy as np
import torch
from typing import Optional, Tuple

class ImageBuilder:
    """
    Builds 2D image representations of price action.
    """
    
    def __init__(self, height: int = 64, width: int = 64):
        self.height = height
        self.width = width
        
    def create_chart_image(self, df: pd.DataFrame, columns: List[str] = ['close']) -> torch.Tensor:
        """
        Create a simple line chart image tensor.
        
        Args:
            df: DataFrame with price data (must be length == width)
            columns: Columns to plot (each becomes a channel)
            
        Returns:
            Tensor of shape (C, H, W)
        """
        if len(df) != self.width:
            # Resample or slice to match width
            # For now, just take the last 'width' rows
            if len(df) > self.width:
                df = df.iloc[-self.width:]
            else:
                # Pad with zeros or first value?
                # Raise error for now
                raise ValueError(f"DataFrame length {len(df)} < width {self.width}")
                
        num_channels = len(columns)
        image = torch.zeros((num_channels, self.height, self.width))
        
        for c_idx, col in enumerate(columns):
            series = df[col].values
            
            # Normalize to 0-1 range within the window
            min_val = series.min()
            max_val = series.max()
            
            if max_val == min_val:
                normalized = np.zeros_like(series) + 0.5
            else:
                normalized = (series - min_val) / (max_val - min_val)
                
            # Map to height indices (0 is bottom, height-1 is top)
            # We want 0 at bottom, so index = floor(norm * (H-1))
            indices = np.floor(normalized * (self.height - 1)).astype(int)
            
            # Set pixels
            for w in range(self.width):
                h = indices[w]
                image[c_idx, h, w] = 1.0
                
        return image

```

---

