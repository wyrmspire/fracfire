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
