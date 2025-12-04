
import numpy as np
import pandas as pd
import tensorflow as tf
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime, timedelta

@dataclass
class TrajectoryPlan:
    """The 'Flight Plan' for the next hour."""
    displacement: float   # Net move (points)
    high_excursion: float # Points above start
    low_excursion: float  # Points below start
    total_distance: float # Total travel (volatility)
    close_location: float # 0.0 (Low) to 1.0 (High)

class FractalPlanner:
    """
    Mission Control.
    Uses a trained FractalNet (Multi-Scale CNN) to predict the trajectory
    of the next hour based on Micro (4h), Meso (24h), and Macro (5d) history.
    """
    
    def __init__(self, model_path: str = "out/fractal_net.keras"):
        self.model = None
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"FractalPlanner: Loaded model from {model_path}")
        except Exception as e:
            print(f"FractalPlanner: Failed to load model: {e}")
            
        # History Buffers (DataFrames with OHLCV)
        self.history_1m = pd.DataFrame()
        
        # Config
        self.micro_len = 240 # 4h
        self.meso_len = 96   # 24h (15m bars)
        self.macro_len = 120 # 5d (1h bars)
        
    def update_history(self, bar_1m: dict):
        """Add a new 1-minute bar to history."""
        # Convert dict to DataFrame row
        row = pd.DataFrame([bar_1m])
        # Map 'time' to 'timestamp' for internal consistency if needed, or just use 'time'
        # The model training used 'timestamp' as index?
        # Let's rename 'time' to 'timestamp' to match training data expectations if any.
        if 'time' in row.columns:
            row.rename(columns={'time': 'timestamp'}, inplace=True)
            
        row['timestamp'] = pd.to_datetime(row['timestamp'])
        row.set_index('timestamp', inplace=True)
        
        self.history_1m = pd.concat([self.history_1m, row])
        
        # Trim history to keep memory usage low (keep enough for macro)
        # We need 5 days + buffer. Say 7 days.
        cutoff = row.index[0] - timedelta(days=7)
        self.history_1m = self.history_1m[self.history_1m.index > cutoff]

    def warmup_history(self, df_history: pd.DataFrame):
        """
        Load historical data to warm up the model's buffers.
        Expects DataFrame with index 'timestamp' (or 'time') and columns: open, high, low, close, volume.
        """
        df = df_history.copy()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'time' in df.columns:
                df.set_index('time', inplace=True)
            elif 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            else:
                # Try to convert index
                df.index = pd.to_datetime(df.index)
        
        # Standardize columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                print(f"FractalPlanner: Warmup data missing column '{col}'")
                return
                
        self.history_1m = df[required_cols].sort_index()
        print(f"FractalPlanner: Warmed up with {len(self.history_1m)} bars.")
        
    def _normalize_sequence(self, df_seq, base_price, base_vol):
        """Normalize sequence for model input."""
        seq = np.zeros((len(df_seq), 5))
        seq[:, 0] = (df_seq['open'] / base_price) - 1.0
        seq[:, 1] = (df_seq['high'] / base_price) - 1.0
        seq[:, 2] = (df_seq['low'] / base_price) - 1.0
        seq[:, 3] = (df_seq['close'] / base_price) - 1.0
        seq[:, 4] = (df_seq['volume'] / (base_vol + 1e-9)) - 1.0
        return seq
        
    def _process_array(self, df, length, base_price, base_vol):
        """Process array to exact length."""
        arr = self._normalize_sequence(df, base_price, base_vol)
        if len(arr) < length:
            pad = np.zeros((length - len(arr), 5))
            arr = np.vstack([pad, arr])
        elif len(arr) > length:
            arr = arr[-length:]
        return arr

    def get_plan(self) -> Optional[TrajectoryPlan]:
        """
        Generate a plan for the next hour.
        Returns None if insufficient history.
        """
        if self.model is None:
            return None
            
        # Need at least some history
        if len(self.history_1m) < 240: # Need at least micro window
            return None
            
        # 1. Prepare Inputs
        # Resample on the fly
        df_15m = self.history_1m.resample('15min').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        ).dropna()
        
        df_1h = self.history_1m.resample('1h').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        ).dropna()
        
        # Base metrics
        base_price = self.history_1m['close'].iloc[-1]
        base_vol = self.history_1m['volume'].iloc[-240:].mean() # Micro avg
        
        # Extract windows
        x_mic = self._process_array(self.history_1m, self.micro_len, base_price, base_vol)
        x_mes = self._process_array(df_15m, self.meso_len, base_price, base_vol)
        x_mac = self._process_array(df_1h, self.macro_len, base_price, base_vol)
        
        # Batch dimension
        X_micro = np.expand_dims(x_mic, axis=0)
        X_meso = np.expand_dims(x_mes, axis=0)
        X_macro = np.expand_dims(x_mac, axis=0)
        
        # 2. Predict
        # Output: [Displacement, HighExc, LowExc, TotalDist, CloseLoc]
        preds = self.model.predict([X_micro, X_meso, X_macro], verbose=0)[0]
        
        return TrajectoryPlan(
            displacement=float(preds[0]),
            high_excursion=float(preds[1]),
            low_excursion=float(preds[2]),
            total_distance=float(preds[3]),
            close_location=float(preds[4])
        )
