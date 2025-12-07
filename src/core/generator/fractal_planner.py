
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import os

"""
FractalPlanner: PyTorch-only planner

This file was updated to remove TensorFlow/Keras fallbacks and operate
only with PyTorch. The project uses PyTorch on GPU for training and
inference. If a converted PyTorch state_dict exists at `out/fractal_net.pt`
it will be loaded; otherwise a lightweight TorchFractalNet will be
instantiated and used (random init).
"""

# Try to use PyTorch if available (we prefer PyTorch GPU)
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

try:
    from .fractal_planner_torch import build_torch_model_from_shapes
except Exception:
    build_torch_model_from_shapes = None

# TensorFlow is intentionally disabled in the PyTorch-first workflow
TF_AVAILABLE = False


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
        self.torch_model = None
        # Prefer PyTorch GPU model if available
        pt_path = model_path.replace('.keras', '.pt')
        if TORCH_AVAILABLE and os.path.exists(pt_path):
            try:
                # Load state_dict safely; prefer GPU later
                self.torch_state = torch.load(pt_path, map_location='cpu', weights_only=True)
                self.model_type = 'torch'
                print(f"FractalPlanner: Found PyTorch model at {pt_path}; will use PyTorch if CUDA available")
            except Exception as e:
                print(f"FractalPlanner: Failed to load PyTorch model: {e}")
                self.torch_state = None
        else:
            self.torch_state = None
            
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
        # Require minimum history for feature construction
        if len(self.history_1m) < self.micro_len:
            return None
        # If PyTorch is available, prefer a Torch model (either converted state or a new GPU-capable model)
        if TORCH_AVAILABLE:
            # Lazily build a Torch model instance if we have the builder helper
            if self.torch_model is None and build_torch_model_from_shapes is not None:
                try:
                    tm, _ = build_torch_model_from_shapes(self.micro_len, self.meso_len, self.macro_len)
                    self.torch_model = tm
                    # If a converted state exists, try to load it
                    pt_path = 'out/fractal_net.pt'
                    if os.path.exists(pt_path):
                        try:
                            state = torch.load(pt_path, map_location='cpu', weights_only=True)
                            self.torch_model.load_state_dict(state)
                            print(f"FractalPlanner: Loaded PyTorch state_dict from {pt_path}")
                        except Exception:
                            pass
                    # Move to GPU if available
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.torch_model.to(device)
                except Exception as e:
                    print('FractalPlanner: Failed to initialize Torch model:', e)

        # If we have a Torch model instance, use it for inference (prefer GPU)
        if self.torch_model is not None and TORCH_AVAILABLE:
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = self.torch_model
                model.to(device)

                base_price = self.history_1m['close'].iloc[-1]
                base_vol = self.history_1m['volume'].iloc[-240:].mean()

                x_mic = self._process_array(self.history_1m, self.micro_len, base_price, base_vol)
                df_15m = self.history_1m.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
                x_mes = self._process_array(df_15m, self.meso_len, base_price, base_vol)
                df_1h = self.history_1m.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
                x_mac = self._process_array(df_1h, self.macro_len, base_price, base_vol)

                # Use (batch, length, channels) input and let model handle transpose
                X_micro = torch.from_numpy(np.expand_dims(x_mic, axis=0).astype('float32')).to(device)
                X_meso = torch.from_numpy(np.expand_dims(x_mes, axis=0).astype('float32')).to(device)
                X_macro = torch.from_numpy(np.expand_dims(x_mac, axis=0).astype('float32')).to(device)

                model.eval()
                with torch.no_grad():
                    if device.type == 'cuda':
                        with torch.amp.autocast(device_type='cuda'):
                            preds = model(X_micro, X_meso, X_macro).float().cpu().numpy()[0]
                    else:
                        preds = model(X_micro, X_meso, X_macro).cpu().numpy()[0]

                return TrajectoryPlan(
                    displacement=float(preds[0]),
                    high_excursion=float(preds[1]),
                    low_excursion=float(preds[2]),
                    total_distance=float(preds[3]),
                    close_location=float(preds[4])
                )
            except Exception as e:
                print('FractalPlanner: Torch inference failed, falling back:', e)

        # If torch model state is present, prefer that path (and send to GPU if available)
        if self.torch_state is not None and TORCH_AVAILABLE:
            try:
                # Build Torch model architecture matching training script
                # Create Torch module on CPU and load state
                class _TmpNet(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        # minimal architecture matching training
                        self.m_conv1 = torch.nn.Conv1d(5, 32, kernel_size=3)
                        self.m_pool1 = torch.nn.MaxPool1d(2)
                        self.m_conv2 = torch.nn.Conv1d(32, 64, kernel_size=3)
                        self.m_pool2 = torch.nn.MaxPool1d(2)
                        self.s_conv1 = torch.nn.Conv1d(5, 32, kernel_size=3)
                        self.s_pool1 = torch.nn.MaxPool1d(2)
                        self.l_conv1 = torch.nn.Conv1d(5, 32, kernel_size=3)
                        self.l_pool1 = torch.nn.MaxPool1d(2)
                        # fusion dims will be inferred after a dummy forward; we create the FCs to match saved state
                        # create placeholders and then load state_dict will overwrite shapes if matching
                        self.fc1 = torch.nn.Linear(1, 128)
                        self.fc2 = torch.nn.Linear(128, 64)
                        self.out = torch.nn.Linear(64, 5)

                    def forward(self, xm, xs, xl):
                        x = torch.relu(self.m_conv1(xm))
                        x = self.m_pool1(x)
                        x = torch.relu(self.m_conv2(x))
                        x = self.m_pool2(x)
                        x = torch.flatten(x, 1)
                        y = torch.relu(self.s_conv1(xs))
                        y = self.s_pool1(y)
                        y = torch.flatten(y, 1)
                        z = torch.relu(self.l_conv1(xl))
                        z = self.l_pool1(z)
                        z = torch.flatten(z, 1)
                        merged = torch.cat([x, y, z], dim=1)
                        out = torch.relu(self.fc1(merged))
                        out = torch.nn.functional.dropout(out, p=0.3, training=self.training)
                        out = torch.relu(self.fc2(out))
                        out = self.out(out)
                        return out

                net = _TmpNet()
                # Attempt to load state dict; keys must match those saved by converter script
                try:
                    net.load_state_dict(self.torch_state)
                except Exception:
                    # ignore and proceed — converter may have different naming
                    pass

                # Move to GPU if available
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                net.to(device)

                # Prepare inputs like the TF path: arrays shaped (length,5) -> (1,5,length) for Conv1d
                base_price = self.history_1m['close'].iloc[-1]
                base_vol = self.history_1m['volume'].iloc[-240:].mean()

                x_mic = self._process_array(self.history_1m, self.micro_len, base_price, base_vol)
                x_mes = self._process_array(self.history_1m.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(), self.meso_len, base_price, base_vol)
                x_mac = self._process_array(self.history_1m.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(), self.macro_len, base_price, base_vol)

                X_micro = torch.from_numpy(np.expand_dims(np.transpose(x_mic, (1, 0)), axis=0).astype('float32')).to(device)
                X_meso = torch.from_numpy(np.expand_dims(np.transpose(x_mes, (1, 0)), axis=0).astype('float32')).to(device)
                X_macro = torch.from_numpy(np.expand_dims(np.transpose(x_mac, (1, 0)), axis=0).astype('float32')).to(device)

                net.eval()
                with torch.no_grad():
                    if device.type == 'cuda':
                        with torch.amp.autocast(device_type='cuda'):
                            preds = net(X_micro, X_meso, X_macro).float().cpu().numpy()[0]
                    else:
                        preds = net(X_micro, X_meso, X_macro).cpu().numpy()[0]

                return TrajectoryPlan(
                    displacement=float(preds[0]),
                    high_excursion=float(preds[1]),
                    low_excursion=float(preds[2]),
                    total_distance=float(preds[3]),
                    close_location=float(preds[4])
                )
            except Exception as e:
                print('FractalPlanner: Torch inference failed:', e)
                return None

        return None
