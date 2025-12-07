import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_and_resample(csv_path):
    print(f"Loading {csv_path}...")
    if csv_path.endswith('.json'):
        df = pd.read_json(csv_path)
    else:
        df = pd.read_csv(csv_path)
        
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Resample to 15min
    ohlc = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    df_15m = df.resample('15min').agg(ohlc).dropna()
    
    # Calculate ATR for dynamic 1:1 targets
    high = df_15m['high']
    low = df_15m['low']
    close = df_15m['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_15m['atr'] = tr.rolling(14).mean()
    
    return df_15m.dropna()

def prepare_midnight_dataset(df_15m, lookback=20):
    """
    Extract features (00:00-05:00) and labels (Outcome of 1:1 trade at 05:00).
    """
    # Filter for the "Trigger Bar" -> 05:00
    # We want to make a decision at 05:00 close.
    # Features: 20 bars ending at 05:00 (inclusive? or 04:45?).
    # User said "midnight to 5am". 00:00 to 05:00 is 5 hours = 20 bars.
    # So bars are [00:00, 00:15, ..., 04:45].
    # At 05:00 (the time of the *next* bar open, or the timestamp of the 04:45 bar close?), we trade.
    # Let's assume we trade at the Open of the 05:00 bar (which is the Close of 04:45 bar).
    # So we identify the 04:45 bar.
    
    # Filter indices where time is 04:45
    # Pandas 15m timestamps usually label the left edge.
    # 00:00 bar covers 00:00-00:15.
    # 04:45 bar covers 04:45-05:00.
    # So at 05:00, we have the 04:45 bar completed.
    
    times = df_15m.index
    is_trigger = (times.hour == 4) & (times.minute == 45)
    
    indices = np.where(is_trigger)[0]
    print(f"Found {len(indices)} midnight sessions.")
    
    features = []
    labels = []
    returns = []
    
    # Convert to numpy for speed
    opens = df_15m['open'].values
    highs = df_15m['high'].values
    lows = df_15m['low'].values
    closes = df_15m['close'].values
    vols = df_15m['volume'].values
    atrs = df_15m['atr'].values
    
    data_len = len(df_15m)
    
    for idx in indices:
        # Check if we have enough history (20 bars)
        if idx < lookback - 1:
            continue
            
        # Extract window: [idx - 19 : idx + 1] (20 bars)
        start_idx = idx - lookback + 1
        end_idx = idx + 1
        
        # Features
        # Shape (20, 5)
        win_o = opens[start_idx:end_idx]
        win_h = highs[start_idx:end_idx]
        win_l = lows[start_idx:end_idx]
        win_c = closes[start_idx:end_idx]
        win_v = vols[start_idx:end_idx]
        
        # Normalize
        # (Price - Mean) / Std
        prices = np.stack([win_o, win_h, win_l, win_c], axis=1)
        mean = prices.mean()
        std = prices.std() + 1e-6
        
        norm_o = (win_o - mean) / std
        norm_h = (win_h - mean) / std
        norm_l = (win_l - mean) / std
        norm_c = (win_c - mean) / std
        norm_v = (win_v - win_v.mean()) / (win_v.std() + 1e-6)
        
        feat = np.stack([norm_o, norm_h, norm_l, norm_c, norm_v], axis=1) # (20, 5)
        features.append(feat)
        
        # Labeling
        # Trade at Open of next bar (idx + 1)
        if idx + 1 >= data_len:
            continue
            
        entry_price = opens[idx + 1] # Open of 05:00 bar

        # Calculate Session Range (00:00 - 05:00)
        # We have the window data in `features` loop, but we need the raw prices to get range.
        # Window is [idx - 19 : idx + 1] -> 20 bars.
        # Highs/Lows for this window:
        start_idx = idx - lookback + 1
        end_idx = idx + 1
        
        session_high = highs[start_idx:end_idx].max()
        session_low = lows[start_idx:end_idx].min()
        session_range = session_high - session_low
        
        # Time-based exit at 10:30
        # Entry is at 05:00 (idx + 1)
        # 10:30 is 5.5 hours later -> 22 bars
        exit_idx = idx + 1 + 22
        
        if exit_idx >= data_len:
            continue
            
        exit_price = opens[exit_idx] # Open of 10:30 bar (or Close of 10:15)
        # Actually, let's use Close of 10:15 bar (which is index idx + 1 + 21) or Open of 10:30 bar.
        # Let's use Open of 10:30 bar to be precise about "until 10:30".
        
        price_diff = exit_price - entry_price
        
        # Normalize PnL by "Risk" unit (0.66 * Range) for comparison
        risk_unit = session_range * 0.66
        if risk_unit == 0: risk_unit = 1.0
        
        r_outcome = price_diff / risk_unit
        
        # Label: 1 if Long wins (Price went up), 0 if Short wins (Price went down)
        if r_outcome > 0:
            labels.append(1)
            returns.append(r_outcome)
        elif r_outcome < 0:
            labels.append(0)
            returns.append(r_outcome)
        else:
            features.pop()
            continue # Flat
            
        # We need to store the return for this sample to calculate PnL later.
        # Hack: Append it to features? No, features are normalized.
        # Let's change the function signature to return (X, y, returns)
        # But I need to update the caller.
        
        # Actually, let's just stick to classification accuracy for training,
        # and for evaluation, we might need to re-calculate or just assume average?
        # No, "see if it is profitable" requires actual PnL.
        # I will modify the function to return `returns` as well.
        
    return np.array(features), np.array(labels), np.array(returns)

class MidnightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        # 20 -> 10 -> 5
        flat_size = 64 * 5
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def train_model(X, y, epochs=30):
    X_t = torch.tensor(X, dtype=torch.float32).transpose(1, 2).to(device) # (N, 5, 20)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
    print(f"y_t shape: {y_t.shape}")
    
    model = MidnightCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Training on {len(X)} samples...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            acc = ((torch.sigmoid(out) > 0.5) == (y_t > 0.5)).float().mean()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Acc: {acc:.2%}")
            
    return model

def evaluate(model, X, y, returns):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).transpose(1, 2).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    returns_t = torch.tensor(returns, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
    acc = (preds == y_t).float().mean()
    print(f"Test Accuracy: {acc:.2%}")
    
    # PnL Simulation
    # If pred 1 (Long) -> PnL = returns
    # If pred 0 (Short) -> PnL = -returns
    
    trade_dir = torch.where(probs > 0.5, 1.0, -1.0)
    
    # returns contains the "R outcome" for a Long trade.
    # So if Long (1), PnL = returns.
    # If Short (-1), PnL = -returns.
    
    pnl = trade_dir * returns_t
    
    total_pnl = pnl.sum().item()
    
    print(f"Total PnL: {total_pnl:.2f} R over {len(y)} trades")
    print(f"EV: {total_pnl/len(y):.2f} R")

def main():
    # 1. Load Data
    df_y1 = load_and_resample('out/synthetic_year.csv')
    df_y2 = load_and_resample('out/synthetic_year_2.csv')
    df_real = load_and_resample('src/data/continuous_contract.json')
    
    # 2. Prepare
    print("\n--- Preparing Year 1 (Synth) ---")
    X1, y1, r1 = prepare_midnight_dataset(df_y1)
    print(f"Samples: {len(X1)}")
    
    print("\n--- Preparing Year 2 (Synth) ---")
    X2, y2, r2 = prepare_midnight_dataset(df_y2)
    print(f"Samples: {len(X2)}")
    
    print("\n--- Preparing Real Data ---")
    X_real, y_real, r_real = prepare_midnight_dataset(df_real)
    print(f"Samples: {len(X_real)}")
    
    # 3. Train on Year 1
    print("\n--- Training on Year 1 Synth ---")
    model = train_model(X1, y1)
    
    # 4. Test on Year 2
    print("\n--- Testing on Year 2 Synth ---")
    evaluate(model, X2, y2, r2)
    
    # 5. Train on All Synth (Year 1 + Year 2)
    print("\n--- Training on All Synth (Y1 + Y2) ---")
    X_synth = np.concatenate([X1, X2])
    y_synth = np.concatenate([y1, y2])
    model_synth = train_model(X_synth, y_synth)
    
    # 6. Test on Real Data
    print("\n--- Testing Synth-Trained Model on REAL DATA ---")
    evaluate(model_synth, X_real, y_real, r_real)

if __name__ == '__main__':
    main()
