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

def load_data(json_path):
    print(f"Loading {json_path}...")
    df = pd.read_json(json_path)
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
    
    # Calculate ATR
    high = df_15m['high']
    low = df_15m['low']
    close = df_15m['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_15m['atr'] = tr.rolling(14).mean()
    
    return df_15m.dropna()

def create_dataset(df, lookback=64):
    """
    Create a dataset where the model scans EVERY bar to find profitable setups.
    Label 0: Neutral/Loss
    Label 1: Long Win (1.4R)
    Label 2: Short Win (1.4R)
    """
    print(f"Generating labels for {len(df)} bars...")
    
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    vols = df['volume'].values
    atrs = df['atr'].values
    
    features = []
    labels = []
    
    # Parameters
    risk_mult = 1.0
    reward_mult = 1.4
    horizon = 32 # Look ahead 8 hours (32 * 15m)
    
    # We need a sliding window of `lookback` bars for features
    # And `horizon` bars ahead for labels.
    
    start_idx = lookback
    end_idx = len(df) - horizon
    
    for i in range(start_idx, end_idx):
        # 1. Determine Label (Future Outcome)
        entry_price = opens[i+1] # Enter at Open of NEXT bar
        current_atr = atrs[i]
        
        risk = current_atr * risk_mult
        reward = risk * reward_mult
        
        # Long Targets
        l_target = entry_price + reward
        l_stop = entry_price - risk
        
        # Short Targets
        s_target = entry_price - reward
        s_stop = entry_price + risk
        
        # Check Future
        future_h = highs[i+1 : i+1+horizon]
        future_l = lows[i+1 : i+1+horizon]
        
        outcome = 0 # Default Neutral
        
        # Check Long
        long_win = False
        for k in range(len(future_h)):
            if future_l[k] <= l_stop:
                break # Stopped out
            if future_h[k] >= l_target:
                long_win = True
                break
                
        # Check Short
        short_win = False
        for k in range(len(future_h)):
            if future_h[k] >= s_stop:
                break # Stopped out
            if future_l[k] <= s_target:
                short_win = True
                break
        
        # Prioritize?
        # If both win (rare), maybe label as Long (1) or Short (2)?
        # Let's be strict: If Long wins, label 1. If Short wins, label 2.
        # If both, maybe 0 (chop)? Or just pick one.
        # Let's pick Long if both (arbitrary).
        
        if long_win:
            outcome = 1
        elif short_win:
            outcome = 2
        else:
            outcome = 0
            
        # 2. Extract Features (Past)
        # Window [i-lookback+1 : i+1]
        win_o = opens[i-lookback+1 : i+1]
        win_h = highs[i-lookback+1 : i+1]
        win_l = lows[i-lookback+1 : i+1]
        win_c = closes[i-lookback+1 : i+1]
        win_v = vols[i-lookback+1 : i+1]
        
        # Normalize
        prices = np.stack([win_o, win_h, win_l, win_c], axis=1)
        mean = prices.mean()
        std = prices.std() + 1e-6
        
        norm_o = (win_o - mean) / std
        norm_h = (win_h - mean) / std
        norm_l = (win_l - mean) / std
        norm_c = (win_c - mean) / std
        norm_v = (win_v - win_v.mean()) / (win_v.std() + 1e-6)
        
        feat = np.stack([norm_o, norm_h, norm_l, norm_c, norm_v], axis=1)
        
        features.append(feat)
        labels.append(outcome)
        
    return np.array(features), np.array(labels)

class SignatureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Deeper network to learn complex signatures
        self.features = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2), # 64 -> 32
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2), # 32 -> 16
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2), # 16 -> 8
            
            nn.Flatten()
        )
        
        flat_size = 128 * 8
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3) # 3 Classes: Neutral, Long, Short
        )
        
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def train_model(X, y, epochs=20):
    X_t = torch.tensor(X, dtype=torch.float32).transpose(1, 2).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)
    
    model = SignatureCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Handle Class Imbalance
    # Count classes
    counts = np.bincount(y)
    print(f"Class Counts: {counts}")
    # Weights: Inverse frequency
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"Training on {len(X)} samples...")
    batch_size = 64
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_t.size()[0])
        
        epoch_loss = 0
        
        for i in range(0, X_t.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_t[indices], y_t[indices]
            
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {epoch_loss / (len(X)/batch_size):.4f}")
            
    return model

def evaluate(model, X, y, df_source, start_idx_offset):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).transpose(1, 2).to(device)
    
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1) # (N, 3)
        
    # Check predictions
    # We want high confidence Long (Class 1) or Short (Class 2)
    threshold = 0.7 # High confidence
    
    long_probs = probs[:, 1].cpu().numpy()
    short_probs = probs[:, 2].cpu().numpy()
    
    long_signals = long_probs > threshold
    short_signals = short_probs > threshold
    
    print(f"--- Evaluation (Threshold {threshold}) ---")
    print(f"Long Signals: {long_signals.sum()}")
    print(f"Short Signals: {short_signals.sum()}")
    
    # Calculate PnL
    # We need to reconstruct the trades.
    # We have y (actual outcome), but that's binary (Win/Loss).
    # We need the actual R result.
    # For simplicity, let's assume fixed 1.4R win / -1.0R loss based on the label.
    # (Since we defined the label based on hitting target/stop).
    
    total_pnl = 0.0
    trades = 0
    wins = 0
    
    # Long Trades
    # If Signal Long (1) and Label is 1 -> Win (+1.4)
    # If Signal Long (1) and Label is NOT 1 -> Loss (-1.0)
    
    # Vectorized PnL
    # Longs
    long_wins = (y[long_signals] == 1)
    long_pnl = np.where(long_wins, 1.4, -1.0).sum()
    total_pnl += long_pnl
    trades += long_signals.sum()
    wins += long_wins.sum()
    
    # Shorts
    short_wins = (y[short_signals] == 2)
    short_pnl = np.where(short_wins, 1.4, -1.0).sum()
    total_pnl += short_pnl
    trades += short_signals.sum()
    wins += short_wins.sum()
    
    if trades > 0:
        print(f"Total Trades: {trades}")
        print(f"Win Rate: {wins/trades:.2%}")
        print(f"Total PnL: {total_pnl:.2f} R")
        print(f"EV: {total_pnl/trades:.2f} R")
    else:
        print("No trades taken.")

def main():
    # 1. Load Data
    df = load_data('src/data/continuous_contract.json')
    
    # 2. Create Dataset
    # This is expensive, so we do it once
    X, y = create_dataset(df, lookback=64)
    
    # 3. Split 50/50
    split_idx = len(X) // 2
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 4. Train
    print("\n--- Training Unconstrained Model ---")
    model = train_model(X_train, y_train)
    
    # 5. Test
    print("\n--- Testing on Unseen Data ---")
    evaluate(model, X_test, y_test, df, split_idx)

if __name__ == '__main__':
    main()
