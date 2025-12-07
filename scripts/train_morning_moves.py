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

def load_data_to_gpu(csv_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    
    # Extract time components for filtering
    # We assume data is continuous minute data
    # We need to filter for Chicago 08:45 - 10:00
    # Let's add hour/minute columns
    hours = df['time'].dt.hour.values
    minutes = df['time'].dt.minute.values
    
    # Convert OHLCV to tensor
    data = df[['open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
    
    return torch.tensor(data, device=device), torch.tensor(hours, device=device), torch.tensor(minutes, device=device)

def find_morning_opportunities(ohlcv, hours, minutes, lookahead=120, rr_target=3.0):
    """
    Identify bars in the morning session (08:45 - 10:00) where a trade would have yielded 3:1.
    """
    # Filter: (Hour == 8 AND Minute >= 45) OR (Hour == 9) OR (Hour == 10 AND Minute == 0)
    # Chicago Time
    is_8 = hours == 8
    is_9 = hours == 9
    is_10 = hours == 10
    
    in_window = (is_8 & (minutes >= 45)) | (is_9) | (is_10 & (minutes == 0))
    
    indices = torch.nonzero(in_window).squeeze()
    print(f"Found {len(indices)} bars in morning window (08:45-10:00).")
    
    if len(indices) == 0:
        return indices, torch.tensor([], device=device)
        
    # Define Trade Parameters for Labeling
    # Entry: Close of the bar
    # Stop: Lowest Low of last 5 bars (for Long)
    # Target: 3R
    
    # We need to vectorize "Lowest Low of last 5 bars"
    # Unfold?
    # ohlcv: (N, 5) -> Low is col 2
    lows = ohlcv[:, 2]
    highs = ohlcv[:, 1]
    closes = ohlcv[:, 3]
    
    # Create windows of 5 previous lows
    # We can use unfold: dimension 0, size 5, step 1
    # But unfold is tricky with indices.
    # Let's just use a loop or simple shifting for 5 bars since it's small
    # L[i], L[i-1], L[i-2], L[i-3], L[i-4]
    
    l0 = lows
    l1 = torch.roll(lows, 1)
    l2 = torch.roll(lows, 2)
    l3 = torch.roll(lows, 3)
    l4 = torch.roll(lows, 4)
    
    recent_lows = torch.stack([l0, l1, l2, l3, l4], dim=1)
    stop_lows = recent_lows.min(dim=1)[0]
    
    # For Shorts: Highest High
    h0 = highs
    h1 = torch.roll(highs, 1)
    h2 = torch.roll(highs, 2)
    h3 = torch.roll(highs, 3)
    h4 = torch.roll(highs, 4)
    
    recent_highs = torch.stack([h0, h1, h2, h3, h4], dim=1)
    stop_highs = recent_highs.max(dim=1)[0]
    
    # Filter indices to ensure we have history
    indices = indices[indices > 5]
    
    # Calculate Outcomes for Longs
    # Entry = Close[i]
    # Stop = Stop_Lows[i] - 1 tick (approx 0.25)
    # Risk = Entry - Stop
    
    entries = closes[indices]
    stops = stop_lows[indices] - 0.25
    risks = entries - stops
    
    # Filter valid risks (> 0)
    valid_mask = risks > 1.0 # Minimum risk to avoid noise
    indices = indices[valid_mask]
    entries = entries[valid_mask]
    stops = stops[valid_mask]
    risks = risks[valid_mask]
    
    targets = entries + (risks * rr_target)
    
    # Check future outcomes (Lookahead)
    # Similar to previous script
    num_samples = len(indices)
    offsets = torch.arange(1, lookahead + 1, device=device).unsqueeze(0).expand(num_samples, lookahead)
    base_indices = indices.unsqueeze(1).expand(num_samples, lookahead)
    lookahead_indices = (base_indices + offsets).clamp(max=len(ohlcv)-1)
    
    future_highs = highs[lookahead_indices]
    future_lows = lows[lookahead_indices]
    
    target_matrix = targets.unsqueeze(1).expand(num_samples, lookahead)
    stop_matrix = stops.unsqueeze(1).expand(num_samples, lookahead)
    
    hit_target = (future_highs >= target_matrix)
    hit_stop = (future_lows <= stop_matrix)
    
    # Determine first hit
    # We use a large number for "never"
    target_times = torch.where(hit_target.any(dim=1), hit_target.float().argmax(dim=1), torch.tensor(lookahead+1, device=device))
    stop_times = torch.where(hit_stop.any(dim=1), hit_stop.float().argmax(dim=1), torch.tensor(lookahead+1, device=device))
    
    # Success: Target before Stop AND Target within window
    success = (target_times < stop_times) & (target_times < lookahead)
    
    labels = success.float()
    print(f"Morning Opportunities (Longs): {labels.sum().item()} / {len(labels)} ({labels.mean().item():.2%})")
    
    return indices, labels

def extract_features(ohlcv, indices, window_size=60):
    """
    Extract previous 60 bars (1 hour) of price action.
    Normalize: (Price - Mean) / Std
    """
    num_samples = len(indices)
    offsets = torch.arange(-window_size, 0, device=device).unsqueeze(0).expand(num_samples, window_size)
    base_indices = indices.unsqueeze(1).expand(num_samples, window_size)
    window_indices = (base_indices + offsets).clamp(min=0)
    
    flat_indices = window_indices.flatten()
    windows = ohlcv[flat_indices].view(num_samples, window_size, 5)
    
    means = windows.mean(dim=1, keepdim=True)
    stds = windows.std(dim=1, keepdim=True) + 1e-6
    
    norm_windows = (windows - means) / stds
    return norm_windows.transpose(1, 2) # (N, 5, 60)

class MorningCNN(nn.Module):
    def __init__(self, window_size=60):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        
        # 60 -> 30 -> 15 -> 7
        flat_size = 128 * (window_size // 8)
        
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def train_model(X, y, epochs=20):
    model = MorningCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_pos = y.sum()
    num_neg = len(y) - num_pos
    pos_weight = num_neg / (num_pos + 1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    batch_size = 1024
    num_samples = len(X)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Training on {num_samples} samples...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        perm = torch.randperm(num_samples, device=device)
        X = X[perm]
        y = y[perm]
        
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, num_samples)
            X_batch = X[start:end]
            y_batch = y[start:end].unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.4f}")
            
    return model

def backtest(model, ohlcv, indices, labels, threshold=0.7):
    model.eval()
    X = extract_features(ohlcv, indices)
    
    with torch.no_grad():
        logits = model(X).squeeze()
        probs = torch.sigmoid(logits)
        
    trades_mask = probs > threshold
    num_trades = trades_mask.sum().item()
    
    if num_trades == 0:
        print(f"Threshold {threshold}: No trades.")
        return
        
    actual_outcomes = labels[trades_mask]
    wins = actual_outcomes.sum().item()
    losses = num_trades - wins
    win_rate = wins / num_trades
    pnl = (wins * 3.0) - (losses * 1.0)
    
    print(f"--- Threshold {threshold} ---")
    print(f"Trades: {num_trades} ({num_trades/len(indices)*100:.1f}%)")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"PnL: {pnl:.2f} R")
    print(f"EV: {pnl/num_trades:.2f} R")
    
    if threshold == 0.7:
        # Plot PnL
        pnl_series = torch.where(actual_outcomes == 1, 3.0, -1.0).cumsum(dim=0).cpu().numpy()
        plt.figure(figsize=(10, 6))
        plt.plot(pnl_series)
        plt.title("Cumulative PnL (Morning Moves Strategy)")
        plt.xlabel("Trade #")
        plt.ylabel("R")
        plt.grid(True)
        plt.savefig("out/charts/morning_pnl.png")
        print("Saved out/charts/morning_pnl.png")

def main():
    # 1. Load Data
    data_y1, h1, m1 = load_data_to_gpu('out/synthetic_year.csv')
    data_y2, h2, m2 = load_data_to_gpu('out/synthetic_year_2.csv')
    
    # 2. Label Training Data (Year 1)
    print("\n--- Year 1: Finding Opportunities ---")
    indices_y1, labels_y1 = find_morning_opportunities(data_y1, h1, m1)
    X_y1 = extract_features(data_y1, indices_y1)
    
    # 3. Train
    model = train_model(X_y1, labels_y1, epochs=20)
    torch.save(model.state_dict(), 'out/morning_cnn.pt')
    
    # 4. Backtest (Year 2)
    print("\n--- Year 2: Backtesting ---")
    # We use the same logic to find *candidates* (every morning bar)
    # But we don't cheat by peeking at the label. We just use the indices.
    indices_y2, labels_y2 = find_morning_opportunities(data_y2, h2, m2)
    
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        backtest(model, data_y2, indices_y2, labels_y2, threshold=t)

if __name__ == '__main__':
    main()
