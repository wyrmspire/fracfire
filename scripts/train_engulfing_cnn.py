import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_data_to_gpu(csv_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    # Convert to tensor
    # We need Open, High, Low, Close, Volume
    # Shape: (N, 5)
    data = df[['open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
    return torch.tensor(data, device=device)

def find_engulfing_setups(ohlcv, lookahead=60, rr_target=3.0):
    """
    Vectorized detection of Bullish Engulfing patterns and their outcomes.
    ohlcv: (N, 5) tensor [O, H, L, C, V]
    """
    # Unpack columns for readability
    op = ohlcv[:, 0]
    hi = ohlcv[:, 1]
    lo = ohlcv[:, 2]
    cl = ohlcv[:, 3]
    
    # Shifted (Previous bar)
    prev_op = torch.roll(op, 1)
    prev_cl = torch.roll(cl, 1)
    prev_hi = torch.roll(hi, 1)
    prev_lo = torch.roll(lo, 1)
    
    # 1. Detect Bullish Engulfing
    # Prev bar is Red: prev_cl < prev_op
    # Curr bar is Green: cl > op
    # Engulfing body: op <= prev_cl AND cl >= prev_op
    # (Strict definition: body engulfs body)
    
    is_red_prev = prev_cl < prev_op
    is_green_curr = cl > op
    engulfs = (op <= prev_cl) & (cl >= prev_op)
    
    # Combine conditions
    # Also ignore first bar (index 0) due to roll
    signals = is_red_prev & is_green_curr & engulfs
    signals[0] = False
    
    indices = torch.nonzero(signals).squeeze()
    print(f"Found {len(indices)} Bullish Engulfing candidates.")
    
    if len(indices) == 0:
        return indices, torch.tensor([], device=device)

    # 2. Determine Outcomes (Vectorized Lookahead)
    # We need to check if price hits Target before Stop in next `lookahead` bars
    
    # Entry: Close
    entries = cl[signals]
    # Stop: Low of engulfing bar
    stops = lo[signals]
    risks = entries - stops
    
    # Filter out zero risk (doji/flat)
    valid_risk = risks > 0
    indices = indices[valid_risk]
    entries = entries[valid_risk]
    stops = stops[valid_risk]
    risks = risks[valid_risk]
    
    targets = entries + (risks * rr_target)
    
    # Prepare windows for outcome checking
    # We want to check [i+1 : i+1+lookahead] for each i in indices
    # Construct a matrix of indices: (Num_Signals, Lookahead)
    # signal_indices: [i1, i2, ...] -> [[i1+1, i1+2...], [i2+1, ...]]
    
    num_signals = len(indices)
    offsets = torch.arange(1, lookahead + 1, device=device).unsqueeze(0).expand(num_signals, lookahead)
    base_indices = indices.unsqueeze(1).expand(num_signals, lookahead)
    lookahead_indices = base_indices + offsets
    
    # Clamp to max length
    max_idx = len(ohlcv) - 1
    lookahead_indices = lookahead_indices.clamp(max=max_idx)
    
    # Gather Highs and Lows for the future windows
    # Shape: (Num_Signals, Lookahead)
    future_highs = hi[lookahead_indices]
    future_lows = lo[lookahead_indices]
    
    # Check hits
    # Hit Target: High >= Target
    # Hit Stop: Low <= Stop
    
    # Broadcast targets/stops to (Num_Signals, Lookahead)
    target_matrix = targets.unsqueeze(1).expand(num_signals, lookahead)
    stop_matrix = stops.unsqueeze(1).expand(num_signals, lookahead)
    
    hit_target_mask = future_highs >= target_matrix
    hit_stop_mask = future_lows <= stop_matrix
    
    # Find first occurrence of hit
    # We can use argmax on the boolean mask to find first True
    # But if no True, argmax returns 0. We need to handle "never hit".
    
    # Add a "sentinel" column at the end that is always True? No.
    # Let's convert to float, add a small ramp to favor earlier hits?
    # Simpler:
    # 1. Did it ever hit?
    did_hit_target = hit_target_mask.any(dim=1)
    did_hit_stop = hit_stop_mask.any(dim=1)
    
    # 2. When did it hit?
    # (argmax returns index of first True)
    target_idx = hit_target_mask.float().argmax(dim=1)
    stop_idx = hit_stop_mask.float().argmax(dim=1)
    
    # If it didn't hit, set index to infinity (or lookahead + 1)
    target_idx[~did_hit_target] = lookahead + 1
    stop_idx[~did_hit_stop] = lookahead + 1
    
    # 3. Outcome: Target hit before Stop?
    # success = (target_idx < stop_idx) AND (target_idx <= lookahead)
    success = (target_idx < stop_idx) & (target_idx < lookahead)
    
    labels = success.float() # 1.0 for Win, 0.0 for Loss/Timeout
    
    print(f"Win Rate (3:1 RR): {labels.mean().item():.4f}")
    
    return indices, labels

def extract_features(ohlcv, indices, window_size=32):
    """
    Extract previous `window_size` bars for each index.
    Normalize relative to the entry bar's Open price? Or last Close?
    Let's normalize relative to the 'Open' of the first bar in the window to capture shape.
    Or better: Log returns?
    Let's use: (Price - Window_Mean) / Window_Std for robust normalization.
    """
    num_samples = len(indices)
    
    # Create index matrix for windows
    # [i-window : i]
    offsets = torch.arange(-window_size, 0, device=device).unsqueeze(0).expand(num_samples, window_size)
    base_indices = indices.unsqueeze(1).expand(num_samples, window_size)
    window_indices = base_indices + offsets
    
    # Handle indices < 0 (padding) - though unlikely if we skip start
    window_indices = window_indices.clamp(min=0)
    
    # Gather data: (Num_Samples, Window, 5)
    # ohlcv is (N, 5)
    # We need fancy indexing
    # Flatten window_indices to gather then reshape
    flat_indices = window_indices.flatten()
    windows = ohlcv[flat_indices].view(num_samples, window_size, 5)
    
    # Normalize
    # Calculate mean/std per window per feature (dim 1)
    means = windows.mean(dim=1, keepdim=True)
    stds = windows.std(dim=1, keepdim=True) + 1e-6
    
    norm_windows = (windows - means) / stds
    
    # Transpose for CNN: (Batch, Channels, Length) -> (N, 5, Window)
    return norm_windows.transpose(1, 2)

class EngulfingCNN(nn.Module):
    def __init__(self, window_size=32):
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
        
        # Calculate flat size
        # 32 -> 16 -> 8
        flat_size = 64 * (window_size // 4)
        
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
            # No Sigmoid here, using BCEWithLogitsLoss
        )
        
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def train_model(X, y, epochs=20):
    model = EngulfingCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Handle class imbalance
    # Win rate is ~0.23, so neg/pos ratio is roughly 3:1
    num_pos = y.sum()
    num_neg = len(y) - num_pos
    pos_weight = num_neg / num_pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Batching
    batch_size = 1024
    num_samples = len(X)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Training on {num_samples} samples for {epochs} epochs (Pos Weight: {pos_weight:.2f})...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Shuffle
        perm = torch.randperm(num_samples, device=device)
        X = X[perm]
        y = y[perm]
        
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, num_samples)
            
            X_batch = X[start:end]
            y_batch = y[start:end].unsqueeze(1)
            
            optimizer.zero_grad()
            # Remove Sigmoid from model for BCEWithLogitsLoss? 
            # No, EngulfingCNN has Sigmoid at end. 
            # We should remove Sigmoid from model or use BCELoss.
            # Let's use BCELoss with manual weighting or just stick to BCEWithLogitsLoss and remove Sigmoid.
            # Easier: Just use BCELoss and manual weighting in the loop? No, BCELoss takes weight arg.
            # Actually, let's just modify the model to remove Sigmoid and use BCEWithLogitsLoss.
            outputs = model(X_batch) 
            # Wait, model has Sigmoid. 
            # Let's strip the sigmoid in the forward pass or just use BCELoss? 
            # BCELoss is unstable. Better to remove Sigmoid.
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.4f}")
            
    return model

def backtest(model, ohlcv, indices, labels, threshold=0.6):
    model.eval()
    
    # Extract features for all candidates
    X = extract_features(ohlcv, indices)
    
    with torch.no_grad():
        logits = model(X).squeeze()
        probs = torch.sigmoid(logits)
        
    # Filter trades
    trades_mask = probs > threshold
    
    num_trades = trades_mask.sum().item()
    if num_trades == 0:
        print("No trades taken with this threshold.")
        return
    
    # Calculate PnL
    # If label is 1 (Win), we gain 3R. If 0 (Loss), we lose 1R.
    # PnL = (Wins * 3) - (Losses * 1)
    
    actual_outcomes = labels[trades_mask]
    wins = actual_outcomes.sum().item()
    losses = num_trades - wins
    
    win_rate = wins / num_trades
    pnl_r = (wins * 3.0) - (losses * 1.0)
    
    print(f"--- Backtest Results (Threshold {threshold}) ---")
    print(f"Total Candidates: {len(indices)}")
    print(f"Trades Taken: {num_trades} ({num_trades/len(indices)*100:.1f}%)")
    print(f"Win Rate: {win_rate*100:.2f}%")
    print(f"Total PnL (R): {pnl_r:.2f} R")
    print(f"Expected Value per Trade: {pnl_r/num_trades:.2f} R")

def main():
    # 1. Load Data
    data_y1 = load_data_to_gpu('out/synthetic_year.csv')
    data_y2 = load_data_to_gpu('out/synthetic_year_2.csv')
    
    # 2. Label Data (Year 1)
    print("\n--- Processing Year 1 (Training) ---")
    indices_y1, labels_y1 = find_engulfing_setups(data_y1)
    X_y1 = extract_features(data_y1, indices_y1)
    
    # 3. Train Model
    model = train_model(X_y1, labels_y1)
    
    # 4. Backtest (Year 2)
    print("\n--- Processing Year 2 (Testing) ---")
    indices_y2, labels_y2 = find_engulfing_setups(data_y2)
    
    # Run backtest with different thresholds
    for thresh in [0.5, 0.6, 0.7, 0.8]:
        backtest(model, data_y2, indices_y2, labels_y2, threshold=thresh)

if __name__ == '__main__':
    main()
