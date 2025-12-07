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
    return df_15m

def prepare_box_breakout_dataset(df_15m, lookback=5):
    """
    Identify box breakouts and extract features/labels.
    Box defined by 00:00-05:00 session, scaled by 0.66.
    """
    # Group by day
    # We need to iterate day by day to define the box
    days = df_15m.groupby(df_15m.index.date)
    
    features = []
    labels = []
    returns = [] # For PnL calculation
    
    print(f"Processing {len(days)} days...")
    
    for date, day_df in days:
        # Define Session: 00:00 to 05:00
        # 05:00 bar is the one starting at 05:00? Or ending?
        # "midnight to 5". Let's assume up to 05:00 timestamp (exclusive of 05:00 bar price action? or inclusive?)
        # Usually "to 5" means 05:00 is the cutoff.
        # 15m bars: 00:00, 00:15, ..., 04:45. The 04:45 bar ends at 05:00.
        
        session = day_df.between_time('00:00', '04:59')
        if session.empty:
            continue
            
        midnight_open = session.iloc[0]['open']
        session_high = session['high'].max()
        session_low = session['low'].min()
        
        # Define Box (Scaled by 0.66 relative to Midnight Open)
        # BoxHigh = Open + 0.66 * (High - Open)
        # BoxLow = Open + 0.66 * (Low - Open)
        # Note: Low - Open is negative.
        
        box_high = midnight_open + 0.66 * (session_high - midnight_open)
        box_low = midnight_open + 0.66 * (session_low - midnight_open)
        
        box_range = box_high - box_low
        if box_range <= 0:
            continue
            
        # Scan for triggers from 05:00 onwards
        trading_session = day_df.between_time('05:00', '15:59') # Until 16:00 close
        
        # Convert to numpy for speed in loop
        ts_opens = trading_session['open'].values
        ts_highs = trading_session['high'].values
        ts_lows = trading_session['low'].values
        ts_closes = trading_session['close'].values
        ts_vols = trading_session['volume'].values
        ts_times = trading_session.index
        
        # We need global indices to look back for features
        # Map local index i to global index in df_15m
        # Or just extract features from day_df if possible.
        # Easier: Iterate trading_session, find timestamp, look up in day_df (or keep a rolling window).
        
        # Let's use the full day_df values and indices
        day_opens = day_df['open'].values
        day_highs = day_df['high'].values
        day_lows = day_df['low'].values
        day_closes = day_df['close'].values
        day_vols = day_df['volume'].values
        
        # Find start index of trading session in day_df
        if trading_session.empty:
            continue
            
        start_time = trading_session.index[0]
        # Find integer location of start_time in day_df
        # This is efficient enough for daily loop
        start_iloc = day_df.index.get_loc(start_time)
        
        for i in range(len(trading_session)):
            current_iloc = start_iloc + i
            
            # Check for Breakout (Close)
            close = ts_closes[i]
            
            signal = 0 # 0: None, 1: Long, -1: Short
            
            if close > box_high:
                signal = 1
            elif close < box_low:
                signal = -1
                
            if signal == 0:
                continue
                
            # We have a trigger.
            # 1. Extract Features (Previous 5 candles)
            # [current_iloc - 5 : current_iloc]
            # Note: "5 candles before it breaks out".
            # Does this include the breakout candle? "before it breaks out".
            # Usually means the setup *leading* to the break.
            # But the breakout candle itself contains info (momentum).
            # Let's use [current_iloc - 4 : current_iloc + 1] (5 bars ending with breakout)?
            # Or [current_iloc - 5 : current_iloc] (5 bars strictly before).
            # User said "price that is 5 candles before it breaks out".
            # Let's use strictly before to be predictive?
            # But the trigger is the close.
            # Let's use the 5 bars *including* the breakout bar, as that's the "pattern" we see when we decide.
            
            feat_start = current_iloc - lookback + 1
            feat_end = current_iloc + 1
            
            if feat_start < 0:
                continue
                
            win_o = day_opens[feat_start:feat_end]
            win_h = day_highs[feat_start:feat_end]
            win_l = day_lows[feat_start:feat_end]
            win_c = day_closes[feat_start:feat_end]
            win_v = day_vols[feat_start:feat_end]
            
            if len(win_o) < lookback:
                continue
                
            # Normalize
            prices = np.stack([win_o, win_h, win_l, win_c], axis=1)
            mean = prices.mean()
            std = prices.std() + 1e-6
            
            norm_o = (win_o - mean) / std
            norm_h = (win_h - mean) / std
            norm_l = (win_l - mean) / std
            norm_c = (win_c - mean) / std
            norm_v = (win_v - win_v.mean()) / (win_v.std() + 1e-6)
            
            feat = np.stack([norm_o, norm_h, norm_l, norm_c, norm_v], axis=1) # (5, 5)
            
            # 2. Determine Outcome
            # Entry = Close
            entry_price = close
            
            # Stop = Other side of box
            # Target = 1.4 * Risk
            if signal == 1: # Long
                stop_price = box_low
                risk = entry_price - stop_price
                target_price = entry_price + (risk * 1.4)
            else: # Short
                stop_price = box_high
                risk = stop_price - entry_price
                target_price = entry_price - (risk * 1.4)
                
            if risk <= 0:
                continue
                
            # Check future bars for outcome
            # Look ahead until end of day? "untill close".
            # Or "wait for midnight draw another box".
            # Let's look ahead until end of day data.
            
            future_o = day_opens[current_iloc+1:]
            future_h = day_highs[current_iloc+1:]
            future_l = day_lows[current_iloc+1:]
            
            outcome = 0 # 0: Loss, 1: Win
            
            hit_target = False
            hit_stop = False
            
            for k in range(len(future_h)):
                h = future_h[k]
                l = future_l[k]
                
                if signal == 1: # Long
                    if l <= stop_price:
                        hit_stop = True
                        break
                    if h >= target_price:
                        hit_target = True
                        break
                else: # Short
                    if h >= stop_price:
                        hit_stop = True
                        break
                    if l <= target_price:
                        hit_target = True
                        break
            
            # If neither hit by end of day, close at last price?
            # "untill close".
            # Let's assume close at end of day.
            
            r_result = 0.0
            
            if hit_target:
                r_result = 1.4
                outcome = 1
            elif hit_stop:
                r_result = -1.0
                outcome = 0
            else:
                # Close at end of day
                last_close = day_closes[-1]
                if signal == 1:
                    r_result = (last_close - entry_price) / risk
                else:
                    r_result = (entry_price - last_close) / risk
                
                # Binarize for training: > 0 is Win?
                # Or strict 1.4 target?
                # Let's say if r_result > 0.5 it's a "Win" for training purposes?
                # Or just train on Hit Target vs Hit Stop?
                # Let's use r_result > 0 as Win.
                outcome = 1 if r_result > 0 else 0
            
            features.append(feat)
            labels.append(outcome)
            returns.append(r_result)
            
    return np.array(features), np.array(labels), np.array(returns)

class BoxCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2), # 5 -> 2
            nn.Flatten()
        )
        flat_size = 32 * 2
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def train_model(X, y, epochs=30):
    X_t = torch.tensor(X, dtype=torch.float32).transpose(1, 2).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
    
    model = BoxCNN().to(device)
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

def evaluate(model, X, y, returns, threshold=0.5):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).transpose(1, 2).to(device)
    
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits)
        
    # Filter trades by threshold
    mask = probs.squeeze() > threshold
    
    selected_returns = returns[mask.cpu().numpy()]
    
    num_trades = len(selected_returns)
    if num_trades == 0:
        print(f"Threshold {threshold}: No trades.")
        return
        
    total_pnl = selected_returns.sum()
    win_rate = (selected_returns > 0).mean()
    
    print(f"--- Threshold {threshold} ---")
    print(f"Trades: {num_trades} ({num_trades/len(returns)*100:.1f}%)")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Total PnL: {total_pnl:.2f} R")
    print(f"EV: {total_pnl/num_trades:.2f} R")

def main():
    # 1. Load Real Data
    df = load_data('src/data/continuous_contract.json')
    
    # 2. Prepare Dataset
    print("\n--- Preparing Box Breakout Dataset ---")
    X, y, r = prepare_box_breakout_dataset(df)
    print(f"Total Samples: {len(X)}")
    
    # 3. Split 50/50
    split_idx = len(X) // 2
    X_train, y_train, r_train = X[:split_idx], y[:split_idx], r[:split_idx]
    X_test, y_test, r_test = X[split_idx:], y[split_idx:], r[split_idx:]
    
    print(f"Train Samples: {len(X_train)}")
    print(f"Test Samples: {len(X_test)}")
    
    # 4. Train
    print("\n--- Training on First Half ---")
    model = train_model(X_train, y_train)
    
    # 5. Test
    print("\n--- Testing on Second Half ---")
    for t in [0.0, 0.4, 0.5, 0.6, 0.7, 0.8]:
        evaluate(model, X_test, y_test, r_test, threshold=t)

if __name__ == '__main__':
    main()
