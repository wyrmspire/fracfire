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
    return df

def find_reversal_patterns(df_15m, df_1m):
    """
    Find 'Failure to make Lower Low + Break above Last High' (Bullish Reversal)
    and Inverse (Bearish Reversal) on 15m chart.
    Then extract next 10 1m candles for training.
    """
    # We need to iterate through 15m bars to find the pattern
    # Pattern (Bullish):
    # 1. Swing Low (L1)
    # 2. Swing High (H1)
    # 3. Higher Low (L2 > L1) -> "Fails to make lower low"
    # 4. Break above H1 -> Trigger
    
    # Simplified detection on rolling window of 3 bars?
    # Bar i-2: Low1
    # Bar i-1: High1 (between Low1 and Low2)
    # Bar i: Low2 > Low1
    # Trigger: Close[i] > High[i-1]?
    
    # Let's use a slightly more robust definition:
    # Identify local extrema (fractals)
    # But for speed, let's use a rolling window of 5 bars.
    # [i-4, i-3, i-2, i-1, i]
    
    # Bullish Reversal (Inverse H&S style):
    # L(i-4) is a low.
    # H(i-3) is a high.
    # L(i-2) is a higher low ( > L(i-4) ).
    # Breakout: Close(i) > H(i-3).
    
    # Bearish Reversal:
    # H(i-4) is a high.
    # L(i-3) is a low.
    # H(i-2) is a lower high ( < H(i-4) ).
    # Breakout: Close(i) < L(i-3).
    
    opens = df_15m['open'].values
    highs = df_15m['high'].values
    lows = df_15m['low'].values
    closes = df_15m['close'].values
    times = df_15m.index
    
    features = []
    labels = []
    returns = []
    
    print(f"Scanning {len(df_15m)} 15m bars for patterns...")
    
    for i in range(4, len(df_15m)):
        # Bullish Pattern
        l1 = lows[i-4]
        h1 = highs[i-3]
        l2 = lows[i-2]
        break_candle = closes[i]
        
        is_bullish = (l2 > l1) and (break_candle > h1)
        
        # Bearish Pattern
        h_a = highs[i-4]
        l_a = lows[i-3]
        h_b = highs[i-2]
        
        is_bearish = (h_b < h_a) and (break_candle < l_a)
        
        if not is_bullish and not is_bearish:
            continue
            
        # We found a pattern trigger at 15m bar `i`.
        # Trigger time is `times[i]`.
        trigger_time = times[i]
        
        # Get 1m data starting from trigger time
        # "train on the next 10 1m candles"
        # Does this mean the 10 1m candles *after* the trigger to predict outcome?
        # Or the 10 1m candles *leading up to* the trigger to confirm?
        # User said: "train on the next 10 1m candles... then take a 1 to 1.4 trade based on that signature"
        # This implies we look at the *immediate price action after the 15m break* (the "signature") 
        # to decide whether to take the trade.
        # So: Trigger happens -> Wait 10 minutes (observe 10 1m candles) -> Decide -> Enter.
        
        # Find location in 1m df
        # 15m bar timestamp is usually start time.
        # So bar `i` covers [time, time+15).
        # Trigger happens at Close of bar `i`, which is time+15.
        # So we look at 1m data starting at time+15.
        
        # 15m timestamp `times[i]` is the start of the 15m bar.
        # The close happens at `times[i] + 15min`.
        breakout_time = trigger_time + pd.Timedelta(minutes=15)
        
        # Get next 10 1m candles starting from breakout_time
        # Slice 1m df
        # We need efficient lookup.
        # Let's assume df_1m is sorted index.
        
        # Use searchsorted?
        # Or just loc slice
        
        # Slice: [breakout_time : breakout_time + 10min]
        # We need exactly 10 bars.
        
        start_slice = breakout_time
        end_slice = breakout_time + pd.Timedelta(minutes=10)
        
        # Note: 1m bars are [start, end).
        # Bars: 00, 01, ..., 09. (10 bars).
        
        m1_slice = df_1m[(df_1m.index >= start_slice) & (df_1m.index < end_slice)]
        
        if len(m1_slice) < 10:
            continue
            
        # Features: 10 1m candles
        win_o = m1_slice['open'].values
        win_h = m1_slice['high'].values
        win_l = m1_slice['low'].values
        win_c = m1_slice['close'].values
        win_v = m1_slice['volume'].values
        
        # Normalize
        prices = np.stack([win_o, win_h, win_l, win_c], axis=1)
        mean = prices.mean()
        std = prices.std() + 1e-6
        
        norm_o = (win_o - mean) / std
        norm_h = (win_h - mean) / std
        norm_l = (win_l - mean) / std
        norm_c = (win_c - mean) / std
        norm_v = (win_v - win_v.mean()) / (win_v.std() + 1e-6)
        
        feat = np.stack([norm_o, norm_h, norm_l, norm_c, norm_v], axis=1) # (10, 5)
        
        # Trade Execution
        # We enter *after* the 10th 1m candle closes.
        # Entry Price = Close of 10th 1m candle.
        entry_price = win_c[-1]
        
        # Stop/Target
        # "1 to 1.4 trade"
        # Based on what risk?
        # Usually pattern height or ATR.
        # Let's use the 15m pattern height (High - Low of the breakout bar? Or the swing?)
        # Let's use the 15m breakout bar range as a proxy for volatility.
        # Or ATR.
        # Let's use the range of the 10 1m candles we just observed.
        # Risk = (Max High - Min Low) of the 10m signature.
        
        sig_high = win_h.max()
        sig_low = win_l.min()
        risk_dist = sig_high - sig_low
        if risk_dist == 0: risk_dist = 1.0
        
        reward_dist = risk_dist * 1.4
        
        if is_bullish:
            target_price = entry_price + reward_dist
            stop_price = entry_price - risk_dist
        else:
            target_price = entry_price - reward_dist
            stop_price = entry_price + risk_dist
            
        # Outcome
        # Look ahead in 1m data from end_slice onwards
        future_1m = df_1m[df_1m.index >= end_slice]
        
        # Limit lookahead to say 4 hours (240 bars)
        future_1m = future_1m.iloc[:240]
        
        if future_1m.empty:
            continue
            
        f_h = future_1m['high'].values
        f_l = future_1m['low'].values
        f_c = future_1m['close'].values
        
        hit_target = False
        hit_stop = False
        
        for k in range(len(f_h)):
            h = f_h[k]
            l = f_l[k]
            
            if is_bullish:
                if l <= stop_price:
                    hit_stop = True
                    break
                if h >= target_price:
                    hit_target = True
                    break
            else:
                if h >= stop_price:
                    hit_stop = True
                    break
                if l <= target_price:
                    hit_target = True
                    break
                    
        r_result = 0.0
        if hit_target:
            r_result = 1.4
            outcome = 1
        elif hit_stop:
            r_result = -1.0
            outcome = 0
        else:
            # Close at end
            last_c = f_c[-1]
            if is_bullish:
                r_result = (last_c - entry_price) / risk_dist
            else:
                r_result = (entry_price - last_c) / risk_dist
            outcome = 1 if r_result > 0 else 0
            
        features.append(feat)
        labels.append(outcome)
        returns.append(r_result)
        
    return np.array(features), np.array(labels), np.array(returns)

class ReversalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2), # 10 -> 5
            nn.Flatten()
        )
        flat_size = 32 * 5
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
    
    model = ReversalCNN().to(device)
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
    df_1m = load_data('src/data/continuous_contract.json')
    
    # Resample to 15m for pattern detection
    ohlc = {'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}
    df_15m = df_1m.resample('15min').agg(ohlc).dropna()
    
    # 2. Prepare Dataset
    print("\n--- Preparing Reversal Pattern Dataset ---")
    X, y, r = find_reversal_patterns(df_15m, df_1m)
    print(f"Total Samples: {len(X)}")
    
    if len(X) < 10:
        print("Not enough samples.")
        return
    
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
    for t in [0.0, 0.4, 0.5, 0.6, 0.7]:
        evaluate(model, X_test, y_test, r_test, threshold=t)

if __name__ == '__main__':
    main()
