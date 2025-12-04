
import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def create_cnn_model(input_shape):
    """
    Simple 1D CNN for time series classification.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def prepare_features(df, window_size=60):
    """
    Create sequences of features for the CNN.
    Features: Open, High, Low, Close, Volume (normalized relative to window start)
    """
    # We need to align X (features) with y (labels)
    # y is already in the dataframe as 'label_long_3r'
    
    # Drop rows with NaN labels or features
    df = df.dropna()
    
    # Feature columns
    feature_cols = ['open', 'high', 'low', 'close', 'volume']
    data = df[feature_cols].values
    labels = df['label_long_3r'].values # Focusing on Longs for now
    
    X = []
    y = []
    
    # Sliding window
    # We need 'window_size' bars ending at index 'i' to predict label at 'i'
    # So range starts at window_size
    
    print("Preparing features...")
    for i in range(window_size, len(data)):
        window = data[i-window_size:i]
        
        # Normalize window relative to first bar in window
        # Prices / Open[0] - 1.0
        # Volume / Volume[0] (add epsilon)
        
        base_price = window[0, 0] # Open of first bar
        base_vol = window[0, 4] + 1e-9
        
        norm_window = np.zeros_like(window)
        norm_window[:, :4] = (window[:, :4] / base_price) - 1.0
        norm_window[:, 4] = (window[:, 4] / base_vol) - 1.0
        
        X.append(norm_window)
        y.append(labels[i])
        
    return np.array(X), np.array(y)

def main():
    print("Loading Synthetic Data...")
    synth_path = "data/processed/synthetic_3m.parquet"
    if not os.path.exists(synth_path):
        print(f"Error: {synth_path} not found. Run generate_training_data.py first.")
        return
        
    df_synth = pd.read_parquet(synth_path)
    X_synth, y_synth = prepare_features(df_synth)
    
    print(f"Synthetic Data Shape: {X_synth.shape}")
    print(f"Class Balance: {y_synth.mean():.2%}")
    
    # Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(X_synth, y_synth, test_size=0.2, shuffle=False)
    
    print("Training CNN on Synthetic Data...")
    model = create_cnn_model((X_train.shape[1], X_train.shape[2]))
    
    history = model.fit(
        X_train, y_train,
        epochs=2,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Save Model
    model.save("out/cnn_synthetic_3r.keras")
    
    # --- Real Data Comparison ---
    print("\n--- Testing on Real Data ---")
    real_path = "data/processed/real_3m.parquet"
    
    if not os.path.exists(real_path):
        print(f"Warning: {real_path} not found. Skipping real data test.")
        print("Please process real data first.")
        return
        
    df_real = pd.read_parquet(real_path)
    X_real, y_real = prepare_features(df_real)
    
    print(f"Real Data Shape: {X_real.shape}")
    print(f"Real Class Balance (Base Rate): {y_real.mean():.2%}")
    
    # Evaluate
    loss, acc = model.evaluate(X_real, y_real, verbose=0)
    print(f"Model Accuracy on Real Data: {acc:.2%}")
    
    # Precision/Recall Analysis
    probs = model.predict(X_real)
    threshold = 0.7 # High confidence only
    
    preds = (probs > threshold).astype(int).flatten()
    n_triggers = preds.sum()
    
    if n_triggers > 0:
        win_rate = y_real[preds == 1].mean()
        print(f"\nHigh Confidence Triggers (> {threshold}): {n_triggers}")
        print(f"Win Rate of Triggers: {win_rate:.2%}")
        print(f"Lift over Base Rate: {win_rate / y_real.mean():.2f}x")
    else:
        print(f"\nNo triggers found with confidence > {threshold}")

    # --- Baseline: Train on Real Data ---
    print("\n--- Baseline: Training on Real Data (Train/Test Split) ---")
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real, y_real, test_size=0.2, shuffle=False)
    
    print("Training CNN on Real Data...")
    model_real = create_cnn_model((X_train_real.shape[1], X_train_real.shape[2]))
    
    model_real.fit(
        X_train_real, y_train_real,
        epochs=2,
        batch_size=32,
        validation_data=(X_test_real, y_test_real),
        verbose=1
    )
    
    probs_real = model_real.predict(X_test_real)
    preds_real = (probs_real > threshold).astype(int).flatten()
    n_triggers_real = preds_real.sum()
    
    base_rate_test = y_test_real.mean()
    
    if n_triggers_real > 0:
        win_rate_real = y_test_real[preds_real == 1].mean()
        print(f"\n[Real->Real] High Confidence Triggers: {n_triggers_real}")
        print(f"[Real->Real] Win Rate: {win_rate_real:.2%}")
        print(f"[Real->Real] Base Rate: {base_rate_test:.2%}")
        print(f"[Real->Real] Lift: {win_rate_real / base_rate_test:.2f}x")
    else:
        print(f"\n[Real->Real] No triggers found.")

if __name__ == "__main__":
    main()
