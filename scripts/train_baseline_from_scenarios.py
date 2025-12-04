"""
Baseline Training Script

Trains a simple baseline model from generated scenarios/data.
Verifies the end-to-end pipeline from synthetic data to model artifact.
"""

import argparse
import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ml.features.builder import FeatureBuilder

def check_venv():
    """Check if running in a virtual environment."""
    if sys.prefix == sys.base_prefix:
        print("WARNING: You are NOT running inside a virtual environment (venv).")
        print("Please activate your venv and try again.")

class BaselineModel(nn.Module):
    """Simple linear model for baseline verification."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def load_data(data_root: str) -> pd.DataFrame:
    """Load all parquet/csv files from data_root."""
    files = glob.glob(os.path.join(data_root, "*.parquet"))
    if not files:
        files = glob.glob(os.path.join(data_root, "*.csv"))
        
    if not files:
        print(f"No data files found in {data_root}")
        return pd.DataFrame()
        
    dfs = []
    for f in files:
        try:
            if f.endswith(".parquet"):
                df = pd.read_parquet(f)
            else:
                df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def train_baseline(data_root: str, model_out: str):
    print(f"Loading data from {data_root}...")
    df = load_data(data_root)
    
    if df.empty:
        print("No data loaded. Exiting.")
        return
        
    print(f"Loaded {len(df)} rows.")
    
    # Prepare Features
    # We assume the dataframe might already have features or we need to build them.
    # If it's raw 1m data, we use FeatureBuilder.
    # If it's labeled data from Factory, it might have labels.
    
    # Check for target column
    target_col = "is_setup" # Example target
    if target_col not in df.columns:
        # If no target, we can't train. 
        # But for this verification script, we might want to generate features first.
        print(f"Target column '{target_col}' not found. Attempting to build features/targets...")
        # This part depends on how we want to define the baseline.
        # Let's assume we want to predict 'close > open' (simple direction) if no setup label.
        df['target'] = (df['close'] > df['open']).astype(int)
        target_col = 'target'
    
    builder = FeatureBuilder()
    try:
        # We need to handle the case where df is just a concatenation of days.
        # FeatureBuilder expects a single continuous dataframe usually, or we apply per group.
        # For simplicity, apply to whole df (ignoring day boundaries for baseline).
        features = builder.extract_features(df)
        
        # Drop non-numeric columns for X
        X_df = features.select_dtypes(include=[np.number])
        # Remove target from X if present
        if target_col in X_df.columns:
            X_df = X_df.drop(columns=[target_col])
            
        X = X_df.values.astype(np.float32)
        y = df[target_col].values.astype(np.float32).reshape(-1, 1)
        
        # Simple Train Loop
        input_dim = X.shape[1]
        model = BaselineModel(input_dim, 1)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        print(f"Training baseline model on {len(X)} samples (Input Dim: {input_dim})...")
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
                
        # Save Model
        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        torch.save(model.state_dict(), model_out)
        print(f"Saved baseline model to {model_out}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    check_venv()
    parser = argparse.ArgumentParser(description="Train baseline model from scenarios")
    parser.add_argument("--data-root", type=str, required=True, help="Folder containing training data")
    parser.add_argument("--model-out", type=str, default="out/models/baseline.pt", help="Output path for model")
    args = parser.parse_args()
    
    train_baseline(args.data_root, args.model_out)

if __name__ == "__main__":
    main()
