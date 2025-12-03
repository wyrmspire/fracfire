"""
Apply Optimized Model

Applies the balanced Random Forest model to real data with custom probability thresholds.
This allows detecting directional moves even when the signal is weak due to low volatility.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.data.loader import RealDataLoader
from src.features.builder import FeatureBuilder

def main():
    print("=" * 60)
    print("APPLYING OPTIMIZED MODEL (Balanced + Thresholds)")
    print("=" * 60)
    
    # 1. Load Real Data
    data_path = root / "src" / "data" / "continuous_contract.json"
    loader = RealDataLoader()
    try:
        df = loader.load_json(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Build Features
    print("\nExtracting features...")
    builder = FeatureBuilder(window_size=60)
    features = builder.extract_features(df)
    
    # 3. Load Balanced Model
    model_path = root / "out" / "models" / "balanced_rf.joblib"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
        
    print(f"Loading model from {model_path}...")
    clf = joblib.load(model_path)
    
    # 4. Predict Probabilities
    print("Running inference...")
    X = features.values
    y_prob = clf.predict_proba(X)
    classes = clf.classes_
    
    # 5. Apply Custom Thresholds
    # We want to prioritize directional moves.
    # If P(Rally) > threshold, predict Rally.
    # Order matters! Check rarest/most important first.
    
    # Map class names to indices
    class_map = {c: i for i, c in enumerate(classes)}
    
    # Define thresholds (tuned heuristically based on drift analysis)
    # Real data has much lower volatility, so confidence will be lower.
    THRESHOLDS = {
        'breakdown': 0.30,
        'rally': 0.30,
        'impulsive': 0.35,
        'breakout': 0.25
    }
    
    print("\nApplying thresholds:")
    for c, t in THRESHOLDS.items():
        print(f"  {c}: > {t}")
        
    final_preds = []
    
    # Vectorized approach would be faster, but loop is clearer for logic
    # Let's try a vectorized approach for speed
    
    # Default to 'ranging' (or whatever max prob is if not ranging)
    # Actually, let's start with standard argmax
    max_indices = np.argmax(y_prob, axis=1)
    base_preds = classes[max_indices]
    
    # Create a Series for easy updating
    pred_series = pd.Series(base_preds, index=df.index)
    
    # Apply overrides
    # We iterate through thresholds. Later ones overwrite earlier ones if multiple trigger?
    # Usually we want the "strongest" signal.
    # But here we just want ANY directional signal to override 'ranging'.
    
    for state, threshold in THRESHOLDS.items():
        if state not in class_map:
            continue
            
        idx = class_map[state]
        probs = y_prob[:, idx]
        
        # Where probability exceeds threshold, force this state
        # Note: This is a simple override. If multiple exceed, the last one in loop wins.
        # Ideally we'd check which exceeds its threshold by the most relative margin.
        # But for now, let's just apply them.
        mask = probs > threshold
        pred_series[mask] = state
        
    df['predicted_state'] = pred_series.values
    
    # 6. Save Results
    output_dir = root / "out" / "data" / "real" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "optimized_states.parquet"
    df.to_parquet(output_path)
    
    print(f"\nPredictions saved to: {output_path}")
    
    # Print summary
    print("\nOptimized State Distribution:")
    dist = df['predicted_state'].value_counts(normalize=True) * 100
    print(dist)

if __name__ == "__main__":
    main()
