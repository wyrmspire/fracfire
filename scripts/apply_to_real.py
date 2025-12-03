"""
Apply Model to Real Data

Loads the pre-trained baseline model and applies it to the real continuous contract data.
Saves the predictions for visualization.
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
    print("APPLYING MODEL TO REAL DATA")
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
    
    # 3. Load Model
    model_path = root / "out" / "models" / "baseline_rf.joblib"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
        
    print(f"Loading model from {model_path}...")
    clf = joblib.load(model_path)
    
    # 4. Predict
    print("Running inference...")
    # FeatureBuilder returns DataFrame, sklearn needs values
    X = features.values
    
    # Predict probabilities and classes
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)
    
    # 5. Save Results
    print("\nSaving results...")
    
    # Add predictions to original dataframe
    df['predicted_state'] = y_pred
    
    # Add confidence (max probability)
    df['confidence'] = np.max(y_prob, axis=1)
    
    # Save to parquet
    output_dir = root / "out" / "data" / "real" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "predicted_states.parquet"
    df.to_parquet(output_path)
    
    print(f"Predictions saved to: {output_path}")
    
    # Print summary
    print("\nPredicted State Distribution:")
    dist = df['predicted_state'].value_counts(normalize=True) * 100
    print(dist)

if __name__ == "__main__":
    main()
