"""
Train Balanced Model

Trains a Random Forest classifier with class_weight='balanced' to address
the dominance of 'ranging' states and improve sensitivity to directional moves.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.training.data_loader import DataLoader
from src.features.builder import FeatureBuilder

def main():
    print("=" * 60)
    print("TRAINING BALANCED MODEL")
    print("=" * 60)
    
    # 1. Load Data
    data_dir = root / "out" / "data" / "synthetic" / "archetypes"
    loader = DataLoader(data_dir)
    
    print("Loading data...")
    df = loader.load_archetypes()
    train_df, test_df = loader.prepare_training_data(df)
    
    # 2. Build Features
    print("\nExtracting features...")
    builder = FeatureBuilder(window_size=60)
    
    X_train, y_train = builder.create_dataset(train_df, target_col='state')
    X_test, y_test = builder.create_dataset(test_df, target_col='state')
    
    # 3. Train with Class Weights
    print("\nTraining Balanced Random Forest...")
    # class_weight='balanced' automatically adjusts weights inversely proportional to class frequencies
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight='balanced',  # <--- KEY CHANGE
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    clf.fit(X_train, y_train)
    
    # 4. Evaluate
    print("\nEvaluating...")
    y_pred = clf.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Save Model
    model_dir = root / "out" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "balanced_rf.joblib"
    
    joblib.dump(clf, model_path)
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main()
