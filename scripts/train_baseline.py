"""
Train Baseline Model

Trains a Random Forest classifier to predict market states (RALLY, RANGING, etc.)
based on tick features extracted from synthetic archetypes.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.training.data_loader import DataLoader
from src.features.builder import FeatureBuilder

def main():
    print("=" * 60)
    print("TRAINING BASELINE MODEL (Random Forest)")
    print("=" * 60)
    
    # 1. Load Data
    data_dir = root / "out" / "data" / "synthetic" / "archetypes"
    loader = DataLoader(data_dir)
    
    # Load all data (might be large, so be careful)
    # For baseline, maybe limit if too slow
    print("Loading data...")
    df = loader.load_archetypes()
    
    # Split
    train_df, test_df = loader.prepare_training_data(df)
    
    # 2. Build Features
    print("\nExtracting features...")
    builder = FeatureBuilder(window_size=60)
    
    # We predict 'state' column
    X_train, y_train = builder.create_dataset(train_df, target_col='state')
    X_test, y_test = builder.create_dataset(test_df, target_col='state')
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 3. Train
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
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
    model_path = model_dir / "baseline_rf.joblib"
    
    joblib.dump(clf, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # 6. Feature Importance
    feature_names = builder.extract_features(train_df.iloc[:5]).columns
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Importances:")
    for f in range(min(10, len(feature_names))):
        print(f"{f+1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")

if __name__ == "__main__":
    main()
