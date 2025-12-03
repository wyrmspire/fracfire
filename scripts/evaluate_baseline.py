"""
Evaluate Baseline Model

Loads the trained Random Forest model and evaluates it on a held-out test set.
Generates confusion matrices and feature importance plots.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.training.data_loader import DataLoader
from src.features.builder import FeatureBuilder

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    print("=" * 60)
    print("EVALUATING BASELINE MODEL")
    print("=" * 60)
    
    # 1. Load Model
    model_path = root / "out" / "models" / "baseline_rf.joblib"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
        
    print(f"Loading model from {model_path}...")
    clf = joblib.load(model_path)
    
    # 2. Load Data (Test Set)
    # We need to reload and resplit to ensure we get the same test set
    # ideally we would have saved the split, but for now we rely on seed
    data_dir = root / "out" / "data" / "synthetic" / "archetypes"
    loader = DataLoader(data_dir)
    
    print("Loading data...")
    df = loader.load_archetypes()
    _, test_df = loader.prepare_training_data(df, seed=42) # Must use same seed!
    
    # 3. Build Features
    print("Extracting features...")
    builder = FeatureBuilder(window_size=60)
    X_test, y_test = builder.create_dataset(test_df, target_col='state')
    
    # 4. Predict
    print("Running inference...")
    y_pred = clf.predict(X_test)
    
    # 5. Visualize
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion Matrix
    labels = sorted(list(set(y_test)))
    cm_path = output_dir / "baseline_confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, labels, cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Feature Importance
    feature_names = builder.extract_features(test_df.iloc[:5]).columns
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    fi_path = output_dir / "baseline_feature_importance.png"
    plt.savefig(fi_path)
    print(f"Feature importance plot saved to: {fi_path}")

if __name__ == "__main__":
    main()
