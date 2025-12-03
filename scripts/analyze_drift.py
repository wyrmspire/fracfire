"""
Analyze Feature Drift

Compares feature distributions between Synthetic Archetypes and Real Data.
Helps identify if synthetic data is "too clean" or has different scale than real data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.data.loader import RealDataLoader
from src.training.data_loader import DataLoader
from src.features.builder import FeatureBuilder

def main():
    print("=" * 60)
    print("ANALYZING FEATURE DRIFT")
    print("=" * 60)
    
    # 1. Load Real Data
    print("Loading Real Data...")
    real_loader = RealDataLoader()
    real_path = root / "src" / "data" / "continuous_contract.json"
    real_df = real_loader.load_json(real_path)
    
    # 2. Load Synthetic Data (Sample)
    print("Loading Synthetic Data (Sample)...")
    syn_loader = DataLoader(root / "out" / "data" / "synthetic" / "archetypes")
    # Load just 50 files to get a representative distribution without being too slow
    syn_df = syn_loader.load_archetypes(limit=50)
    
    # 3. Extract Features
    print("Extracting features...")
    builder = FeatureBuilder(window_size=60)
    
    real_features = builder.extract_features(real_df)
    syn_features = builder.extract_features(syn_df)
    
    # 4. Compare Distributions
    features_to_check = [
        'volatility', 
        'tick_momentum', 
        'relative_range', 
        'volume_intensity',
        'tick_rsi',
        'cum_delta' # This one is tricky as it's cumulative, might not be comparable directly if day lengths differ
    ]
    
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(features_to_check):
        if feature not in real_features.columns:
            continue
            
        plt.subplot(2, 3, i+1)
        
        # Clip outliers for better visualization
        p01 = min(real_features[feature].quantile(0.01), syn_features[feature].quantile(0.01))
        p99 = max(real_features[feature].quantile(0.99), syn_features[feature].quantile(0.99))
        
        # Plot KDE
        sns.kdeplot(real_features[feature].clip(p01, p99), label='Real', fill=True, alpha=0.3)
        sns.kdeplot(syn_features[feature].clip(p01, p99), label='Synthetic', fill=True, alpha=0.3)
        
        plt.title(feature)
        plt.legend()
        
    plt.tight_layout()
    output_path = output_dir / "feature_drift.png"
    plt.savefig(output_path)
    print(f"\nDrift analysis chart saved to: {output_path}")
    
    # Print summary stats
    print("\nSummary Statistics Comparison:")
    print(f"{'Feature':<20} {'Real Mean':>10} {'Syn Mean':>10} {'Real Std':>10} {'Syn Std':>10}")
    print("-" * 65)
    
    for feature in features_to_check:
        r_mean = real_features[feature].mean()
        s_mean = syn_features[feature].mean()
        r_std = real_features[feature].std()
        s_std = syn_features[feature].std()
        
        print(f"{feature:<20} {r_mean:>10.4f} {s_mean:>10.4f} {r_std:>10.4f} {s_std:>10.4f}")

if __name__ == "__main__":
    main()
