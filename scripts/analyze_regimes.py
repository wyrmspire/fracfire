"""
Analyze Regimes

Slices real market data into 2-hour overlapping patches and clusters them
to discover recurring market regimes (e.g., Open Drive, Chop, Grind).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.data.loader import RealDataLoader

def compute_fingerprint(df_slice: pd.DataFrame) -> dict:
    """
    Compute feature fingerprint for a 2-hour slice.
    """
    if len(df_slice) < 10:
        return None
        
    start_price = df_slice['open'].iloc[0]
    end_price = df_slice['close'].iloc[-1]
    high = df_slice['high'].max()
    low = df_slice['low'].min()
    
    # 1. Net Move & Drift
    net_move_ticks = (end_price - start_price) / 0.25
    total_range_ticks = (high - low) / 0.25
    
    # Directionality: |Net Move| / Total Range (0 = pure chop, 1 = pure trend)
    directionality = abs(net_move_ticks) / max(1, total_range_ticks)
    
    # 2. Volatility
    avg_bar_range = df_slice['range_ticks'].mean()
    max_bar_range = df_slice['range_ticks'].max()
    
    # 3. Monotonicity (Trend consistency)
    # Fraction of bars that move in the direction of the net move
    if net_move_ticks > 0:
        monotonicity = (df_slice['delta_ticks'] > 0).mean()
    elif net_move_ticks < 0:
        monotonicity = (df_slice['delta_ticks'] < 0).mean()
    else:
        monotonicity = 0.0
        
    # 4. Choppiness / Reversals
    # Count how many times the sign of delta_ticks flips
    deltas = df_slice['delta_ticks'].values
    flips = np.sum(np.diff(np.sign(deltas[deltas != 0])) != 0)
    choppiness = flips / len(deltas)
    
    # 5. Volume Profile
    total_vol = df_slice['volume'].sum()
    # Skew: (First half vol - Second half vol) / Total vol
    mid = len(df_slice) // 2
    vol_skew = (df_slice['volume'].iloc[:mid].sum() - df_slice['volume'].iloc[mid:].sum()) / total_vol
    
    return {
        'net_move_ticks': net_move_ticks,
        'total_range_ticks': total_range_ticks,
        'directionality': directionality,
        'avg_bar_range': avg_bar_range,
        'max_bar_range': max_bar_range,
        'monotonicity': monotonicity,
        'choppiness': choppiness,
        'vol_skew': vol_skew,
        'total_vol': total_vol
    }

def main():
    print("=" * 60)
    print("ANALYZING 2-HOUR REGIMES")
    print("=" * 60)
    
    # 1. Load Real Data
    loader = RealDataLoader()
    data_path = root / "src" / "data" / "continuous_contract.json"
    df = loader.load_json(data_path)
    
    # 2. Slice into 2-hour patches
    # We'll use a sliding window of 120 minutes, stepping every 30 minutes
    print("\nSlicing data into 2-hour patches...")
    
    patches = []
    timestamps = []
    
    # Group by day first to avoid spanning across days
    # Assuming 'time' index is datetime
    days = df.groupby(df.index.date)
    
    for date, day_df in days:
        if len(day_df) < 120:
            continue
            
        # Sliding window
        # We assume 1-minute bars roughly. 
        # Better to use time-based indexing if possible, but iloc is faster for fixed steps
        # Let's assume continuous bars for now (loader sorts them)
        
        step = 30 # 30 min step
        window = 120 # 2 hours
        
        for i in range(0, len(day_df) - window, step):
            slice_df = day_df.iloc[i : i+window]
            fp = compute_fingerprint(slice_df)
            if fp:
                patches.append(fp)
                timestamps.append(slice_df.index[0])
                
    patch_df = pd.DataFrame(patches)
    print(f"Extracted {len(patch_df)} patches.")
    
    # 3. Cluster
    print("\nClustering regimes...")
    
    # Features for clustering (normalize them)
    features = ['directionality', 'avg_bar_range', 'choppiness', 'monotonicity', 'max_bar_range']
    X = patch_df[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means with k=8 (as suggested in directive)
    k = 8
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    patch_df['cluster'] = labels
    
    # 4. Analyze Clusters
    print("\nCluster Analysis:")
    print(f"{'Cluster':<8} {'Count':<6} {'Direct.':<8} {'AvgRng':<8} {'Chop':<8} {'Mono':<8} {'Description'}")
    print("-" * 80)
    
    cluster_stats = patch_df.groupby('cluster')[features].mean()
    
    # Heuristic naming
    descriptions = {}
    for c in range(k):
        stats = cluster_stats.loc[c]
        desc = []
        if stats['directionality'] > 0.6: desc.append("Trend")
        elif stats['directionality'] < 0.3: desc.append("Range")
        else: desc.append("Mixed")
        
        if stats['avg_bar_range'] > patch_df['avg_bar_range'].mean() * 1.2: desc.append("Volatile")
        elif stats['avg_bar_range'] < patch_df['avg_bar_range'].mean() * 0.8: desc.append("Quiet")
        
        if stats['choppiness'] > 0.6: desc.append("Choppy")
        
        name = " ".join(desc)
        descriptions[c] = name
        
        print(f"{c:<8} {sum(labels==c):<6} {stats['directionality']:<8.2f} {stats['avg_bar_range']:<8.2f} {stats['choppiness']:<8.2f} {stats['monotonicity']:<8.2f} {name}")

    # 5. Save Results
    output_dir = root / "out" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save cluster centers
    joblib.dump(kmeans, output_dir / "regime_kmeans.joblib")
    joblib.dump(scaler, output_dir / "regime_scaler.joblib")
    
    # Save labeled patches
    patch_df['description'] = patch_df['cluster'].map(descriptions)
    patch_df.to_csv(output_dir / "regime_patches.csv")
    
    print(f"\nResults saved to {output_dir}")
    
    # 6. Visualize
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=patch_df, x='directionality', y='avg_bar_range', hue='cluster', palette='tab10', alpha=0.6)
    plt.title("Regime Clusters: Directionality vs Volatility")
    plt.savefig(root / "out" / "charts" / "regime_clusters.png")
    print("Chart saved to out/charts/regime_clusters.png")

if __name__ == "__main__":
    main()
