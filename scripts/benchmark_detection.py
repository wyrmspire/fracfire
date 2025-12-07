import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.core.detector.engine import run_setups, SetupConfig, SetupFamilyConfig
from src.core.detector.indicators import IndicatorConfig

def benchmark_detection():
    # 1. Load Data
    csv_path = root / 'out' / 'synthetic_year.csv'
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run benchmark_generation.py first.")
        return

    print(f"Loading {csv_path}...")
    t0 = time.time()
    df_1m = pd.read_csv(csv_path)
    df_1m['time'] = pd.to_datetime(df_1m['time'])
    df_1m.set_index('time', inplace=True)
    t1 = time.time()
    print(f"Loaded {len(df_1m)} rows in {t1-t0:.4f}s")

    # 2. Resample to 5m
    print("Resampling to 5m...")
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df_5m = df_1m.resample('5min').agg(ohlc).dropna()
    t2 = time.time()
    print(f"Resampled to {len(df_5m)} 5m bars in {t2-t1:.4f}s")

    # 3. Configure Setups
    # Enable multiple families to test throughput
    cfg = SetupConfig(
        indicator_cfg=IndicatorConfig(
            ema_fast=20,
            ema_slow=200,
            rsi_period=14,
            atr_period=14
        ),
        orb=SetupFamilyConfig(enabled=True),
        ema200_continuation=SetupFamilyConfig(enabled=True),
        breakout=SetupFamilyConfig(enabled=True),
        reversal=SetupFamilyConfig(enabled=True),
        opening_push=SetupFamilyConfig(enabled=True),
        moc=SetupFamilyConfig(enabled=True)
    )

    # 4. Run Detection
    print("Running detection on 1 year of data...")
    t3 = time.time()
    outcomes = run_setups(df_1m, df_5m, cfg)
    t4 = time.time()
    
    detection_time = t4 - t3
    print(f"Detection Time: {detection_time:.4f}s")
    print(f"Found {len(outcomes)} setups")
    
    # Per-day metrics
    n_days = len(df_5m) / (12 * 24) # approx
    print(f"Speed: {detection_time / n_days * 1000:.2f} ms per day")

if __name__ == "__main__":
    benchmark_detection()
