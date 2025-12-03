"""
Synthetic Data Factory

A pipeline that:
1. Generates synthetic price data (PriceGenerator)
2. Calculates indicators (src.core.detector.indicators)
3. Detects setups (src.core.detector.engine)
4. Labels the data (src.ml.training.labeler)
5. Saves the result for training
"""

import sys
import os

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta
import uuid

from src.core.generator.engine import PriceGenerator, PhysicsConfig, MarketState
from src.core.detector.engine import run_setups, SetupConfig
from src.core.detector.indicators import add_5m_indicators, IndicatorConfig
from src.ml.training.labeler import DataLabeler

class SyntheticFactory:
    def __init__(self, output_dir: str = "data/synthetic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.labeler = DataLabeler()
        
    def generate_dataset(
        self, 
        num_days: int = 10, 
        physics_config: Optional[PhysicsConfig] = None,
        setup_config: Optional[SetupConfig] = None,
        prefix: str = "train"
    ):
        """
        Generate a batch of synthetic days, process them, and save to parquet.
        """
        if physics_config is None:
            physics_config = PhysicsConfig()
        if setup_config is None:
            setup_config = SetupConfig()
            
        gen = PriceGenerator(physics_config=physics_config)
        
        print(f"Generating {num_days} days of synthetic data...")
        
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        for i in range(num_days):
            # 1. Generate
            date = start_date - timedelta(days=i)
            df_1m = gen.generate_day(date)
            
            if df_1m.empty:
                continue
                
            df_1m.set_index('time', inplace=True)
            
            # 2. Resample to 5m for detection
            ohlc = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }
            df_5m = df_1m.resample("5min").agg(ohlc).dropna()
            
            # 3. Detect
            outcomes = run_setups(df_1m, df_5m, setup_config)
            
            # 4. Label
            # We label the 1m data because that's our finest granularity
            df_labeled = self.labeler.apply_labels(df_1m, outcomes)
            
            # 5. Save
            run_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
            filename = self.output_dir / f"{run_id}.parquet"
            df_labeled.to_parquet(filename)
            
            print(f"Saved {filename} with {len(outcomes)} setups")

if __name__ == "__main__":
    # Simple test run
    factory = SyntheticFactory(output_dir="lab/data/test_batch")
    factory.generate_dataset(num_days=2)
