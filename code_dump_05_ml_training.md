# Code Dump: 05_ml_training

## File: src/ml/training/data_loader.py
```python
"""
Data Loader Module

Handles loading, splitting, and preparing synthetic archetype data for training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm

class DataLoader:
    """
    Loads and manages archetype datasets.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        
    def load_archetypes(self, pattern: str = "*.parquet", limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load all archetypes into a single DataFrame with metadata.
        
        Args:
            pattern: Glob pattern for files
            limit: Max files to load (for testing)
            
        Returns:
            Combined DataFrame
        """
        files = list(self.data_dir.rglob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No files found in {self.data_dir} matching {pattern}")
            
        if limit:
            files = files[:limit]
            
        dfs = []
        print(f"Loading {len(files)} archetype files...")
        
        for f in tqdm(files, desc="Loading"):
            try:
                df = pd.read_parquet(f)
                
                # Add metadata
                # Folder name is archetype label (e.g., 'rally_day')
                archetype_label = f.parent.name
                
                # Use filename stem as run_id (e.g., 'rally_day_001')
                run_id = f.stem
                
                df['archetype'] = archetype_label
                df['run_id'] = run_id
                
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
                
        if not dfs:
            raise ValueError("No valid dataframes loaded")
            
        return pd.concat(dfs, ignore_index=True)
        
    def prepare_training_data(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets by run_id to avoid leakage.
        
        We must split by DAY (run_id), not by row, because rows within a day 
        are highly correlated.
        
        Args:
            df: Combined DataFrame
            test_size: Fraction of runs to use for testing
            seed: Random seed
            
        Returns:
            train_df, test_df
        """
        run_ids = df['run_id'].unique()
        rng = np.random.default_rng(seed)
        rng.shuffle(run_ids)
        
        split_idx = int(len(run_ids) * (1 - test_size))
        train_runs = run_ids[:split_idx]
        test_runs = run_ids[split_idx:]
        
        train_df = df[df['run_id'].isin(train_runs)].copy()
        test_df = df[df['run_id'].isin(test_runs)].copy()
        
        print(f"Split data: {len(train_runs)} training days, {len(test_runs)} test days")
        
        return train_df, test_df

```

---

## File: src/ml/training/labeler.py
```python
"""
Labeler Module

Responsible for taking raw price data and setup detection results,
and producing a labeled dataset where each timestep has targets for ML training.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

from src.core.detector.models import SetupOutcome

class DataLabeler:
    """
    Labels time-series data based on detected setups and outcomes.
    """
    
    def __init__(self):
        pass
        
    def apply_labels(self, df: pd.DataFrame, outcomes: List[SetupOutcome]) -> pd.DataFrame:
        """
        Add label columns to the DataFrame based on trade outcomes.
        
        Adds:
        - is_setup: bool (True if a setup triggered here)
        - setup_dir: int (1 for long, -1 for short, 0 for none)
        - outcome_r: float (Realized R-multiple of the trade)
        - outcome_win: bool (True if trade hit target)
        """
        df = df.copy()
        
        # Initialize columns
        df['is_setup'] = False
        df['setup_dir'] = 0
        df['outcome_r'] = 0.0
        df['outcome_win'] = False
        
        # Map outcomes to timestamps
        # Note: If multiple setups trigger on the same bar, the last one overwrites.
        # For simple training, this is acceptable. For complex cases, we might need list-columns.
        for outcome in outcomes:
            ts = outcome.entry.time
            if ts in df.index:
                df.at[ts, 'is_setup'] = True
                df.at[ts, 'setup_dir'] = 1 if outcome.entry.direction == 'long' else -1
                df.at[ts, 'outcome_r'] = outcome.r_multiple
                df.at[ts, 'outcome_win'] = outcome.hit_target
                
        return df

```

---

## File: src/ml/training/factory.py
```python
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

```

---

