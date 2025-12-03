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
