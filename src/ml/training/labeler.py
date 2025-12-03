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
