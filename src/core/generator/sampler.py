
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

@dataclass
class PhysicsProfile:
    """Physics parameters derived from a real market segment."""
    volatility: float
    trend_efficiency: float
    direction: float
    wick_ratio: float
    source_segment_start: pd.Timestamp

class PhysicsSampler:
    """
    Retrieves historical physics profiles based on current market context.
    Acts as a 'Generative Model' by sampling from the library of real segments.
    """
    
    def __init__(self, library_path: str = "data/processed/physics_library.parquet"):
        self.library_path = Path(library_path)
        self.library = None
        self._load_library()
        
    def _load_library(self):
        if not self.library_path.exists():
            print(f"Warning: Physics library not found at {self.library_path}")
            return
            
        self.library = pd.read_parquet(self.library_path)
        print(f"Loaded Physics Library: {len(self.library)} segments.")
        
    def sample_physics(self, 
                       current_time: pd.Timestamp, 
                       recent_volatility: float,
                       recent_trend: float,
                       recent_direction: float) -> Optional[PhysicsProfile]:
        """
        Find historical segments that match the current context and sample one.
        
        Context:
        - Time of Day (Hour)
        - Recent Volatility (relative match)
        - Recent Trend/Direction (relative match)
        """
        if self.library is None:
            return None
            
        # Filter by Time of Day (Hour +/- 1)
        # We want to capture the "vibe" of this time.
        target_hour = current_time.hour
        
        # Handle wrapping for 23 -> 0
        hours = [target_hour]
        # hours = [(target_hour - 1) % 24, target_hour, (target_hour + 1) % 24] # Broaden search?
        
        # Initial Filter: Time of Day
        candidates = self.library[self.library['hour'].isin(hours)]
        
        if candidates.empty:
            # Fallback to any time if no match (unlikely)
            candidates = self.library
            
        # Calculate Similarity Score
        # We want segments where the PREVIOUS 2 hours looked like OUR recent history.
        
        # Normalize metrics for distance calculation
        # Simple Euclidean distance on normalized features
        
        # Features: prev_volatility, prev_trend, prev_direction
        
        # Scale factors (approximate std devs from data)
        vol_scale = 10.0 
        trend_scale = 0.5
        dir_scale = 0.01
        
        dist = (
            ((candidates['prev_volatility'] - recent_volatility) / vol_scale) ** 2 +
            ((candidates['prev_trend'] - recent_trend) / trend_scale) ** 2 +
            ((candidates['prev_direction'] - recent_direction) / dir_scale) ** 2
        )
        
        # Select top N matches (Nearest Neighbors)
        # We want some variety, so pick from top 50
        top_n = 50
        closest_indices = dist.nsmallest(top_n).index
        
        if len(closest_indices) == 0:
            return None
            
        # Randomly sample one
        chosen_idx = np.random.choice(closest_indices)
        row = self.library.loc[chosen_idx]
        
        return PhysicsProfile(
            volatility=row['volatility'],
            trend_efficiency=row['trend_efficiency'],
            direction=row['direction'],
            wick_ratio=row['wick_ratio'],
            source_segment_start=row['start_time']
        )
