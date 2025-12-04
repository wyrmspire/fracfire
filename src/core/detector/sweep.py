"""
Parameter Sweep Module

Systematically explores parameter grids for setup families to find optimal configurations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import itertools
import copy
import pandas as pd
import numpy as np

from src.core.pipeline.scenario_runner import ScenarioSpec, run_scenario, SetupConfig

@dataclass
class SweepResult:
    setup_name: str
    params: Dict[str, Any]
    num_trades: int
    win_rate: float
    avg_r: float
    total_r: float
    notes: str = ""

def sweep_setup_family(
    family_name: str,
    base_spec: ScenarioSpec,
    param_grid: Dict[str, List[Any]],
    seed: int = 42,
) -> List[SweepResult]:
    """
    Run a parameter sweep for a specific setup family.
    
    Args:
        family_name: Name of the setup family (e.g., "orb")
        base_spec: Base scenario specification
        param_grid: Dictionary mapping parameter names to lists of values
        seed: Random seed
        
    Returns:
        List of SweepResult objects
    """
    results = []
    
    # Generate all combinations of parameters
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"Sweeping {family_name} with {len(combinations)} combinations...")
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        # Create a copy of the spec to modify
        # We need to deepcopy the setup_cfg to avoid side effects
        spec = copy.deepcopy(base_spec)
        
        # Ensure only the target family is active
        spec.setup_cfg.active_families = [family_name]
        
        # Apply parameters to the specific setup config
        # We assume the setup config has an attribute matching the family name
        if not hasattr(spec.setup_cfg, family_name):
            print(f"Error: SetupConfig has no attribute '{family_name}'")
            continue
            
        family_config = getattr(spec.setup_cfg, family_name)
        
        for k, v in params.items():
            if hasattr(family_config, k):
                setattr(family_config, k, v)
            else:
                print(f"Warning: {family_name} config has no attribute '{k}'")
        
        # Run Scenario
        # Note: For efficiency, we should ideally generate data once and reuse it,
        # but run_scenario currently does everything. 
        # If the sweep only changes setup params, we can optimize by generating data outside loop.
        # But run_scenario is the API. Let's stick to it for simplicity for now.
        # Optimization: If we pass a pre-generated dataframe to run_scenario? 
        # run_scenario doesn't support that yet.
        # Given this is a "backend console" for dev, re-generating (or re-loading) is acceptable for now.
        
        result = run_scenario(spec, seed=seed)
        
        # Compute Metrics
        outcomes = result.outcomes
        num_trades = len(outcomes)
        
        if num_trades > 0:
            wins = sum(1 for o in outcomes if o.hit_target)
            win_rate = wins / num_trades
            total_r = sum(o.r_multiple for o in outcomes)
            avg_r = total_r / num_trades
        else:
            win_rate = 0.0
            total_r = 0.0
            avg_r = 0.0
            
        results.append(SweepResult(
            setup_name=family_name,
            params=params,
            num_trades=num_trades,
            win_rate=win_rate,
            avg_r=avg_r,
            total_r=total_r
        ))
        
        if (i + 1) % 5 == 0:
            print(f"Completed {i + 1}/{len(combinations)} runs.")
            
    return results
