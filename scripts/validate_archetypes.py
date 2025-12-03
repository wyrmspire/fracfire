"""
Validate Synthetic Archetypes

Verifies the integrity and statistical properties of the generated archetype library.
Checks if the generated patterns match their intended definitions.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from lab.generators.utils import summarize_day

# Configuration
ARCHETYPE_DIR = root / "out" / "data" / "synthetic" / "archetypes"
SAMPLES_TO_CHECK = 20  # Check random 20 samples per archetype to save time

def validate_rally_day(stats: Dict) -> bool:
    """Rally day should have positive net move and decent range"""
    return stats['overall']['net_move'] > 0 and stats['overall']['total_range_ticks'] > 50

def validate_range_day(stats: Dict) -> bool:
    """Range day should have low net move relative to range"""
    net_move = abs(stats['overall']['net_move_ticks'])
    total_range = stats['overall']['total_range_ticks']
    return net_move < (total_range * 0.6)

def validate_breakout(stats: Dict) -> bool:
    """Breakout should have significant move"""
    return abs(stats['overall']['net_move_ticks']) > 50

def validate_zombie(stats: Dict) -> bool:
    """Zombie should be low volatility but directional"""
    # Low average bar range
    return stats['overall']['avg_range_ticks'] < 10

def validate_volatile(stats: Dict) -> bool:
    """Volatile should have high average range"""
    return stats['overall']['avg_range_ticks'] > 15

def validate_news_event(stats: Dict) -> bool:
    """News event should have a volatility spike (high max range)"""
    # Even if average is low, max bar range should be high
    return stats['overall'].get('max_range_ticks', 0) > 25

VALIDATORS = {
    "rally_day": validate_rally_day,
    "range_day": validate_range_day,
    "breakout_pattern": validate_breakout,
    "breakdown_pattern": validate_breakout, # Same logic (big move)
    "reversal_pattern": None, # Hard to validate with simple stats
    "zombie_grind": validate_zombie,
    "volatile_chop": validate_volatile,
    "opening_bell": None,
    "closing_squeeze": None,
    "news_event": validate_news_event, # Should be volatile spike
}

def main():
    print("=" * 60)
    print("VALIDATING ARCHETYPES")
    print("=" * 60)
    
    if not ARCHETYPE_DIR.exists():
        print(f"Archetype directory not found: {ARCHETYPE_DIR}")
        print("Run generate_archetypes.py first.")
        sys.exit(1)
    
    results = {}
    
    for archetype_path in ARCHETYPE_DIR.iterdir():
        if not archetype_path.is_dir():
            continue
            
        name = archetype_path.name
        print(f"\nChecking {name}...")
        
        files = list(archetype_path.glob("*.parquet"))
        if not files:
            print(f"  No files found!")
            continue
            
        # Sample files
        if len(files) > SAMPLES_TO_CHECK:
            files = np.random.choice(files, SAMPLES_TO_CHECK, replace=False)
        
        passed = 0
        failed = 0
        errors = 0
        
        validator = VALIDATORS.get(name)
        
        for file_path in tqdm(files):
            try:
                df = pd.read_parquet(file_path)
                
                # Check required columns
                required_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'state']
                if not all(col in df.columns for col in required_cols):
                    print(f"  Missing columns in {file_path.name}")
                    errors += 1
                    continue
                
                # Run stats
                stats = summarize_day(df)
                
                # Run specific validation if exists
                if validator:
                    if validator(stats):
                        passed += 1
                    else:
                        failed += 1
                else:
                    passed += 1 # No specific validator, just load check
                    
            except Exception as e:
                print(f"  Error reading {file_path.name}: {e}")
                errors += 1
        
        results[name] = {
            "total": len(files),
            "passed": passed,
            "failed": failed,
            "errors": errors
        }
        
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"{'Archetype':<20} {'Checked':>8} {'Passed':>8} {'Failed':>8} {'Errors':>8}")
    print("-" * 60)
    
    for name, res in results.items():
        print(f"{name:<20} {res['total']:>8} {res['passed']:>8} {res['failed']:>8} {res['errors']:>8}")
        
    print("=" * 60)

if __name__ == "__main__":
    main()
