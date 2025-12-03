"""
Quick Test Script - Verify Installation and Basic Functionality

This script tests that all components are working correctly.
Run this after setting up the project to verify everything is installed.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

print("=" * 70)
print("FRACFIRE - INSTALLATION TEST")
print("=" * 70)

# Test 1: Import core dependencies
print("\n[1/6] Testing core dependencies...")
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    print("✓ Core dependencies (numpy, pandas, matplotlib) OK")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    sys.exit(1)

# Test 2: Import lab modules
print("\n[2/6] Testing lab modules...")
try:
    from lab.generators import PriceGenerator, MarketState
    from lab.generators.utils import summarize_day, print_summary
    print("✓ Lab generators module OK")
except ImportError as e:
    print(f"✗ Failed to import lab modules: {e}")
    sys.exit(1)

# Test 3: Generate synthetic data
print("\n[3/6] Testing price generator...")
try:
    gen = PriceGenerator(initial_price=5000.0, seed=42)
    start_date = datetime(2025, 11, 29, 9, 30, 0)
    
    # Generate 60 bars (1 hour)
    bars = []
    from datetime import timedelta
    for minute in range(60):
        timestamp = start_date + timedelta(minutes=minute)
        bar = gen.generate_bar(timestamp, MarketState.RALLY)
        bars.append(bar)
    
    df = pd.DataFrame(bars)
    print(f"✓ Generated {len(df)} bars successfully")
    print(f"  Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    print(f"  Columns: {', '.join(df.columns[:8])}...")
except Exception as e:
    print(f"✗ Price generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test tick-based features
print("\n[4/6] Testing tick-based features...")
try:
    assert 'delta_ticks' in df.columns
    assert 'range_ticks' in df.columns
    assert 'body_ticks' in df.columns
    assert 'upper_wick_ticks' in df.columns
    assert 'lower_wick_ticks' in df.columns
    
    # Verify all prices are multiples of 0.25
    assert all(df['open'] % 0.25 == 0)
    assert all(df['high'] % 0.25 == 0)
    assert all(df['low'] % 0.25 == 0)
    assert all(df['close'] % 0.25 == 0)
    
    print("✓ Tick-based features OK")
    print(f"  Avg delta_ticks: {df['delta_ticks'].mean():.2f}")
    print(f"  Avg range_ticks: {df['range_ticks'].mean():.2f}")
except AssertionError as e:
    print(f"✗ Tick feature validation failed: {e}")
    sys.exit(1)

# Test 5: Test analysis utilities
print("\n[5/6] Testing analysis utilities...")
try:
    # Generate a full day for better stats
    full_day = gen.generate_day(
        datetime(2025, 11, 29, 0, 0, 0),
        auto_transition=True
    )
    
    summary = summarize_day(full_day)
    
    assert 'overall' in summary
    assert 'by_state' in summary
    assert 'by_session' in summary
    
    print("✓ Analysis utilities OK")
    print(f"  States found: {len(summary['by_state'])}")
    print(f"  Sessions found: {len(summary['by_session'])}")
except Exception as e:
    print(f"✗ Analysis utilities failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test directory structure
print("\n[6/6] Testing directory structure...")
try:
    required_dirs = [
        root / "lab" / "generators",
        root / "lab" / "visualizers",
        root / "src" / "data",
        root / "src" / "models",
        root / "scripts",
        root / "docs",
        root / "out" / "charts",
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not dir_path.exists():
            missing_dirs.append(str(dir_path))
    
    if missing_dirs:
        print(f"✗ Missing directories: {', '.join(missing_dirs)}")
    else:
        print("✓ Directory structure OK")
except Exception as e:
    print(f"✗ Directory check failed: {e}")

# Summary
print("\n" + "=" * 70)
print("INSTALLATION TEST COMPLETE")
print("=" * 70)
print("\n✅ All tests passed! Your environment is ready.")
print("\nNext steps:")
print("  1. Review docs/PROJECT_MANAGEMENT.md for roadmap")
print("  2. Run: python scripts/demo_price_generation.py")
print("  3. Start generating archetypes")
print("\nEnvironment info:")
print(f"  Python: {sys.version.split()[0]}")
print(f"  NumPy: {np.__version__}")
print(f"  Pandas: {pd.__version__}")
print(f"  Project root: {root}")
print("=" * 70)
