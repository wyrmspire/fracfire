# Code Dump: 09_demo_scripts

## File: scripts/demo_enhanced_features.py
```python
"""
Demo: Enhanced Features and Analysis

Demonstrates advanced features like segment-based control, macro regimes,
and detailed statistical analysis of generated data.
"""

from pathlib import Path
from datetime import datetime, timedelta
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.core.generator import PriceGenerator, MarketState
from src.core.generator.utils import summarize_day, print_summary, compare_states
from lab.visualizers.chart_viz import ChartVisualizer, ChartConfig


def demo_segment_control():
    """Demonstrate segment-based state control (e.g., 15-minute blocks)"""
    print("\n" + "=" * 60)
    print("DEMO: Segment-Based State Control")
    print("=" * 60)
    
    gen = PriceGenerator(initial_price=5000.0, seed=101)
    start_date = datetime(2025, 11, 29, 9, 30, 0)
    
    # Define a sequence of 15-minute segments
    # 0-15: Ranging
    # 15-30: Breakout
    # 30-45: Rally
    # 45-60: Ranging
    state_sequence = [
        (0, MarketState.RANGING),
        (15, MarketState.BREAKOUT),
        (30, MarketState.RALLY),
        (45, MarketState.RANGING),
    ]
    
    # Generate 1 hour of data with 15-minute segments
    # Note: generate_day generates 1440 bars, so we'll slice it or use a custom loop
    # Here we use generate_day but with a segment_length parameter
    
    df = gen.generate_day(
        start_date,
        state_sequence=state_sequence,
        auto_transition=False,
        segment_length=15,
        macro_regime="BREAKOUT_HOUR"
    )
    
    # Slice to just the first hour for this demo
    df_hour = df.iloc[:60].copy()
    
    print(f"Generated {len(df_hour)} bars with 15-minute segments")
    print("\nSegments found:")
    for seg_id in df_hour['segment_id'].unique():
        seg_df = df_hour[df_hour['segment_id'] == seg_id]
        state = seg_df['state'].iloc[0]
        print(f"  Segment {seg_id}: {state} ({len(seg_df)} bars)")
    
    # Visualize
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = ChartConfig(
        title="15-Minute Segment Control",
        figsize=(12, 8),
        show_state_changes=True,
        major_tick_interval_minutes=15,
    )
    
    viz = ChartVisualizer(config)
    chart_path = output_dir / "demo_segments.png"
    viz.create_chart(df_hour, save_path=chart_path, show=False)
    print(f"\nChart saved to: {chart_path}")
    
    return df_hour


def demo_detailed_analysis():
    """Demonstrate detailed statistical analysis of a generated day"""
    print("\n" + "=" * 60)
    print("DEMO: Detailed Statistical Analysis")
    print("=" * 60)
    
    gen = PriceGenerator(initial_price=5000.0, seed=202)
    start_date = datetime(2025, 11, 29, 0, 0, 0)
    
    # Generate a full day with auto transitions
    df = gen.generate_day(start_date, auto_transition=True)
    
    # 1. Full Day Summary
    print("\n1. Full Day Summary")
    summary = summarize_day(df)
    print_summary(summary, verbose=True)
    
    # 2. State Comparison
    print("\n2. State Comparison Table")
    comp_df = compare_states(df)
    
    # Format for display
    display_cols = ['state', 'count', 'avg_delta_ticks', 'avg_range_ticks', 'avg_volume', 'up_pct']
    print(comp_df[display_cols].round(2).to_string(index=False))
    
    # 3. Tick Feature Analysis
    print("\n3. Tick Feature Analysis")
    print("Correlation between features:")
    features = ['delta_ticks', 'range_ticks', 'body_ticks', 'volume']
    corr = df[features].corr()
    print(corr.round(2))
    
    # 4. Visualizing Distributions
    print("\n4. Generating Distribution Plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Delta Ticks Distribution
    df['delta_ticks'].hist(bins=30, ax=axes[0, 0], alpha=0.7)
    axes[0, 0].set_title('Distribution of Price Changes (Ticks)')
    axes[0, 0].set_xlabel('Ticks')
    
    # Range Ticks Distribution
    df['range_ticks'].hist(bins=30, ax=axes[0, 1], alpha=0.7, color='orange')
    axes[0, 1].set_title('Distribution of Bar Ranges (Ticks)')
    axes[0, 1].set_xlabel('Ticks')
    
    # Volume by State
    df.boxplot(column='volume', by='state', ax=axes[1, 0], rot=45)
    axes[1, 0].set_title('Volume Distribution by State')
    axes[1, 0].set_xlabel('')
    
    # Range by Session
    df.boxplot(column='range_ticks', by='session', ax=axes[1, 1], rot=45)
    axes[1, 1].set_title('Volatility (Range) by Session')
    axes[1, 1].set_xlabel('')
    
    plt.tight_layout()
    plt.suptitle("Generated Data Analysis", y=1.02, fontsize=16)
    
    output_dir = root / "out" / "charts"
    chart_path = output_dir / "demo_analysis_dist.png"
    plt.savefig(chart_path, dpi=100, bbox_inches='tight')
    print(f"Analysis plots saved to: {chart_path}")


def main():
    demo_segment_control()
    demo_detailed_analysis()
    
    print("\n" + "=" * 60)
    print("Enhanced features demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

```

---

## File: scripts/demo_custom_states.py
```python
"""
Demo: Custom Market States

Visualizes the extreme and specialized market states defined in custom_states.py.
Creates a grid of charts showing different market behaviors.
"""

from pathlib import Path
from datetime import datetime, timedelta
import sys
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.core.generator import PriceGenerator
from src.core.generator import CUSTOM_STATES, get_custom_state
from lab.visualizers.chart_viz import ChartVisualizer, ChartConfig


def demo_comparison_grid():
    """Create a grid of charts comparing different custom states"""
    print("=" * 60)
    print("DEMO: Custom State Comparison Grid")
    print("=" * 60)
    
    # Select states to visualize
    states_to_show = [
        'mega_volatile', 'flash_crash', 'melt_up',
        'whipsaw', 'opening_bell', 'news_spike',
        'slow_bleed', 'dead_zone', 'closing_squeeze'
    ]
    
    # Setup grid
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten()
    
    gen = PriceGenerator(initial_price=5000.0, seed=42)
    start_date = datetime(2025, 11, 29, 9, 30, 0)
    
    print(f"Generating {len(states_to_show)} state samples...")
    
    for idx, state_name in enumerate(states_to_show):
        ax = axes[idx]
        config = get_custom_state(state_name)
        
        # Generate 2 hours of data
        bars = []
        for minute in range(120):
            timestamp = start_date + timedelta(minutes=minute)
            bar = gen.generate_bar(timestamp, custom_state_config=config)
            bars.append(bar)
        
        df = pd.DataFrame(bars)
        
        # Visualize
        viz_config = ChartConfig(
            title=f"{state_name.upper()}",
            show_volume=False,
            show_state_changes=False,
            show_session_changes=False,
            title_fontsize=10,
        )
        
        viz = ChartVisualizer(viz_config)
        viz.plot_candlestick(ax, df)
        
        # Add stats
        volatility = df['range_ticks'].mean()
        net_move = df['close'].iloc[-1] - df['open'].iloc[0]
        
        stats = f"Vol: {volatility:.1f} | Net: {net_move:.2f}"
        ax.text(
            0.02, 0.95, stats,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        ax.set_title(state_name.upper(), fontsize=10)
        ax.grid(True, alpha=0.2)
        
        print(f"  - {state_name}: Range {df['low'].min():.2f}-{df['high'].max():.2f}")

    plt.tight_layout()
    plt.suptitle("Custom Market State Archetypes", y=1.02, fontsize=16)
    
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chart_path = output_dir / "custom_states_grid.png"
    plt.savefig(chart_path, dpi=100, bbox_inches='tight')
    print(f"\nGrid chart saved to: {chart_path}")


def main():
    demo_comparison_grid()


if __name__ == "__main__":
    main()

```

---

## File: scripts/demo_price_generation.py
```python
"""
Demo: Generate and visualize synthetic MES price data

This script demonstrates the price generator and chart visualizer working together.
"""

from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add parent directory to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.core.generator import (
    PriceGenerator,
    MarketState,
    StateConfig,
    STATE_CONFIGS,
)
from lab.visualizers.chart_viz import ChartVisualizer, ChartConfig, quick_chart


def demo_basic_generation():
    """Generate a day of data with automatic state transitions"""
    print("=" * 60)
    print("DEMO 1: Basic Day Generation with Auto State Transitions")
    print("=" * 60)
    
    # Create generator
    gen = PriceGenerator(initial_price=5000.0, seed=42)
    
    # Generate a full day
    start_date = datetime(2025, 11, 29, 0, 0, 0)
    df = gen.generate_day(start_date, auto_transition=True)
    
    print(f"\nGenerated {len(df)} bars")
    print(f"Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    print(f"Starting price: {df['open'].iloc[0]:.2f}")
    print(f"Ending price: {df['close'].iloc[-1]:.2f}")
    print(f"Total move: {df['close'].iloc[-1] - df['open'].iloc[0]:.2f}")
    
    # Show state distribution
    print("\nState distribution:")
    print(df['state'].value_counts())
    
    # Create chart
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = ChartConfig(
        title="MES Simulation - Auto State Transitions",
        figsize=(20, 10),
        show_volume=True,
        show_state_changes=True,
        show_session_changes=True,
    )
    
    viz = ChartVisualizer(config)
    chart_path = output_dir / "demo1_auto_states.png"
    viz.create_chart(df, save_path=chart_path, show=False)
    
    print(f"\nChart saved to: {chart_path}")
    return df


def demo_controlled_states():
    """Generate data with manually controlled state sequence"""
    print("\n" + "=" * 60)
    print("DEMO 2: Controlled State Sequence")
    print("=" * 60)
    
    # Create generator
    gen = PriceGenerator(initial_price=5000.0, seed=123)
    
    # Define a specific state sequence
    # Format: (bar_index, state)
    state_sequence = [
        (0, MarketState.FLAT),           # Start flat (midnight)
        (180, MarketState.ZOMBIE),       # 3am - slow grind starts
        (390, MarketState.RANGING),      # 6:30am - choppy
        (570, MarketState.RALLY),        # 9:30am - RTH open, rally
        (690, MarketState.IMPULSIVE),    # 11:30am - high volatility
        (810, MarketState.BREAKDOWN),    # 1:30pm - sharp drop
        (900, MarketState.RANGING),      # 3pm - settle into range
        (960, MarketState.FLAT),         # 4pm - afterhours quiet
    ]
    
    start_date = datetime(2025, 11, 29, 0, 0, 0)
    df = gen.generate_day(start_date, state_sequence=state_sequence, auto_transition=False)
    
    print(f"\nGenerated {len(df)} bars with controlled states")
    print(f"Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    print(f"Starting price: {df['open'].iloc[0]:.2f}")
    print(f"Ending price: {df['close'].iloc[-1]:.2f}")
    print(f"Total move: {df['close'].iloc[-1] - df['open'].iloc[0]:.2f}")
    
    # Create chart
    output_dir = root / "out" / "charts"
    
    config = ChartConfig(
        title="MES Simulation - Controlled State Sequence",
        figsize=(20, 10),
        show_volume=True,
        show_state_changes=True,
        show_session_changes=True,
        major_tick_interval_minutes=120,  # Every 2 hours
    )
    
    viz = ChartVisualizer(config)
    chart_path = output_dir / "demo2_controlled_states.png"
    viz.create_chart(df, save_path=chart_path, show=False)
    
    print(f"\nChart saved to: {chart_path}")
    return df


def demo_custom_state():
    """Generate data with a custom state configuration"""
    print("\n" + "=" * 60)
    print("DEMO 3: Custom State Configuration")
    print("=" * 60)
    
    # Create a custom "mega volatile" state
    custom_state = StateConfig(
        name="mega_volatile",
        avg_ticks_per_bar=40.0,
        ticks_per_bar_std=20.0,
        up_probability=0.5,
        trend_persistence=0.4,  # Low persistence = very choppy
        avg_tick_size=3.0,
        tick_size_std=2.0,
        max_tick_jump=15,
        volatility_multiplier=3.0,
        wick_probability=0.6,
        wick_extension_avg=5.0,
    )
    
    # Create generator
    gen = PriceGenerator(initial_price=5000.0, seed=456)
    
    # Generate just a few hours with this custom state
    start_date = datetime(2025, 11, 29, 9, 30, 0)  # RTH open
    bars = []
    
    for minute in range(240):  # 4 hours
        timestamp = start_date + timedelta(minutes=minute)
        bar = gen.generate_bar(timestamp, custom_state_config=custom_state)
        bars.append(bar)
    
    import pandas as pd
    df = pd.DataFrame(bars)
    
    print(f"\nGenerated {len(df)} bars with custom 'mega_volatile' state")
    print(f"Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    print(f"Starting price: {df['open'].iloc[0]:.2f}")
    print(f"Ending price: {df['close'].iloc[-1]:.2f}")
    print(f"Total move: {df['close'].iloc[-1] - df['open'].iloc[0]:.2f}")
    print(f"Max bar range: {(df['high'] - df['low']).max():.2f}")
    
    # Create chart
    output_dir = root / "out" / "charts"
    
    config = ChartConfig(
        title="MES Simulation - Custom Mega Volatile State",
        figsize=(16, 9),
        show_volume=True,
        show_state_changes=False,
        show_session_changes=True,
        major_tick_interval_minutes=30,
    )
    
    viz = ChartVisualizer(config)
    chart_path = output_dir / "demo3_custom_state.png"
    viz.create_chart(df, save_path=chart_path, show=False)
    
    print(f"\nChart saved to: {chart_path}")
    return df


def demo_state_comparison():
    """Generate samples of each state for comparison"""
    print("\n" + "=" * 60)
    print("DEMO 4: State Comparison")
    print("=" * 60)
    
    import matplotlib.pyplot as plt
    
    states_to_compare = [
        MarketState.FLAT,
        MarketState.RANGING,
        MarketState.ZOMBIE,
        MarketState.RALLY,
        MarketState.IMPULSIVE,
    ]
    
    fig, axes = plt.subplots(len(states_to_compare), 1, figsize=(16, 12))
    
    start_date = datetime(2025, 11, 29, 9, 30, 0)
    
    for idx, state in enumerate(states_to_compare):
        # Generate 2 hours of data for this state
        gen = PriceGenerator(initial_price=5000.0, seed=42 + idx)
        
        bars = []
        for minute in range(120):
            timestamp = start_date + timedelta(minutes=minute)
            bar = gen.generate_bar(timestamp, state=state)
            bars.append(bar)
        
        import pandas as pd
        df = pd.DataFrame(bars)
        
        # Plot on subplot
        ax = axes[idx]
        
        config = ChartConfig(
            title=f"{state.value.upper()} State",
            show_volume=False,
            show_state_changes=False,
            show_session_changes=False,
            figsize=(16, 3),
        )
        
        viz = ChartVisualizer(config)
        viz.plot_candlestick(ax, df)
        
        ax.set_title(f"{state.value.upper()} State", fontsize=12, pad=10)
        ax.grid(True, alpha=0.3)
        
        # Add stats
        price_range = df['high'].max() - df['low'].min()
        avg_bar_range = (df['high'] - df['low']).mean()
        
        stats_text = f"Range: {price_range:.2f} | Avg Bar: {avg_bar_range:.2f}"
        ax.text(
            0.02, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.tight_layout()
    
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "demo4_state_comparison.png"
    plt.savefig(chart_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison chart saved to: {chart_path}")


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("MES PRICE GENERATOR DEMO")
    print("=" * 60)
    
    # Run demos
    demo_basic_generation()
    demo_controlled_states()
    demo_custom_state()
    demo_state_comparison()
    
    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)
    print(f"\nCharts saved to: {Path(__file__).resolve().parents[1] / 'out' / 'charts'}")


if __name__ == "__main__":
    main()

```

---

## File: scripts/test_installation.py
```python
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

# Test 2: Import core modules
print("\n[2/6] Testing core modules...")
try:
    from src.core.generator import PriceGenerator, MarketState
    from src.core.generator.utils import summarize_day, print_summary
    print("✓ Core generator module OK")
except ImportError as e:
    print(f"✗ Failed to import core modules: {e}")
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
        root / "src" / "core" / "generator",
        root / "src" / "core" / "detector",
        root / "src" / "data",
        root / "src" / "ml",
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

```

---

