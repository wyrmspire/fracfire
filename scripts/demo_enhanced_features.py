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
