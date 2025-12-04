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
