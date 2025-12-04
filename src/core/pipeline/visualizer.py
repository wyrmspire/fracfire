"""
Visualization Module

Handles plotting of scenario results (charts with trade overlays).
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import List, Optional
from datetime import datetime

from src.core.pipeline.scenario_runner import ScenarioResult
from src.core.detector.library import SetupOutcome

COLOR_UP = "#2ecc71"
COLOR_DN = "#e74c3c"
COLOR_BODY = "#34495e"
COLOR_ENTRY = {
    "win": "#27ae60",
    "loss": "#c0392b",
    "open": "#f1c40f",
}

def plot_candles(ax: plt.Axes, df: pd.DataFrame):
    """Plot candlesticks on the given axes using ax.bar for bodies."""
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('time') if 'time' in df.columns else df

    # Define colors
    up_color = '#26a69a'   # Green
    down_color = '#ef5350' # Red
    wick_color = '#666666' # Gray
    
    # Calculate colors for each bar
    colors = [up_color if c >= o else down_color for c, o in zip(df['close'], df['open'])]
    
    # Plot wicks
    ax.vlines(df.index, df['low'], df['high'], color=wick_color, linewidth=1, alpha=0.6)
    
    # Plot bodies
    if len(df) > 1:
        # Calculate width based on minimum time difference
        min_diff = (df.index[1] - df.index[0]).total_seconds()
        # Width in days (matplotlib units)
        width = min_diff / 86400 * 0.8
    else:
        width = 0.0005
        
    bottoms = df[['open', 'close']].min(axis=1)
    heights = (df['close'] - df['open']).abs()
    
    # Ensure minimum height for dojis
    heights = heights.replace(0, 0.01)
    
    ax.bar(df.index, heights, bottom=bottoms, width=width, color=colors, align='center')

def overlay_trades(ax: plt.Axes, outcomes: List[SetupOutcome]):
    """Overlay trade entries and outcomes."""
    for outcome in outcomes:
        entry = outcome.entry
        ts = entry.time
        price = entry.entry_price
        
        status = "win" if outcome.hit_target else ("loss" if outcome.hit_stop else "open")
        color = COLOR_ENTRY.get(status, "blue")
        
        # Plot entry marker
        ax.scatter([ts], [price], color=color, s=60, zorder=5)
        
        # Plot stop/target lines
        # We can plot a short line segment
        left = ts
        right = outcome.exit_time
        
        if right > left:
            # Stop line
            ax.hlines(entry.stop_price, xmin=left, xmax=right, colors="#e74c3c", linestyles="dashed", linewidth=1)
            # Target line
            ax.hlines(entry.target_price, xmin=left, xmax=right, colors="#2ecc71", linestyles="dashed", linewidth=1)
            
            # Exit marker
            exit_price = entry.target_price if outcome.hit_target else (entry.stop_price if outcome.hit_stop else None)
            if exit_price:
                 ax.scatter([right], [exit_price], color=color, marker="x", s=40, zorder=5)

        # Annotation
        label = f"{entry.kind}\nR={outcome.r_multiple:.2f}"
        ax.text(ts, price, label, fontsize=8, color="#2c3e50", verticalalignment='bottom')

def plot_scenario(
    result: ScenarioResult,
    save_dir: str,
    filename_prefix: str = "scenario",
    use_5m: bool = True
):
    """
    Plot the scenario results and save to file.
    """
    df = result.df_5m if use_5m else result.df_1m
    if df.empty:
        print("Warning: No data to plot.")
        return

    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    plot_candles(ax, df)
    overlay_trades(ax, result.outcomes)
    
    from matplotlib.dates import DateFormatter

    title = f"Scenario: {filename_prefix} | Source: {result.metadata.get('source', 'unknown')} | Trades: {len(result.outcomes)}"
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    fig.autofmt_xdate()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    
    fig.savefig(filepath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {filepath}")
