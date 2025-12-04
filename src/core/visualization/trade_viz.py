"""
Visualization utilities for trade plotting with shaded zones.

This module provides consistent styling and helper functions for visualizing
trades with profit/loss zones across different chart types.
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

# Standard color palette for trade visualization
COLORS = {
    # Profit zone (green)
    "profit_fill": "#26a69a",
    "profit_line": "#2ecc71",
    "profit_fill_alpha": 0.15,
    
    # Stop loss zone (red)
    "stop_fill": "#ef5350",
    "stop_line": "#e74c3c",
    "stop_fill_alpha": 0.15,
    
    # Entry markers
    "entry_win": "#27ae60",
    "entry_loss": "#c0392b",
    
    # Candle colors
    "candle_up": "#26a69a",
    "candle_down": "#ef5350",
    "candle_wick": "#666",
}

# Default fallback duration for open trades (minutes)
DEFAULT_TRADE_DURATION_MINUTES = 60


def add_trade_zones_matplotlib(
    ax: plt.Axes,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    entry_price: float,
    stop_price: float,
    target_price: float,
    zorder: int = 1
) -> None:
    """
    Add shaded profit and stop loss zones to a matplotlib axes.
    
    Args:
        ax: Matplotlib axes to draw on
        entry_time: Trade entry timestamp
        exit_time: Trade exit timestamp
        entry_price: Entry price level
        stop_price: Stop loss price level
        target_price: Take profit price level
        zorder: Z-order for layering (lower = background)
    """
    # Stop loss zone (between entry and stop)
    stop_zone_bottom = min(entry_price, stop_price)
    stop_zone_top = max(entry_price, stop_price)
    ax.fill_between(
        [entry_time, exit_time],
        stop_zone_bottom,
        stop_zone_top,
        color=COLORS["stop_fill"],
        alpha=COLORS["stop_fill_alpha"],
        zorder=zorder,
        label="_nolegend_"
    )
    
    # Profit zone (between entry and target)
    profit_zone_bottom = min(entry_price, target_price)
    profit_zone_top = max(entry_price, target_price)
    ax.fill_between(
        [entry_time, exit_time],
        profit_zone_bottom,
        profit_zone_top,
        color=COLORS["profit_fill"],
        alpha=COLORS["profit_fill_alpha"],
        zorder=zorder,
        label="_nolegend_"
    )


def add_trade_lines_matplotlib(
    ax: plt.Axes,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    stop_price: float,
    target_price: float,
    zorder: int = 3
) -> None:
    """
    Add stop loss and take profit lines to a matplotlib axes.
    
    Args:
        ax: Matplotlib axes to draw on
        entry_time: Trade entry timestamp
        exit_time: Trade exit timestamp
        stop_price: Stop loss price level
        target_price: Take profit price level
        zorder: Z-order for layering
    """
    # Stop loss line
    ax.hlines(
        stop_price,
        xmin=entry_time,
        xmax=exit_time,
        colors=COLORS["stop_line"],
        linestyles="dashed",
        linewidth=1.5,
        zorder=zorder
    )
    
    # Take profit line
    ax.hlines(
        target_price,
        xmin=entry_time,
        xmax=exit_time,
        colors=COLORS["profit_line"],
        linestyles="dashed",
        linewidth=1.5,
        zorder=zorder
    )


def get_exit_time_or_fallback(
    entry_time: pd.Timestamp,
    exit_time_value,
    fallback_minutes: int = DEFAULT_TRADE_DURATION_MINUTES
) -> pd.Timestamp:
    """
    Get exit time or fallback to estimated duration if not available.
    
    Args:
        entry_time: Trade entry timestamp
        exit_time_value: Exit time value (may be None, NaN, or valid timestamp)
        fallback_minutes: Fallback duration in minutes if exit_time not available
        
    Returns:
        Valid exit timestamp
    """
    if exit_time_value is None:
        return entry_time + pd.Timedelta(minutes=fallback_minutes)
    
    try:
        exit_ts = pd.Timestamp(exit_time_value)
        if pd.isna(exit_ts):
            return entry_time + pd.Timedelta(minutes=fallback_minutes)
        return exit_ts
    except (ValueError, TypeError):
        return entry_time + pd.Timedelta(minutes=fallback_minutes)


def get_entry_marker_color(r_multiple: float, is_open: bool = False) -> str:
    """
    Get marker color based on trade outcome.
    
    Args:
        r_multiple: R-multiple (profit/loss ratio)
        is_open: Whether trade is still open
        
    Returns:
        Color string
    """
    if is_open:
        return "#f1c40f"  # Yellow for open trades
    return COLORS["entry_win"] if r_multiple > 0 else COLORS["entry_loss"]
