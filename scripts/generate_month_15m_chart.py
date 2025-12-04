#!/usr/bin/env python
"""
Generate one month of 1-minute synthetic data and save a 15m candlestick chart.

- Uses PriceGenerator with auto transitions
- Stitches together N calendar days
- Resamples to 15m OHLCV
- Saves chart to out/charts/month_15m.png
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.core.generator import PriceGenerator               # noqa: E402
from lab.visualizers import ChartVisualizer, ChartConfig # noqa: E402


def generate_month(start_date: datetime, num_days: int = 30) -> pd.DataFrame:
    """
    Generate num_days of 1-minute bars using PriceGenerator.generate_day.

    Returns a single DataFrame with a continuous time column.
    """
    gen = PriceGenerator(initial_price=5000.0, seed=42)

    all_days = []
    current_start = start_date

    for i in range(num_days):
        print(f"Generating day {i+1}/{num_days} starting {current_start}...")
        df_day = gen.generate_day(
            current_start,
            auto_transition=True,   # let session/DOW logic drive behavior
        )
        all_days.append(df_day)
        current_start = current_start + timedelta(days=1)

    df = pd.concat(all_days, ignore_index=True)
    df = df.sort_values("time")  # safety
    return df


def resample_to_15m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-minute OHLCV data to 15-minute candles.
    """
    df = df_1m.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    ohlc = df[["open", "high", "low", "close", "volume"]].resample("15T").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # Drop any all-NaN rows that can appear at edges
    ohlc = ohlc.dropna(subset=["open", "high", "low", "close"])

    ohlc = ohlc.reset_index()  # bring 'time' back as a column
    return ohlc


def main():
    # -----------------------------------------------------------------
    # 1. Generate a month of 1m data
    # -----------------------------------------------------------------
    START_DATE = datetime(2025, 1, 1, 0, 0, 0)  # <-- tweak if you like
    NUM_DAYS = 30                               # <-- tweak if you like

    print("=" * 60)
    print(f"Generating {NUM_DAYS} days of 1-minute synthetic data...")
    print("=" * 60)

    df_1m = generate_month(START_DATE, NUM_DAYS)
    print(f"\nTotal 1m bars: {len(df_1m)}")
    print(f"Time span: {df_1m['time'].iloc[0]}  ->  {df_1m['time'].iloc[-1]}")

    # -----------------------------------------------------------------
    # 2. Resample to 15m OHLCV
    # -----------------------------------------------------------------
    print("\nResampling to 15-minute candles...")
    df_15m = resample_to_15m(df_1m)
    print(f"Total 15m bars: {len(df_15m)}")

    # -----------------------------------------------------------------
    # 3. Create and save candlestick chart
    # -----------------------------------------------------------------
    output_dir = root / "out" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "month_15m.png"

    config = ChartConfig(
        title="Synthetic Month - 15m Candles",
        figsize=(20, 10),
        show_volume=True,
        show_state_changes=False,     # 15m has no state column
        show_session_changes=False,   # resampled data; sessions not tracked
        major_tick_interval_minutes=60,
    )

    viz = ChartVisualizer(config)
    viz.create_chart(df_15m, save_path=chart_path, show=False)

    print(f"\nChart saved to: {chart_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
