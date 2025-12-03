"""
Compare Real vs Generator-Based Synthetic Data

- Always generate synthetic at 1m resolution using PriceGenerator
- Aggregate both real and synthetic to 5m and 15m
- Produce side-by-side candlestick charts for:
    * 1-week window
    * 3-month window

Charts are saved under out/charts/.
"""

import sys
from pathlib import Path
from datetime import timedelta
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from lab.generators.price_generator import PriceGenerator, PhysicsConfig
from src.data.loader import RealDataLoader


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df.resample(timeframe).agg(ohlc).dropna()


def generate_synthetic_matching(real_1m: pd.DataFrame, seed: int, physics_overrides: dict = None) -> pd.DataFrame:
    """Generate synthetic bars aligned to the timestamps of a real 1m DataFrame.

    This ensures the synthetic series has the same time index (including trading hours
    and missing minutes) so that resampling to daily/weekly windows produces directly
    comparable candles.
    """
    if real_1m.empty:
        raise ValueError("Real 1m window is empty")

    # Start from the global PhysicsConfig defaults, then apply any CLI overrides.
    defaults = PhysicsConfig().__dict__.copy()

    if physics_overrides:
        for k, v in physics_overrides.items():
            if v is not None and k in defaults:
                defaults[k] = v

    physics = PhysicsConfig(**defaults)

    gen = PriceGenerator(
        initial_price=float(real_1m["open"].iloc[0]),
        seed=seed,
        physics_config=physics,
    )

    bars = []
    # Iterate over the real index and generate bars at the exact same timestamps
    for ts in real_1m.index:
        bar = gen.generate_bar(ts)
        bars.append(bar)

    synth = pd.DataFrame(bars)
    synth.set_index("time", inplace=True)
    synth.sort_index(inplace=True)
    return synth[["open", "high", "low", "close", "volume"]]


def plot_candlesticks(ax, df, title):
    """Plot candlesticks on the given axes"""
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
        min_diff = (df.index[1] - df.index[0]).total_seconds()
        width = min_diff / 86400 * 0.8
    else:
        width = 0.0005
        
    # Matplotlib bar chart for bodies
    bottoms = df[['open', 'close']].min(axis=1)
    heights = (df['close'] - df['open']).abs()
    
    # Ensure minimum height for dojis so they are visible
    heights = heights.replace(0, 0.01)
    
    ax.bar(df.index, heights, bottom=bottoms, width=width, color=colors, align='center')
    
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.set_ylabel("Price")


def compute_wick_stats(df: pd.DataFrame) -> dict:
    """Compute simple wick/body statistics for a resampled OHLC frame.

    This helps diagnose cases where bars are mostly wicks.
    """
    body = (df["close"] - df["open"]).abs()
    full = df["high"] - df["low"]
    # Avoid division by zero
    full_safe = full.replace(0, pd.NA)
    wick_frac = 1.0 - (body / full_safe)
    wick_frac = wick_frac.fillna(0.0)

    return {
        "n_bars": len(df),
        "avg_body": float(body.mean()),
        "avg_range": float(full.mean()),
        "avg_wick_frac": float(wick_frac.mean()),
        "pct_wick_gt_0_6": float((wick_frac > 0.6).mean()),
    }


def plot_pair(real_df: pd.DataFrame, synth_df: pd.DataFrame, title: str, filename: str) -> None:
    print(f"Plotting {filename}: Real rows={len(real_df)}, Synth rows={len(synth_df)}")
    
    # DEBUG: Print first few timestamps to verify aggregation
    if len(real_df) > 2:
        print(f"  Real index freq check: {real_df.index[0]} -> {real_df.index[1]} -> {real_df.index[2]}")
    if len(synth_df) > 2:
        print(f"  Synth index freq check: {synth_df.index[0]} -> {synth_df.index[1]} -> {synth_df.index[2]}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    real_range = real_df["high"].max() - real_df["low"].min()
    synth_range = synth_df["high"].max() - synth_df["low"].min()

    plot_candlesticks(ax1, real_df, f"REAL - {title}\nRange: {real_range:.2f}")
    plot_candlesticks(ax2, synth_df, f"SYNTH - {title}\nRange: {synth_range:.2f}")

    plt.tight_layout()
    # Treat `filename` as a full path. If it's relative, make parent dirs.
    out_path = Path(filename)
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def pick_real_windows(real_1m: pd.DataFrame, week_days: int = 7, quarter_days: int = 90):
    """Pick one ~`week_days` and one ~`quarter_days` continuous slice from real 1m.

    Avoids April 2025 (black swan event).
    """
    week_bars = week_days * 24 * 60
    quarter_bars = quarter_days * 24 * 60

    if len(real_1m) <= max(week_bars, quarter_bars):
        raise ValueError("Real data does not have enough 1m bars for requested windows")

    # Filter out April 2025 data (black swan event)
    # Only use data before 2025-04-01 or after 2025-04-30
    april_start = pd.Timestamp('2025-04-01', tz='UTC')
    april_end = pd.Timestamp('2025-05-01', tz='UTC')
    
    before_april = real_1m[real_1m.index < april_start]
    after_april = real_1m[real_1m.index >= april_end]
    
    # Try to get 3-month window from after April first
    if len(after_april) >= quarter_bars:
        start_q = 0
        real_q = after_april.iloc[start_q : start_q + quarter_bars].copy()
    elif len(before_april) >= quarter_bars:
        # Fall back to before April
        start_q = max(0, len(before_april) - quarter_bars)
        real_q = before_april.iloc[start_q : start_q + quarter_bars].copy()
    else:
        # Last resort: use all data (includes April)
        start_q = 0
        real_q = real_1m.iloc[start_q : start_q + quarter_bars].copy()

    # 1-week window: choose from after April if possible
    if len(after_april) >= week_bars:
        max_start_week = len(after_april) - week_bars
        start_w = np.random.randint(0, max_start_week)
        real_w = after_april.iloc[start_w : start_w + week_bars].copy()
    elif len(before_april) >= week_bars:
        max_start_week = len(before_april) - week_bars
        start_w = np.random.randint(0, max_start_week)
        real_w = before_april.iloc[start_w : start_w + week_bars].copy()
    else:
        # Last resort
        start_w = 0
        real_w = real_1m.iloc[start_w : start_w + week_bars].copy()

    return real_w, real_q


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Compare real continuous contract vs generated synthetic")
    parser.add_argument('--seed-week', type=int, default=42, help='Seed for weekly synthetic')
    parser.add_argument('--seed-quarter', type=int, default=123, help='Seed for quarter synthetic')
    parser.add_argument('--out-dir', type=str, default=str(root / 'out' / 'charts'), help='Output directory')
    parser.add_argument('--show', action='store_true', help='Show charts interactively')
    parser.add_argument('--week-days', type=int, default=7, help='Number of days to use for the smaller (week) window')
    parser.add_argument('--quarter-days', type=int, default=90, help='Number of days to use for the larger (quarter) window')
    # Default: include 4H/1D with 4min/15min internals for diagnosis
    parser.add_argument('--week-timeframes', type=str, default='4H,4min', help='Comma-separated resample timeframes for week charts (e.g., "4H,4min")')
    parser.add_argument('--quarter-timeframes', type=str, default='1D,15min', help='Comma-separated resample timeframes for quarter charts (e.g., "1D,15min")')
    parser.add_argument('--week-start', type=str, default=None, help='Optional ISO date (YYYY-MM-DD) to anchor the week window (uses data around this date)')

    # Physics knobs (only a subset exposed for quick dialing)
    parser.add_argument('--base-volatility', type=float, default=None)
    parser.add_argument('--avg-ticks-per-bar', type=float, default=None)
    parser.add_argument('--daily-range-mean', type=float, default=None)
    parser.add_argument('--daily-range-std', type=float, default=None)
    parser.add_argument('--runner-prob', type=float, default=None)
    parser.add_argument('--runner-target-mult', type=float, default=None)
    parser.add_argument('--macro-gravity-threshold', type=float, default=None)
    parser.add_argument('--macro-gravity-strength', type=float, default=None)
    parser.add_argument('--wick-probability', type=float, default=None)
    parser.add_argument('--wick-extension-avg', type=float, default=None)

    args = parser.parse_args(argv)

    print("=" * 60)
    print("COMPARE REAL VS GENERATOR (AGG TO 5M/15M)")
    print("=" * 60)

    loader = RealDataLoader()
    real_path = root / "src" / "data" / "continuous_contract.json"
    real_1m = loader.load_json(real_path)

    # Basic sanity: use only OHLCV columns for aggregation
    real_1m_ohlcv = real_1m[["open", "high", "low", "close", "volume"]]

    # Optionally anchor the week window around a specific calendar date
    if args.week_start:
        anchor = pd.Timestamp(args.week_start, tz='UTC')
        half_span = pd.Timedelta(days=args.week_days // 2)
        start = anchor - half_span
        end = start + pd.Timedelta(days=args.week_days)
        real_slice = real_1m_ohlcv.loc[start:end]
        if len(real_slice) == 0:
            raise ValueError(f"No real data found around week_start={args.week_start}")
        real_week_1m = real_slice.copy()
        # Quarter window still selected by the generic helper
        _, real_quarter_1m = pick_real_windows(real_1m_ohlcv, week_days=args.week_days, quarter_days=args.quarter_days)
    else:
        real_week_1m, real_quarter_1m = pick_real_windows(real_1m_ohlcv, week_days=args.week_days, quarter_days=args.quarter_days)

    physics_overrides = dict(
        base_volatility=args.base_volatility,
        avg_ticks_per_bar=args.avg_ticks_per_bar,
        daily_range_mean=args.daily_range_mean,
        daily_range_std=args.daily_range_std,
        runner_prob=args.runner_prob,
        runner_target_mult=args.runner_target_mult,
        macro_gravity_threshold=args.macro_gravity_threshold,
        macro_gravity_strength=args.macro_gravity_strength,
        wick_probability=args.wick_probability,
        wick_extension_avg=args.wick_extension_avg,
    )

    synth_week_1m = generate_synthetic_matching(real_week_1m, args.seed_week, physics_overrides=physics_overrides)
    synth_quarter_1m = generate_synthetic_matching(real_quarter_1m, args.seed_quarter, physics_overrides=physics_overrides)

    # Aggregate to 5m and 15m; we do NOT plot 1m
    real_week_5m = resample_ohlcv(real_week_1m, "5min")
    real_week_15m = resample_ohlcv(real_week_1m, "15min")
    synth_week_5m = resample_ohlcv(synth_week_1m, "5min")
    synth_week_15m = resample_ohlcv(synth_week_1m, "15min")

    # For the 3-month window, aggregate to daily so candles are clearly visible
    real_quarter_daily = resample_ohlcv(real_quarter_1m, "1D")
    synth_quarter_daily = resample_ohlcv(synth_quarter_1m, "1D")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Week charts: support multiple timeframes supplied by the user
    week_timeframes = [t.strip() for t in args.week_timeframes.split(',') if t.strip()]
    for tf in week_timeframes:
        real_week_tf = resample_ohlcv(real_week_1m, tf)
        synth_week_tf = resample_ohlcv(synth_week_1m, tf)
        stats_real = compute_wick_stats(real_week_tf)
        stats_synth = compute_wick_stats(synth_week_tf)
        print(f"Week {tf} stats REAL: {stats_real}")
        print(f"Week {tf} stats SYNTH: {stats_synth}")
        filename = out_dir / f"gen_compare_week_{tf.replace('%','').replace(':','').replace('/','_')}.png"
        plot_pair(real_week_tf, synth_week_tf, title=f"1-Week {tf}", filename=str(filename))

    # Quarter charts: support multiple timeframes
    quarter_timeframes = [t.strip() for t in args.quarter_timeframes.split(',') if t.strip()]
    for tf in quarter_timeframes:
        real_q_tf = resample_ohlcv(real_quarter_1m, tf)
        synth_q_tf = resample_ohlcv(synth_quarter_1m, tf)
        stats_real = compute_wick_stats(real_q_tf)
        stats_synth = compute_wick_stats(synth_q_tf)
        print(f"Quarter {tf} stats REAL: {stats_real}")
        print(f"Quarter {tf} stats SYNTH: {stats_synth}")
        filename = out_dir / f"gen_compare_quarter_{tf.replace('%','').replace(':','').replace('/','_')}.png"
        plot_pair(real_q_tf, synth_q_tf, title=f"Quarter {tf}", filename=str(filename))

    print("Done.")


if __name__ == "__main__":
    main()
