# Code Dump: 08_runner_scripts

## File: scripts/run_orb_trade_runner.py
```python
"""ORB trade runner: keep generating scenarios until an ORB trade is found.

This is designed for an agent to call once. The script will:
- Loop over synthetic days using the MES price generator.
- Aggregate to 5m bars.
- Run the ORB setup detector and evaluator.
- Optionally jiggle ORB knobs in a simple way if no trades are found.
- Stop once at least one setup is detected, then dump:
  - 1m synthetic data for the relevant window (CSV)
  - 5m data (CSV)
  - JSON summary of entries, outcomes, and last miss diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.generator import PhysicsConfig, PriceGenerator
from src.core.detector.library import (
    ORBConfig,
    summarize_orb_day_1m,
)
from src.core.detector import IndicatorConfig, add_1m_indicators, DecisionAnchorConfig
from src.core.detector import SetupEntry, SetupOutcome
from src.core.detector import compute_trade_features, BadTradeConfig, inject_bad_trade_variants


def generate_synth_1m_days(days: int, seed: int) -> pd.DataFrame:
    cfg = PhysicsConfig()
    gen = PriceGenerator(physics_config=cfg, seed=seed)
    all_days = []
    current = pd.Timestamp.utcnow().normalize()

    for i in range(days):
        day_date = current - pd.Timedelta(days=(days - 1 - i))
        df_day = gen.generate_day(day_date)
        all_days.append(df_day)

    df = pd.concat(all_days).sort_values("time")
    df = df.set_index("time")
    return df


def resample_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df_1m.resample("5min").agg(ohlc).dropna()


def orb_trade_runner(
    max_iterations: int,
    base_seed: int,
    days_per_iter: int,
    orb_cfg: ORBConfig,
) -> Dict[str, Any]:
    """Loop generator + ORB detector until we find trades or hit max_iterations.

    Returns a dict structure suitable for JSON serialization with keys:
    - "found": bool
    - "iteration": int
    - "seed": int
    - "entries": [...]
    - "outcomes": [...]
    - "miss": {...}
    - "summary": brief text summary
    """

    last_miss = None
    for i in range(max_iterations):
        seed = base_seed + i
        df_1m = generate_synth_1m_days(days_per_iter, seed)
        df_5m = resample_5m(df_1m)
        df_1m_ind = add_1m_indicators(df_1m, IndicatorConfig())

        # Use 1m-based detection with 5-min decision anchors
        entries, outcomes, miss = summarize_orb_day_1m(df_1m_ind, orb_cfg, DecisionAnchorConfig())
        last_miss = miss

        if entries:
            # For now we just take all entries in this sample; the agent can pick.
            # Compute features for each outcome
            features = []
            try:
                # Convert outcomes to SetupOutcome-like objects if needed
                for o in outcomes:
                    # Build a generic SetupEntry view
                    e = o.entry
                    entry = SetupEntry(
                        time=e.time,
                        direction=e.direction,
                        kind="orb",
                        entry_price=e.entry_price,
                        stop_price=e.stop_price,
                        target_price=e.target_price,
                        context={
                            "or_high": e.or_high,
                            "or_low": e.or_low,
                            "or_start": str(e.or_start),
                            "or_end": str(e.or_end),
                        },
                    )
                    so = SetupOutcome(
                        entry=entry,
                        hit_target=o.hit_target,
                        hit_stop=o.hit_stop,
                        exit_time=o.exit_time,
                        r_multiple=o.r_multiple,
                        mfe=o.mfe,
                        mae=o.mae,
                    )
                    features.append(compute_trade_features(so, df_1m_ind))
            except Exception:
                features = []

            return {
                "found": True,
                "iteration": i,
                "seed": seed,
                "entries": [asdict(e) for e in entries],
                "outcomes": [asdict(o) for o in outcomes],
                "miss": asdict(miss),
                "df_1m": df_1m,
                "df_5m": df_5m,
                "features": features,
                "summary": f"found {len(entries)} entries on iteration {i} seed {seed}",
            }

        # Simple auto-relax: if drive too weak, lower min_drive_rr slightly
        if miss.up_rr < orb_cfg.min_drive_rr and miss.down_rr < orb_cfg.min_drive_rr:
            orb_cfg.min_drive_rr = max(0.1, orb_cfg.min_drive_rr * 0.9)
        # If counter too big, relax that too
        if miss.down_rr > orb_cfg.max_counter_rr or miss.up_rr > orb_cfg.max_counter_rr:
            orb_cfg.max_counter_rr = min(1.0, orb_cfg.max_counter_rr * 1.1)

    # No trade found
    # No trade found
    return {
        "found": False,
        "iteration": max_iterations,
        "seed": base_seed + max_iterations - 1,
        "entries": [],
        "outcomes": [],
        "miss": asdict(last_miss) if last_miss is not None else None,
        "df_1m": df_1m,
        "df_5m": df_5m,
        "features": [],
        "summary": "no entries found within max_iterations",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ORB trade runner over synthetic MES data")
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--days-per-iter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=str, default="out/orb_runner")

    # ORB knobs the agent can override
    parser.add_argument("--or-minutes", type=int, default=30)
    parser.add_argument("--buffer-ticks", type=float, default=2.0)
    parser.add_argument("--tick-size", type=float, default=0.25)
    parser.add_argument("--min-drive-rr", type=float, default=0.5)
    parser.add_argument("--max-counter-rr", type=float, default=0.3)
    parser.add_argument("--target-rr", type=float, default=2.0)
    parser.add_argument("--max-hold-bars", type=int, default=48)
    # Bad trade injection knobs
    parser.add_argument("--inject-hesitation", action="store_true", help="Inject hesitation variants")
    parser.add_argument("--hesitation-minutes", type=int, default=5)
    parser.add_argument("--inject-chase", action="store_true", help="Inject chase variants")
    parser.add_argument("--chase-window-minutes", type=int, default=15)
    parser.add_argument("--max-variants-per-trade", type=int, default=2)

    args = parser.parse_args()

    orb_cfg = ORBConfig(
        or_minutes=args.or_minutes,
        buffer_ticks=args.buffer_ticks,
        tick_size=args.tick_size,
        min_drive_rr=args.min_drive_rr,
        max_counter_rr=args.max_counter_rr,
        target_rr=args.target_rr,
        max_hold_bars=args.max_hold_bars,
    )

    result = orb_trade_runner(
        max_iterations=args.max_iterations,
        base_seed=args.seed,
        days_per_iter=args.days_per_iter,
        orb_cfg=orb_cfg,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Save data
    df_1m: pd.DataFrame = result.pop("df_1m")  # type: ignore
    df_5m: pd.DataFrame = result.pop("df_5m")  # type: ignore
    df_1m.to_csv(os.path.join(args.out_dir, "synthetic_1m.csv"))
    df_5m.to_csv(os.path.join(args.out_dir, "synthetic_5m.csv"))

    # Inject bad trade variants if requested
    feats = result.get("features", [])
    outcomes_dicts = result.get("outcomes", [])
    # Reconstruct SetupOutcome-like objects minimally for variants (use original entries)
    outcomes_for_variants = []
    try:
        from dataclasses import asdict
        for o in outcomes_dicts:
            # Heuristic rebuild; runner already had full objects internally
            pass
    except Exception:
        outcomes_for_variants = []

    if args.inject_hesitation or args.inject_chase:
        df_1m = result.get("df_1m")
        if isinstance(df_1m, pd.DataFrame):
            # We have to rebuild SetupOutcome objects from stored entries/outcomes in summary
            rebuilt: list[SetupOutcome] = []
            for e_dict, o_dict in zip(result.get("entries", []), result.get("outcomes", [])):
                entry = SetupEntry(
                    time=pd.Timestamp(e_dict["time"]),
                    direction=e_dict["direction"],
                    kind="orb",
                    entry_price=e_dict["entry_price"],
                    stop_price=e_dict["stop_price"],
                    target_price=e_dict["target_price"],
                    context={"or_high": e_dict["or_high"], "or_low": e_dict["or_low"]},
                )
                out = SetupOutcome(
                    entry=entry,
                    hit_target=o_dict["hit_target"],
                    hit_stop=o_dict["hit_stop"],
                    exit_time=pd.Timestamp(o_dict["exit_time"]),
                    r_multiple=float(o_dict["r_multiple"]),
                    mfe=float(o_dict["mfe"]),
                    mae=float(o_dict["mae"]),
                )
                rebuilt.append(out)

            bad_cfg = BadTradeConfig(
                enable_hesitation=args.inject_hesitation,
                hesitation_minutes=args.hesitation_minutes,
                enable_chase=args.inject_chase,
                chase_window_minutes=args.chase_window_minutes,
                max_variants_per_trade=args.max_variants_per_trade,
            )
            variants = inject_bad_trade_variants(df_1m, rebuilt, bad_cfg)
            # Append variant features
            for vo in variants:
                feats.append(compute_trade_features(vo, df_1m))

    # Save JSON summary
    json_summary: Dict[str, Any] = {
        k: v for k, v in result.items() if not isinstance(v, (pd.DataFrame,))
    }
    json_summary["features"] = feats
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, default=str)

    # Save features if present
    feats = json_summary.get("features", [])
    if feats:
        pd.DataFrame(feats).to_csv(os.path.join(args.out_dir, "trades_features.csv"), index=False)

    print(json_summary["summary"])


if __name__ == "__main__":
    main()

```

---

## File: scripts/plot_trade_setups.py
```python
"""Plot 1m candlesticks with trade overlays.

Reads a scenario folder containing synthetic_1m.csv and summary.json,
then saves trade_plot.png with entries, stops, targets, and outcomes.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt

COLOR_UP = "#2ecc71"
COLOR_DN = "#e74c3c"
COLOR_BODY = "#34495e"
COLOR_ENTRY = {
    "win": "#27ae60",
    "loss": "#c0392b",
    "open": "#f1c40f",
}


def load_scenario(data_dir: str) -> tuple[pd.DataFrame, Dict[str, Any]]:
    df_1m = pd.read_csv(os.path.join(data_dir, "synthetic_1m.csv"))
    df_1m["time"] = pd.to_datetime(df_1m["time"]) if "time" in df_1m.columns else pd.to_datetime(df_1m.iloc[:, 0])
    df_1m = df_1m.set_index("time")
    with open(os.path.join(data_dir, "summary.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)
    return df_1m, summary


def plot_candles(ax: plt.Axes, df: pd.DataFrame) -> None:
    # Minimal candlestick: lines for high-low, rectangles for body
    for ts, row in df.iterrows():
        o = float(row["open"]) if "open" in row else float(row["close"])  # fallback
        c = float(row["close"]) if "close" in row else o
        h = float(row.get("high", max(o, c)))
        l = float(row.get("low", min(o, c)))
        color = COLOR_UP if c >= o else COLOR_DN
        # wick
        ax.plot([ts, ts], [l, h], color=color, linewidth=1)
        # body
        ax.plot([ts, ts], [o, c], color=COLOR_BODY, linewidth=4)


def overlay_trades(ax: plt.Axes, df: pd.DataFrame, entries: List[Dict[str, Any]], outcomes: List[Dict[str, Any]]) -> None:
    # Map entry times to outcomes
    outcome_by_time: Dict[pd.Timestamp, Dict[str, Any]] = {}
    for o in outcomes:
        outcome_by_time[pd.Timestamp(o["entry"]["time"])] = o

    for e in entries:
        ts = pd.Timestamp(e["time"])
        entry_price = float(e["entry_price"]) if "entry_price" in e else float(df.loc[ts, "close"]) if ts in df.index else None
        if entry_price is None:
            continue
        stop_price = float(e.get("stop_price", entry_price))
        target_price = float(e.get("target_price", entry_price))
        kind = e.get("kind", "orb")

        outcome = outcome_by_time.get(ts)
        if outcome is None:
            status = "open"
            r_text = "R=?"
        else:
            hit_t = outcome.get("hit_target", False)
            hit_s = outcome.get("hit_stop", False)
            status = "win" if hit_t else ("loss" if hit_s else "open")
            r_text = f"R={float(outcome.get("r_multiple", 0.0)):.2f}"

        ax.scatter([ts], [entry_price], color=COLOR_ENTRY[status], s=40, label=None)
        # plot stop/target lines for +/- 30 minutes
        left = ts - pd.Timedelta(minutes=30)
        right = ts + pd.Timedelta(minutes=30)
        ax.hlines(stop_price, xmin=left, xmax=right, colors="#e74c3c", linestyles="dashed", linewidth=1)
        ax.hlines(target_price, xmin=left, xmax=right, colors="#2ecc71", linestyles="dashed", linewidth=1)
        ax.text(ts, entry_price, f"{kind}\n{r_text}", fontsize=8, color="#2c3e50")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 1m candlesticks with trade overlays")
    parser.add_argument("--data-dir", type=str, required=True, help="Scenario folder with synthetic_1m.csv and summary.json")
    parser.add_argument("--out-img", type=str, default="trade_plot.png")
    args = parser.parse_args()

    df_1m, summary = load_scenario(args.data_dir)
    entries = summary.get("entries", [])
    outcomes = summary.get("outcomes", [])

    fig, ax = plt.subplots(figsize=(14, 6))
    plot_candles(ax, df_1m)
    overlay_trades(ax, df_1m, entries, outcomes)
    ax.set_title("1m Candles with Trade Overlays")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    fig.autofmt_xdate()
    out_path = os.path.join(args.data_dir, args.out_img)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()

```

---

## File: scripts/compare_real_vs_generator.py
```python
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

from src.core.generator import PriceGenerator, PhysicsConfig
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

```

---

## File: scripts/analyze_real_vs_synth.py
```python
"""Analyze Real vs Synthetic Data for ML-Style Diagnostics

This script computes side-by-side statistics for real vs synthetic MES data
across multiple timeframes and windows. It is intended to answer:

- Is the synthetic data "close enough" to real to train on?
- Do returns, ranges, and wick/body geometry look similar?
- How do volatility and regime structure compare across scales?

Usage (from project root, with .venv312 active):

    C:/fracfire/.venv312/Scripts/python.exe scripts/analyze_real_vs_synth.py \
        --week-days 14 \
        --timeframes 1min,4min,15min,1H,4H,1D

By default, the script uses the same PhysicsConfig defaults as the
price generator and the comparison harness.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.data.loader import RealDataLoader
from src.core.generator import PriceGenerator, PhysicsConfig
from scripts.compare_real_vs_generator import generate_synthetic_matching, pick_real_windows


@dataclass
class SeriesStats:
    timeframe: str
    kind: str  # "real" or "synth"
    n: int
    ret_mean: float
    ret_std: float
    ret_skew: float
    ret_kurt: float
    body_mean: float
    range_mean: float
    wick_frac_mean: float
    pct_wick_gt_0_6: float
    acf1: float
    acf1_abs: float


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe.lower() in ("1min", "1m", "1"):
        return df

    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df.resample(timeframe).agg(ohlc).dropna()


def compute_stats(df: pd.DataFrame, timeframe: str, kind: str) -> SeriesStats:
    # log returns at this timeframe
    ret = np.log(df["close"]).diff().dropna()
    if len(ret) < 3:
        # Degenerate window; fill with NaNs but keep n
        n = len(df)
        return SeriesStats(
            timeframe=timeframe,
            kind=kind,
            n=n,
            ret_mean=float("nan"),
            ret_std=float("nan"),
            ret_skew=float("nan"),
            ret_kurt=float("nan"),
            body_mean=float("nan"),
            range_mean=float("nan"),
            wick_frac_mean=float("nan"),
            pct_wick_gt_0_6=float("nan"),
            acf1=float("nan"),
            acf1_abs=float("nan"),
        )

    n = len(df)

    ret_mean = float(ret.mean())
    ret_std = float(ret.std())
    if ret_std == 0 or np.isnan(ret_std):
        ret_skew = float("nan")
        ret_kurt = float("nan")
    else:
        z = (ret - ret_mean) / ret_std
        ret_skew = float((z**3).mean())
        ret_kurt = float((z**4).mean() - 3.0)

    body = (df["close"] - df["open"]).abs()
    full = df["high"] - df["low"]
    full_safe = full.replace(0, pd.NA)
    wick_frac = 1.0 - (body / full_safe)
    wick_frac = wick_frac.fillna(0.0)

    body_mean = float(body.mean())
    range_mean = float(full.mean())
    wick_frac_mean = float(wick_frac.mean())
    pct_wick_gt_0_6 = float((wick_frac > 0.6).mean())

    # Autocorrelation of returns and abs returns at lag 1
    def acf1(x: pd.Series) -> float:
        if len(x) < 3:
            return float("nan")
        x = x - x.mean()
        c0 = float((x * x).mean())
        if c0 == 0:
            return float("nan")
        c1 = float((x.iloc[1:] * x.shift(1).iloc[1:]).mean())
        return c1 / c0

    acf_ret = acf1(ret)
    acf_abs = acf1(ret.abs())

    return SeriesStats(
        timeframe=timeframe,
        kind=kind,
        n=n,
        ret_mean=ret_mean,
        ret_std=ret_std,
        ret_skew=ret_skew,
        ret_kurt=ret_kurt,
        body_mean=body_mean,
        range_mean=range_mean,
        wick_frac_mean=wick_frac_mean,
        pct_wick_gt_0_6=pct_wick_gt_0_6,
        acf1=acf_ret,
        acf1_abs=acf_abs,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze real vs synthetic MES data across multiple timeframes.")
    parser.add_argument("--week-days", type=int, default=14, help="Number of days for the main window")
    parser.add_argument("--quarter-days", type=int, default=120, help="Number of days for the large window (used to pick a representative slice)")
    parser.add_argument(
        "--timeframes",
        type=str,
        default="1min,4min,15min,1H,4H,1D",
        help="Comma-separated list of timeframes to analyze",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic generation")

    args = parser.parse_args()

    print("=" * 60)
    print("ANALYZE REAL VS SYNTH (MULTI-TIMEFRAME)")
    print("=" * 60)

    loader = RealDataLoader()
    real_path = root / "src" / "data" / "continuous_contract.json"
    real_1m = loader.load_json(real_path)

    real_1m_ohlcv = real_1m[["open", "high", "low", "close", "volume"]]
    real_week_1m, _ = pick_real_windows(real_1m_ohlcv, week_days=args.week_days, quarter_days=args.quarter_days)

    # Generate synthetic aligned to the week window
    synth_week_1m = generate_synthetic_matching(real_week_1m, seed=args.seed, physics_overrides=None)

    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]

    rows: List[SeriesStats] = []

    for tf in timeframes:
        real_tf = resample(real_week_1m, tf)
        synth_tf = resample(synth_week_1m, tf)

        stats_real = compute_stats(real_tf, tf, kind="real")
        stats_synth = compute_stats(synth_tf, tf, kind="synth")

        rows.append(stats_real)
        rows.append(stats_synth)

    # Convert to DataFrame for pretty printing
    data = [r.__dict__ for r in rows]
    df_stats = pd.DataFrame(data)

    # Pivot into a compact comparison table: rows = timeframe, cols = metric_real/synth
    metrics = [
        "ret_mean",
        "ret_std",
        "ret_skew",
        "ret_kurt",
        "body_mean",
        "range_mean",
        "wick_frac_mean",
        "pct_wick_gt_0_6",
        "acf1",
        "acf1_abs",
    ]

    tables: Dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        sub = df_stats[df_stats["timeframe"] == tf].set_index("kind")
        if sub.empty:
            continue
        tbl = sub[metrics].T
        tbl.columns = [f"{c.upper()}" for c in tbl.columns]
        tables[tf] = tbl

    for tf, tbl in tables.items():
        print("\n" + "-" * 60)
        print(f"TIMEFRAME: {tf}")
        print("-" * 60)
        print(tbl.to_string(float_format=lambda x: f"{x: .4e}"))

    print("\nDone.")


if __name__ == "__main__":
    main()

```

---

