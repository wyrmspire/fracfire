# Setup Dump: 02_detector_scripts

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
import sys
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.visualization import (
    COLORS,
    add_trade_zones_matplotlib,
    add_trade_lines_matplotlib,
    get_exit_time_or_fallback,
    get_entry_marker_color,
)


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
        color = COLORS["candle_up"] if c >= o else COLORS["candle_down"]
        # wick
        ax.plot([ts, ts], [l, h], color=color, linewidth=1)
        # body
        ax.plot([ts, ts], [o, c], color="#34495e", linewidth=4)


def overlay_trades(ax: plt.Axes, df: pd.DataFrame, entries: List[Dict[str, Any]], outcomes: List[Dict[str, Any]]) -> None:
    # Map entry times to outcomes
    outcome_by_time: Dict[pd.Timestamp, Dict[str, Any]] = {}
    for o in outcomes:
        outcome_by_time[pd.Timestamp(o["entry"]["time"])] = o

    for e in entries:
        entry_ts = pd.Timestamp(e["time"])
        entry_price = float(e["entry_price"]) if "entry_price" in e else float(df.loc[entry_ts, "close"]) if entry_ts in df.index else None
        if entry_price is None:
            continue
        stop_price = float(e.get("stop_price", entry_price))
        target_price = float(e.get("target_price", entry_price))
        kind = e.get("kind", "orb")

        outcome = outcome_by_time.get(entry_ts)
        if outcome is None:
            is_open = True
            r_multiple = 0.0
            r_text = "R=?"
            exit_time_value = None
        else:
            hit_t = outcome.get("hit_target", False)
            hit_s = outcome.get("hit_stop", False)
            is_open = not (hit_t or hit_s)
            r_multiple = float(outcome.get('r_multiple', 0.0))
            r_text = f"R={r_multiple:.2f}"
            exit_time_value = outcome.get("exit_time")
        
        # Get exit time with fallback
        exit_ts = get_exit_time_or_fallback(entry_ts, exit_time_value)

        # Entry marker with appropriate color
        marker_color = get_entry_marker_color(r_multiple, is_open)
        ax.scatter([entry_ts], [entry_price], color=marker_color, s=40, label=None, zorder=5)
        
        # Add shaded zones and lines using shared utilities
        add_trade_zones_matplotlib(ax, entry_ts, exit_ts, entry_price, stop_price, target_price, zorder=1)
        add_trade_lines_matplotlib(ax, entry_ts, exit_ts, stop_price, target_price, zorder=3)
        
        ax.text(entry_ts, entry_price, f"{kind}\n{r_text}", fontsize=8, color="#2c3e50", zorder=6)


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

## File: scripts/demo_all_setups.py
```python
"""Demo script for all setup types including new indicators and setups.

Generates synthetic data using PriceGenerator, resamples to 5m,
applies all indicators, and detects all setup types (ORB, EMA200 continuation,
breakout, reversal, opening push, MOC).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

import pandas as pd

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.generator.engine import PhysicsConfig, PriceGenerator
from src.core.detector.indicators import IndicatorConfig, add_5m_indicators
from src.core.detector.library import (
    find_opening_orb_continuations,
    evaluate_orb_entry,
    ORBConfig,
    find_ema200_continuation,
    EMA200ContinuationConfig,
    find_breakout,
    BreakoutConfig,
    find_reversal,
    ReversalConfig,
    find_opening_push,
    OpeningPushConfig,
    find_moc,
    MOCConfig,
)
from src.core.detector.features import evaluate_generic_entry_1m


def generate_synth_1m(days: int, seed: int) -> pd.DataFrame:
    """Generate synthetic 1m data."""
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
    """Resample 1m data to 5m bars."""
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df_5m = df_1m.resample("5min").agg(ohlc).dropna()
    return df_5m


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo all setup types on synthetic data"
    )
    parser.add_argument(
        "--days", type=int, default=5, help="Number of synthetic days to generate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generator")
    args = parser.parse_args()

    print("=" * 70)
    print("FRACFIRE - ALL SETUPS DEMO")
    print("=" * 70)
    print()

    # Generate data
    print(f"[1/6] Generating {args.days} days of synthetic 1m data...")
    df_1m = generate_synth_1m(days=args.days, seed=args.seed)
    df_5m = resample_5m(df_1m)
    print(f"  Generated {len(df_1m)} 1m bars, {len(df_5m)} 5m bars")
    print()

    # Add indicators
    print("[2/6] Computing indicators (EMA, RSI, BB, ATR, MACD, Volume)...")
    ind_cfg = IndicatorConfig()
    df_5m_ind = add_5m_indicators(df_5m, ind_cfg)
    print(f"  Added {len(df_5m_ind.columns)} indicator columns")
    print(f"  Columns: {', '.join(df_5m_ind.columns)}")
    print()

    # Setup configs
    orb_cfg = ORBConfig()
    ema200_cfg = EMA200ContinuationConfig()
    breakout_cfg = BreakoutConfig()
    reversal_cfg = ReversalConfig()
    opening_push_cfg = OpeningPushConfig()
    moc_cfg = MOCConfig()

    all_entries = []
    all_outcomes = []

    # ORB
    print("[3/6] Finding Opening Range Breakout (ORB) setups...")
    orb_entries = find_opening_orb_continuations(df_5m_ind, orb_cfg)
    print(f"  Found {len(orb_entries)} ORB entries")
    for entry in orb_entries:
        outcome = evaluate_orb_entry(df_5m_ind, entry, orb_cfg)
        if outcome:
            all_outcomes.append(outcome)
            print(
                f"    {entry.direction.upper()} @ {entry.time} | "
                f"Entry: {entry.entry_price:.2f} | R: {outcome.r_multiple:.2f}"
            )
    all_entries.extend(orb_entries)
    print()

    # EMA200 Continuation
    print("[4/6] Finding EMA200 Continuation setups...")
    ema200_entries = find_ema200_continuation(df_5m_ind, ema200_cfg)
    print(f"  Found {len(ema200_entries)} EMA200 continuation entries")
    for entry in ema200_entries:
        outcome = evaluate_generic_entry_1m(df_1m, entry, max_minutes=240)
        all_outcomes.append(outcome)
        print(
            f"    {entry.direction.upper()} @ {entry.time} | "
            f"Entry: {entry.entry_price:.2f} | RSI: {entry.context.get('rsi', 0):.1f} | "
            f"R: {outcome.r_multiple:.2f}"
        )
    all_entries.extend(ema200_entries)
    print()

    # Breakout
    print("[5/6] Finding Breakout setups...")
    breakout_entries = find_breakout(df_5m_ind, breakout_cfg)
    print(f"  Found {len(breakout_entries)} breakout entries")
    for entry in breakout_entries:
        outcome = evaluate_generic_entry_1m(df_1m, entry, max_minutes=240)
        all_outcomes.append(outcome)
        print(
            f"    {entry.direction.upper()} @ {entry.time} | "
            f"Entry: {entry.entry_price:.2f} | Level: {entry.context.get('breakout_level', 0):.2f} | "
            f"R: {outcome.r_multiple:.2f}"
        )
    all_entries.extend(breakout_entries)
    print()

    # Reversal
    print("[6/6] Finding Reversal setups...")
    reversal_entries = find_reversal(df_5m_ind, reversal_cfg)
    print(f"  Found {len(reversal_entries)} reversal entries")
    for entry in reversal_entries:
        outcome = evaluate_generic_entry_1m(df_1m, entry, max_minutes=180)
        all_outcomes.append(outcome)
        print(
            f"    {entry.direction.upper()} @ {entry.time} | "
            f"Entry: {entry.entry_price:.2f} | RSI: {entry.context.get('rsi', 0):.1f} | "
            f"R: {outcome.r_multiple:.2f}"
        )
    all_entries.extend(reversal_entries)
    print()

    # Opening Push
    print("[7/8] Finding Opening Push setups...")
    opening_push_entries = find_opening_push(df_5m_ind, opening_push_cfg)
    print(f"  Found {len(opening_push_entries)} opening push entries")
    for entry in opening_push_entries:
        outcome = evaluate_generic_entry_1m(df_1m, entry, max_minutes=240)
        all_outcomes.append(outcome)
        print(
            f"    {entry.direction.upper()} @ {entry.time} | "
            f"Entry: {entry.entry_price:.2f} | Move: {entry.context.get('move_ticks', 0):.1f} ticks | "
            f"R: {outcome.r_multiple:.2f}"
        )
    all_entries.extend(opening_push_entries)
    print()

    # MOC
    print("[8/8] Finding Market on Close (MOC) setups...")
    moc_entries = find_moc(df_5m_ind, moc_cfg)
    print(f"  Found {len(moc_entries)} MOC entries")
    for entry in moc_entries:
        outcome = evaluate_generic_entry_1m(df_1m, entry, max_minutes=120)
        all_outcomes.append(outcome)
        bars_key = "up_bars" if entry.direction == "long" else "down_bars"
        print(
            f"    {entry.direction.upper()} @ {entry.time} | "
            f"Entry: {entry.entry_price:.2f} | Bars: {entry.context.get(bars_key, 0)} | "
            f"R: {outcome.r_multiple:.2f}"
        )
    all_entries.extend(moc_entries)
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total entries found: {len(all_entries)}")
    print(f"Total outcomes evaluated: {len(all_outcomes)}")
    print()

    if all_outcomes:
        winning_trades = sum(1 for o in all_outcomes if o.r_multiple > 0)
        losing_trades = sum(1 for o in all_outcomes if o.r_multiple < 0)
        avg_r = sum(o.r_multiple for o in all_outcomes) / len(all_outcomes)
        
        print(f"Winning trades: {winning_trades} ({winning_trades/len(all_outcomes)*100:.1f}%)")
        print(f"Losing trades: {losing_trades} ({losing_trades/len(all_outcomes)*100:.1f}%)")
        print(f"Average R-multiple: {avg_r:.2f}")
        print()

        # Breakdown by setup type
        print("Breakdown by setup type:")
        setup_types = set(e.kind for e in all_entries)
        for setup_type in sorted(setup_types):
            type_outcomes = [o for o in all_outcomes if o.entry.kind == setup_type]
            if type_outcomes:
                type_wins = sum(1 for o in type_outcomes if o.r_multiple > 0)
                type_avg_r = sum(o.r_multiple for o in type_outcomes) / len(type_outcomes)
                print(f"  {setup_type:20s}: {len(type_outcomes):2d} trades | "
                      f"{type_wins:2d} wins | Avg R: {type_avg_r:5.2f}")
    else:
        print("No trades found on this dataset.")

    print()
    print("=" * 70)
    print("Demo complete!")
    print()


if __name__ == "__main__":
    main()

```

---

## File: scripts/benchmark_detection.py
```python
import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.core.detector.engine import run_setups, SetupConfig, SetupFamilyConfig
from src.core.detector.indicators import IndicatorConfig

def benchmark_detection():
    # 1. Load Data
    csv_path = root / 'out' / 'synthetic_year.csv'
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run benchmark_generation.py first.")
        return

    print(f"Loading {csv_path}...")
    t0 = time.time()
    df_1m = pd.read_csv(csv_path)
    df_1m['time'] = pd.to_datetime(df_1m['time'])
    df_1m.set_index('time', inplace=True)
    t1 = time.time()
    print(f"Loaded {len(df_1m)} rows in {t1-t0:.4f}s")

    # 2. Resample to 5m
    print("Resampling to 5m...")
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df_5m = df_1m.resample('5min').agg(ohlc).dropna()
    t2 = time.time()
    print(f"Resampled to {len(df_5m)} 5m bars in {t2-t1:.4f}s")

    # 3. Configure Setups
    # Enable multiple families to test throughput
    cfg = SetupConfig(
        indicator_cfg=IndicatorConfig(
            ema_fast=20,
            ema_slow=200,
            rsi_period=14,
            atr_period=14
        ),
        orb=SetupFamilyConfig(enabled=True),
        ema200_continuation=SetupFamilyConfig(enabled=True),
        breakout=SetupFamilyConfig(enabled=True),
        reversal=SetupFamilyConfig(enabled=True),
        opening_push=SetupFamilyConfig(enabled=True),
        moc=SetupFamilyConfig(enabled=True)
    )

    # 4. Run Detection
    print("Running detection on 1 year of data...")
    t3 = time.time()
    outcomes = run_setups(df_1m, df_5m, cfg)
    t4 = time.time()
    
    detection_time = t4 - t3
    print(f"Detection Time: {detection_time:.4f}s")
    print(f"Found {len(outcomes)} setups")
    
    # Per-day metrics
    n_days = len(df_5m) / (12 * 24) # approx
    print(f"Speed: {detection_time / n_days * 1000:.2f} ms per day")

if __name__ == "__main__":
    benchmark_detection()

```

---

## File: scripts/demo_orb_setup.py
```python
"""Demo script for Opening Range Breakout (ORB) continuation setup.

Generates synthetic data using the existing PriceGenerator, resamples to 5m,
then finds and evaluates ORB continuation trades using `src/evaluation/setups.py`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import os
import sys

import pandas as pd


# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.generator.engine import PhysicsConfig, PriceGenerator
from src.core.detector.library import (
    ORBConfig,
    evaluate_orb_entry,
    find_opening_orb_continuations,
)


def generate_synth_1m(days: int, seed: int) -> pd.DataFrame:
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
    df_5m = df_1m.resample("5min").agg(ohlc).dropna()
    return df_5m


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo ORB continuation setup on synthetic data")
    parser.add_argument("--days", type=int, default=5, help="Number of synthetic days to generate")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for generator")
    args = parser.parse_args()

    df_1m = generate_synth_1m(days=args.days, seed=args.seed)
    df_5m = resample_5m(df_1m)

    orb_cfg = ORBConfig()
    entries = find_opening_orb_continuations(df_5m, orb_cfg)

    print(f"Generated {len(df_5m)} 5m bars from {args.days} synthetic days")
    print(f"Found {len(entries)} ORB continuation entries")

    for i, entry in enumerate(entries):
        outcome = evaluate_orb_entry(df_5m, entry, orb_cfg)
        print(f"\nEntry {i}:")
        print({k: v for k, v in asdict(entry).items() if k not in {"or_start", "or_end"}})
        if outcome is None:
            print("  Outcome: insufficient future data")
        else:
            print(
                f"  Outcome: hit_target={outcome.hit_target}, hit_stop={outcome.hit_stop}, "
                f"exit={outcome.exit_time}, R={outcome.r_multiple:.2f}"
            )


if __name__ == "__main__":
    main()

```

---

