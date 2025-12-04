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
