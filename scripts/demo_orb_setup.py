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

from lab.generators.price_generator import PhysicsConfig, PriceGenerator
from src.evaluation.setups import (
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
