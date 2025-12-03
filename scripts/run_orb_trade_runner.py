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

from lab.generators.price_generator import PhysicsConfig, PriceGenerator
from src.evaluation.setups import (
    ORBConfig,
    ORBOutcome,
    summarize_orb_day,
)


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

        entries, outcomes, miss = summarize_orb_day(df_5m, orb_cfg)
        last_miss = miss

        if entries:
            # For now we just take all entries in this sample; the agent can pick.
            return {
                "found": True,
                "iteration": i,
                "seed": seed,
                "entries": [asdict(e) for e in entries],
                "outcomes": [asdict(o) for o in outcomes],
                "miss": asdict(miss),
                "df_1m": df_1m,
                "df_5m": df_5m,
                "summary": f"found {len(entries)} entries on iteration {i} seed {seed}",
            }

        # Simple auto-relax: if drive too weak, lower min_drive_rr slightly
        if miss.up_rr < orb_cfg.min_drive_rr and miss.down_rr < orb_cfg.min_drive_rr:
            orb_cfg.min_drive_rr = max(0.1, orb_cfg.min_drive_rr * 0.9)
        # If counter too big, relax that too
        if miss.down_rr > orb_cfg.max_counter_rr or miss.up_rr > orb_cfg.max_counter_rr:
            orb_cfg.max_counter_rr = min(1.0, orb_cfg.max_counter_rr * 1.1)

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

    # Save JSON summary
    json_summary: Dict[str, Any] = {
        k: v for k, v in result.items() if not isinstance(v, (pd.DataFrame,))
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, default=str)

    print(json_summary["summary"])


if __name__ == "__main__":
    main()
