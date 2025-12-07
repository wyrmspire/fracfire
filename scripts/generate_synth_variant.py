"""Generate synthetic 1m data with configurable physics options.

Usage:
  source /C/fracfire/.venv312/Scripts/activate
  /C/fracfire/.venv312/Scripts/python.exe scripts/generate_synth_variant.py --months 3 --seed 123 \
      --out-dir out/synth_old_3mo

This script mirrors `generate_6m_synth_and_plot_4h.py` but exposes
flags to enable fat tails and volatility clustering for
easy side-by-side comparisons.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.generator.engine import PhysicsConfig, PriceGenerator


def generate_synth_1m(days: int, seed: Optional[int], physics_cfg: PhysicsConfig) -> pd.DataFrame:
    gen = PriceGenerator(physics_config=physics_cfg, seed=seed)
    all_days = []
    current = pd.Timestamp.utcnow().normalize()
    for i in range(days):
        day_date = current - pd.Timedelta(days=(days - 1 - i))
        df_day = gen.generate_day(day_date)
        all_days.append(df_day)
    df = pd.concat(all_days).sort_values("time")
    df = df.set_index("time")
    return df


def resample_4h(df_1m: pd.DataFrame) -> pd.DataFrame:
    ohlc = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df_1m.resample("4H").agg(ohlc).dropna()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", type=str, default=os.path.join(PROJECT_ROOT, "out", "synth_variant"))
    parser.add_argument("--fat-tails", dest="fat_tails", action="store_true", help="Enable Student-t fat tails")
    parser.add_argument("--fat-df", type=float, default=3.0, help="Degrees of freedom for Student-t")
    parser.add_argument("--vol-cluster", dest="vol_cluster", action="store_true", help="Enable volatility clustering (GARCH-like)")
    parser.add_argument("--vol-persist", type=float, default=0.3, help="Volatility persistence (0-1)")
    args = parser.parse_args()

    days = int(args.months * 30)
    print(f"Generating ~{days} days ({args.months} months) of 1m synthetic data with seed={args.seed}...")

    cfg = PhysicsConfig()
    cfg.use_fat_tails = bool(args.fat_tails)
    cfg.fat_tail_df = float(args.fat_df)
    cfg.use_volatility_clustering = bool(args.vol_cluster)
    cfg.volatility_persistence = float(args.vol_persist)

    df_1m = generate_synth_1m(days=days, seed=args.seed, physics_cfg=cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    p1 = os.path.join(args.out_dir, "synth_1m.parquet")
    df_1m.to_parquet(p1)
    print("Saved 1m parquet:", p1)

    df_4h = resample_4h(df_1m)
    p4 = os.path.join(args.out_dir, "synth_4h.parquet")
    df_4h.to_parquet(p4)
    print("Saved 4h parquet:", p4)


if __name__ == "__main__":
    main()
