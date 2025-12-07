"""Generate ~6 months of 1m synthetic data and plot the 4H resample.

Usage:
  source /C/fracfire/.venv312/Scripts/activate
  /C/fracfire/.venv312/Scripts/python.exe scripts/generate_6m_synth_and_plot_4h.py --months 6 --seed 123 --out-dir out/synth_6m

Saves:
 - `out/synth_6m/synth_1m.parquet`
 - `out/synth_6m/synth_4h.parquet`
 - `out/synth_6m/synth_4h.png`
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.generator.engine import PhysicsConfig, PriceGenerator


def generate_synth_1m(days: int, seed: Optional[int] = None) -> pd.DataFrame:
    cfg = PhysicsConfig()
    gen = PriceGenerator(physics_config=cfg, seed=seed)
    all_days = []
    # generate recent days ending today (UTC)
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


def plot_4h(df_4h: pd.DataFrame, out_path: str, title: str = "4H Synth") -> None:
    fig, ax = plt.subplots(figsize=(18, 7))
    for ts, row in df_4h.iterrows():
        o = float(row["open"]) ; c = float(row["close"]) ; h = float(row["high"]) ; l = float(row["low"]) 
        color = "#26a69a" if c >= o else "#ef5350"
        ax.vlines(ts, l, h, color="#666", linewidth=1)
        ax.vlines(ts, min(o, c), max(o, c), color=color, linewidth=6)
    ax.set_title(title)
    ax.set_ylabel("Price")
    fig.autofmt_xdate()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", type=str, default=os.path.join(PROJECT_ROOT, "out", "synth_6m"))
    args = parser.parse_args()

    days = int(args.months * 30)
    print(f"Generating ~{days} days ({args.months} months) of 1m synthetic data with seed={args.seed}...")
    df_1m = generate_synth_1m(days=days, seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    p1 = os.path.join(args.out_dir, "synth_1m.parquet")
    df_1m.to_parquet(p1)
    print("Saved 1m parquet:", p1)

    df_4h = resample_4h(df_1m)
    p4 = os.path.join(args.out_dir, "synth_4h.parquet")
    df_4h.to_parquet(p4)
    print("Saved 4h parquet:", p4)

    img = os.path.join(args.out_dir, "synth_4h.png")
    plot_4h(df_4h, img, title=f"Synthetic {args.months}mo | 4H resample | seed={args.seed}")
    print("Saved 4h chart:", img)


if __name__ == "__main__":
    main()
