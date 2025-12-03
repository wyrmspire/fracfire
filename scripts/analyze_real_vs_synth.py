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
from lab.generators.price_generator import PriceGenerator, PhysicsConfig
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
