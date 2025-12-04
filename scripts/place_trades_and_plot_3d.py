"""
Place trades from several setup families on selected time windows and
plot 3D visualizations (time, price, cumulative P&L) using 5m candles.

This script demonstrates the pipeline: indicators -> detectors -> evaluation -> P&L

Outputs:
 - PNG 3D plots under `out/trades_3d/`
 - CSV summary of trades and cumulative P&L under `out/trades_3d/`

Usage (examples):
 python scripts/place_trades_and_plot_3d.py --risk 100 --start-cap 2000 --windows 7,30,90,180

"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Project imports
import sys
from pathlib import Path

# Ensure project root is on sys.path for `src` imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import RealDataLoader
from src.core.detector import (
    add_5m_indicators,
    IndicatorConfig,
)
from src.core.detector.library import (
    find_opening_orb_continuations,
    evaluate_orb_entry,
    find_ema200_continuation,
    find_breakout,
    find_reversal,
    find_opening_push,
    find_moc,
)
from src.core.detector.library import ORBConfig
from src.core.detector.library import EMA200ContinuationConfig, BreakoutConfig, ReversalConfig, OpeningPushConfig, MOCConfig
from src.core.detector.library import ORBEntry
from src.core.detector.library import ORBOutcome
from src.core.detector.library import evaluate_orb_entry_1m
from src.core.detector.indicators import IndicatorConfig as ICfg
from src.core.detector.library import evaluate_generic_entry_1m


def resample_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df_1m.resample("5min").agg(ohlc).dropna()


def pick_windows_from_real(real_1m: pd.DataFrame, days_list: List[int]) -> Dict[int, pd.DataFrame]:
    """Pick contiguous windows of the given day lengths from real data.
    If data is large, pick random non-overlapping windows by sampling start indices.
    Returns a dict days->df_1m slice
    """
    windows = {}
    total_bars = len(real_1m)
    for days in days_list:
        bars = days * 24 * 60
        if bars >= total_bars:
            # fallback: use the whole series
            windows[days] = real_1m.copy()
            continue
        max_start = total_bars - bars
        start_idx = np.random.randint(0, max(1, max_start))
        df_slice = real_1m.iloc[start_idx : start_idx + bars].copy()
        windows[days] = df_slice
    return windows


def detect_setups_and_evaluate(df_5m: pd.DataFrame, df_1m: pd.DataFrame) -> List[Dict[str, Any]]:
    """Run multiple setup finders on the provided 5m frame and evaluate outcomes.
    Returns list of trade dicts with r_multiple and timestamps/prices.
    """
    trades: List[Dict[str, Any]] = []

    # Ensure indicators exist on 5m
    ind_cfg = IndicatorConfig()
    df_5m_ind = add_5m_indicators(df_5m, ind_cfg)

    # 1) ORB (uses 5m)
    orb_cfg = ORBConfig()
    orb_entries = find_opening_orb_continuations(df_5m_ind, orb_cfg)
    for e in orb_entries:
        out = evaluate_orb_entry(df_5m_ind, e, orb_cfg)
        if out is None:
            continue
        trades.append({
            "kind": "orb",
            "time": out.entry.time,
            "entry_price": out.entry.entry_price,
            "stop_price": out.entry.stop_price,
            "target_price": out.entry.target_price,
            "r_multiple": float(out.r_multiple),
        })

    # 2) EMA200 continuation
    ema_entries = find_ema200_continuation(df_5m_ind)
    for e in ema_entries:
        out = evaluate_generic_entry_1m(df_1m, e)
        trades.append({
            "kind": "ema200",
            "time": out.entry.time,
            "entry_price": out.entry.entry_price,
            "stop_price": out.entry.stop_price,
            "target_price": out.entry.target_price,
            "r_multiple": float(out.r_multiple),
        })

    # 3) Breakout
    breakout_entries = find_breakout(df_5m_ind)
    for e in breakout_entries:
        out = evaluate_generic_entry_1m(df_1m, e)
        trades.append({
            "kind": "breakout",
            "time": out.entry.time,
            "entry_price": out.entry.entry_price,
            "stop_price": out.entry.stop_price,
            "target_price": out.entry.target_price,
            "r_multiple": float(out.r_multiple),
        })

    # 4) Reversal
    rev_entries = find_reversal(df_5m_ind)
    for e in rev_entries:
        out = evaluate_generic_entry_1m(df_1m, e)
        trades.append({
            "kind": "reversal",
            "time": out.entry.time,
            "entry_price": out.entry.entry_price,
            "stop_price": out.entry.stop_price,
            "target_price": out.entry.target_price,
            "r_multiple": float(out.r_multiple),
        })

    # 5) Opening push
    push_entries = find_opening_push(df_5m_ind)
    for e in push_entries:
        out = evaluate_generic_entry_1m(df_1m, e)
        trades.append({
            "kind": "opening_push",
            "time": out.entry.time,
            "entry_price": out.entry.entry_price,
            "stop_price": out.entry.stop_price,
            "target_price": out.entry.target_price,
            "r_multiple": float(out.r_multiple),
        })

    # 6) MOC
    moc_entries = find_moc(df_5m_ind)
    for e in moc_entries:
        out = evaluate_generic_entry_1m(df_1m, e)
        trades.append({
            "kind": "moc",
            "time": out.entry.time,
            "entry_price": out.entry.entry_price,
            "stop_price": out.entry.stop_price,
            "target_price": out.entry.target_price,
            "r_multiple": float(out.r_multiple),
        })

    # Sort trades by time
    trades.sort(key=lambda x: pd.Timestamp(x["time"]).to_datetime64())
    return trades


def plot_3d_trades(df_5m: pd.DataFrame, trades: List[Dict[str, Any]], out_path: Path, start_cap: float, risk_per_trade: float) -> None:
    """Create a 3D plot: x=time, y=price, z=cumulative P&L. Candles plotted in x-y plane at z=0.
    Trades plotted as markers at their entry time/price with z equal to cumulative P&L after applying trade.
    """
    if df_5m.empty:
        print("No 5m data to plot")
        return

    times = mdates.date2num(df_5m.index.to_pydatetime())
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot candles as vertical lines at z=0
    xs = times
    ys_low = df_5m["low"].values
    ys_high = df_5m["high"].values
    for x, low, high in zip(xs, ys_low, ys_high):
        ax.plot([x, x], [low, low], [0, high - low], color="#888888", alpha=0.6)
    # Plot close price curve (projected)
    ax.plot(xs, df_5m["close"].values, zs=0, zdir="z", color="#222222", linewidth=1.0)

    # Compute cumulative P&L across trades
    cum_pl = start_cap
    trade_z = []
    trade_x = []
    trade_y = []
    trade_colors = []
    trade_texts = []
    for t in trades:
        r = float(t["r_multiple"]) if t["r_multiple"] is not None else 0.0
        profit = r * float(risk_per_trade)
        cum_pl += profit
        trade_time = pd.Timestamp(t["time"]).to_pydatetime()
        trade_x.append(mdates.date2num(trade_time))
        trade_y.append(float(t["entry_price"]))
        trade_z.append(cum_pl)
        trade_colors.append("g" if profit >= 0 else "r")
        trade_texts.append(f"{t['kind']} R={r:.2f} P=${profit:.2f} CP=${cum_pl:.2f}")

    if trade_x:
        ax.scatter(trade_x, trade_y, trade_z, c=trade_colors, s=40)
        # annotate
        for x, y, z, txt in zip(trade_x, trade_y, trade_z, trade_texts):
            ax.text(x, y, z, txt, fontsize=8)

    # Formatting
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_zlabel("Cumulative P&L ($)")
    ax.set_title(f"Trades (risk=${risk_per_trade}) - {len(trades)} trades")

    # X-axis date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    # rotate for readability
    fig.autofmt_xdate()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D plot to {out_path}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Place trades with setups and plot 3D visualization on 5m candles")
    parser.add_argument("--risk", type=float, default=100.0, help="Dollar risk per trade")
    parser.add_argument("--start-cap", type=float, default=2000.0, help="Starting P&L/capital")
    parser.add_argument("--windows", type=str, default="7,30,90", help="Comma-separated day windows to sample (e.g., 7,30,90)")
    parser.add_argument("--out-dir", type=str, default="out/trades_3d", help="Output directory for plots and CSV")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    np.random.seed(args.seed)

    # Load real 1m
    loader = RealDataLoader()
    root = Path(__file__).resolve().parents[1]
    real_path = root / "src" / "data" / "continuous_contract.json"
    real_1m = loader.load_json(real_path)
    real_1m = real_1m[ ["open", "high", "low", "close", "volume"] ]

    days_list = [int(x) for x in args.windows.split(",") if x.strip()]
    windows = pick_windows_from_real(real_1m, days_list)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_trades_records: List[Dict[str, Any]] = []

    for days, df_1m in windows.items():
        print(f"Processing window {days} days -> {len(df_1m)} 1m bars")
        df_5m = resample_5m(df_1m)
        # Add indicators and detect/evaluate trades
        trades = detect_setups_and_evaluate(df_5m, df_1m)

        # Compute P&L sequence and save CSV per window
        cum = args.start_cap
        rows = []
        for i, t in enumerate(trades):
            r = float(t.get("r_multiple", 0.0))
            pnl = r * float(args.risk)
            cum += pnl
            rows.append({
                "window_days": days,
                "trade_idx": i,
                "kind": t["kind"],
                "time": str(t["time"]),
                "entry_price": t["entry_price"],
                "stop_price": t["stop_price"],
                "target_price": t["target_price"],
                "r_multiple": r,
                "pnl": pnl,
                "cum_pnl": cum,
            })
            all_trades_records.append(rows[-1])

        # Plot 3D for this window
        plot_file = out_dir / f"trades_3d_{days}d.png"
        plot_3d_trades(df_5m, trades, plot_file, args.start_cap, args.risk)

        # Save CSV per window
        csv_file = out_dir / f"trades_{days}d.csv"
        pd.DataFrame(rows).to_csv(csv_file, index=False)
        print(f"Saved trades CSV to {csv_file}")

    # Save aggregate CSV
    agg_csv = out_dir / "trades_all_windows.csv"
    pd.DataFrame(all_trades_records).to_csv(agg_csv, index=False)
    print(f"Saved aggregate trades CSV to {agg_csv}")


if __name__ == "__main__":
    main()
